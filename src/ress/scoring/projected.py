"""The residual-drift score: how far dropping a key block moves what reaches the residual
stream.

Score a block by P_ij * ||W_O^h (v̄_i - V_j)||, not by the deviation measured in head
space. The distinction is the point of the method: a head's output passes through W_O
before it is added to the residual stream, so two deviations of equal head-space norm can
contribute very differently once projected. Ranking by the head-space norm optimises the
wrong quantity.

Computing it does not require forming W_O^h e. With the Gram matrix

    M_h = (W_O^h)^T W_O^h        (hd x hd)

the squared projected norm is the quadratic form e^T M_h e. M_h depends only on the
weights, so it is baked once per model load (`bake_and_attach_M`) and reused for every
scene. At hd = 64 that quadratic form is already cheap; no low-rank approximation of M_h
is implemented, and none should be added before profiling asks for one.
"""

import contextlib
import os

import torch

from ress.models.wiring import global_blocks
from ress.scoring.base import BlockScorer

# Kill-switch for the Triton top-m kernel `_top_m_mask` dispatches to --
# see that function's docstring. Read once at import; set RESS_FUSED_TOPM=0 to force
# the `torch.topk` path (A/B timing, or a GPU/Triton build where the kernel misbehaves).
_FUSED_TOPM = os.environ.get("RESS_FUSED_TOPM", "1") != "0"
# Same for the whole-round kernel (`scoring.fused_round`), which subsumes the top-m one
# on the configurations it covers. RESS_FUSED_ROUND=0 forces the eager round body.
_FUSED_ROUND = os.environ.get("RESS_FUSED_ROUND", "1") != "0"

# Diagnostic-only, opt-in instrumentation, for answering whether the drift actually
# shrinks round over round. None in production; set by
# `record_restore_swap_drift()` around a forward pass. Module-level rather than a
# `restore_swap_rank_by` kwarg so `models.attention.ress_attention_forward`'s call
# site needs no changes -- zero effect on any existing behavior when unset.
_diag_records = None


@contextlib.contextmanager
def record_restore_swap_drift():
    """Diagnostic-only: while
    active, every `restore_swap_rank_by` call appends one list to the yielded
    `records` (one entry per call == one per global-attention layer, in forward
    order), each containing `n_iters` scalars -- the mean (over all batch/head/
    query-block entries) `||g||^2_M` measured at the START of each round, i.e.
    BEFORE that round's restore/drop decision. Comparing round i to round i+1
    shows whether round i's restore actually shrunk the drift round i+1 starts
    from. Purely additive: no effect on `rank_by`'s returned value or any other
    behavior when not active."""
    global _diag_records
    prev = _diag_records
    records = []
    _diag_records = records
    try:
        yield records
    finally:
        _diag_records = prev


# Same pattern as `_diag_records`/`record_restore_swap_drift`, mirrored for
# `pooled_restore_swap_select` (does the pooled
# cross-head competition's per-round total ||g||^2_M actually converges, the same
# question `record_restore_swap_drift` already answered for the per-head version --
# HessBudget's own per-head allocation turned out FAR more skewed than pooled's
# dynamic one yet works fine, so the suspect shifted from "unequal budget" to
# "iterative/myopic reallocation process").




# Same opt-in pattern, separate list so this doesn't disturb `_diag_records`'
# fixed 4-scalars-per-round shape (the diagnostic consumer
# reshapes it straight into a `(n_layers, n_iters, 4)` tensor -- adding fields
# there would silently corrupt that reshape instead of erroring).
_thrash_diag_records = None


# Same opt-in pattern again, for the FINAL realized objective rather than the
# per-round trace (the drift-maximization sanity check): every arm of that
# experiment -- Ours, both anti arms, and `random`, which never calls
# `restore_swap_rank_by` at all -- has to be placed on ONE x-axis, the objective the
# selection actually left behind. `_diag_records` cannot serve: it is restore-swap-only
# and reports the state BEFORE each round's decision, so the shipped `track_best` pick is
# never in it. This one is filled from `models.attention` after `final_map` is built, so
# it measures whatever the criterion in force actually selected.
_realized_records = None


@contextlib.contextmanager
def record_realized_drift():
    """Diagnostic-only: while active, `models.attention.ress_attention_forward` appends
    one float per global-attention layer (forward order) to the yielded list -- the mean
    over all (batch, head, query-block) rows of `J = ||g||^2_M / K^2` for the kept set
    that layer ACTUALLY ran with, sinks included. Criterion-agnostic (see
    `realized_objective`), so arms that share no code still land on the same scale.
    No effect on any returned value when not active."""
    global _realized_records
    prev = _realized_records
    records = []
    _realized_records = records
    try:
        yield records
    finally:
        _realized_records = prev


def realized_objective(attn_pooled, v_bar, M, keep_mask, n_cand=None, dtype=torch.float32):
    """`J = ||g||^2_M / K^2` per (..., nh, q_blk) for an ARBITRARY kept set -- the same
    scalar `restore_swap_rank_by._row_objective` computes, lifted out so a selection made
    by any criterion (or by no criterion at all) can be scored on it.

    attn_pooled/v_bar: the pooled map and block-mean values over the KERNEL's full
    key-block grid (candidates first, sinks last), exactly what `dense_reference` expects.
    keep_mask: (..., nh, q_blk, n_cand) bool over the DROPPABLE candidates only, True =
    kept -- the sinks are always kept and are accounted for by `total_mass`, not here.
    Returns (..., nh, q_blk)."""
    P, V, v_bar_query, total_mass = dense_reference(attn_pooled.to(dtype), v_bar.to(dtype), n_cand)
    Pd = P * (~keep_mask).to(dtype)
    sum_Pd = Pd.sum(dim=-1)
    g = sum_Pd.unsqueeze(-1) * v_bar_query - Pd @ V
    obj = (g * torch.einsum("hij,...hqj->...hqi", M.to(dtype), g)).sum(dim=-1)
    return obj / (total_mass - sum_Pd).clamp(min=KEPT_MASS_EPS).pow(2)




def _top_m_mask(quality, counts, largest=True, k_max=None):
    """Boolean mask of each row's top-`counts[row]` entries of `quality` along the
    last dim. Replaces the
    `rank = argsort(argsort(q, descending=True)); mask = rank < counts` idiom this
    module used everywhere. That idiom materializes a FULL ranking of all `k_blk`
    candidates just to read off the top few, and profiling at real 100-view shapes
    ((nh, q_blk, k_blk) = (16, 810, 1619)) found it to be the single dominant cost
    of the whole restore-swap path -- 5.63 ms per ranking vs 0.20 ms for the
    round's main einsum, ~57% of `restore_swap_rank_by`'s total measured overhead.
    `torch.topk` at the same shapes costs 1.06-1.79 ms depending on k, i.e. 3-5x
    less, because it never sorts the tail it is about to discard.

    Semantically identical to the rank-based idiom except for tie ORDER between
    exactly-equal `quality` values (both pick an arbitrary winner among ties; float
    scores tie with probability ~0 here), so this is a drop-in speedup, NOT a
    method change -- every measured result predating it stays valid.

    quality: (..., k_blk) float. counts: (...) long/int, per-row how many to take
    (clamped to [0, k_blk]). largest=False takes the SMALLEST `counts` instead
    (the compensate-by-lowest-score case).

    `k_max`: optional Python int upper bound on `counts.max()`. Deriving it here
    instead costs a `.item()`, i.e. a GPU->CPU SYNC, on every call -- at n_iters=5
    that is 11 syncs per attention layer (initial cut + 2 per round) x 24 layers =
    264 pipeline stalls per forward pass, which measured as the gap between this
    path's microbenchmarked ~3.9 ms/round and its ~10 ms/round end-to-end cost.
    Callers that can bound the count analytically (the restore-swap schedules can:
    every per-round count is a deterministic fraction of `num_to_select`) should
    pass it and pay ONE sync per call instead. A loose bound only costs a slightly
    larger `topk` k, never correctness -- the per-row `counts` still decides the
    mask. Returns (..., k_blk) bool.

    `RESS_FUSED_TOPM=0` disables the Triton kernel below (`scoring.fused_topm`), which
    is the same selection done in one pass without materializing top-`k_max` values and
    indices -- 3.4x faster at the real shapes, and verified to produce the IDENTICAL
    mask (not merely an equivalent one) on every real dumped layer/count combination,
    so it is on by default. The env kill-switch exists for the case where a different
    GPU/Triton build disagrees, and for A/B timing.
    """
    k_blk = quality.shape[-1]
    if _FUSED_TOPM:
        from . import fused_topm as _fused
        if _fused.can_use(quality):
            return _fused.top_m_mask(quality, counts, largest=largest)
    counts = counts.clamp(min=0, max=k_blk)
    if k_max is None:
        k_max = int(counts.amax().item()) if counts.numel() else 0
    k_max = max(0, min(int(k_max), k_blk))
    mask = torch.zeros_like(quality, dtype=torch.bool)
    if k_max == 0:
        return mask
    idx = torch.topk(quality, k_max, dim=-1, largest=largest).indices  # sorted best-first
    pos = torch.arange(k_max, device=quality.device).expand(idx.shape)
    return mask.scatter(-1, idx, pos < counts.unsqueeze(-1))


KEPT_MASS_EPS = 1e-6


def kept_mass_norm_quality(G, K, delta_G, u, eps=KEPT_MASS_EPS):
    """`J - J_new` (higher = better, 0 = no change) for a single move that changes
    `||g||^2_M` by `delta_G` and the kept probability mass `K` by `u`, where
    `J = ||g||^2_M / K^2` is the kept-mass-normalized drift objective
    (`restore_swap_rank_by(kept_mass_norm=True)`; see that function's docstring for why
    `g` alone is the wrong quantity to minimize). `u = +P_k` restores dropped candidate k,
    `u = -P_j` gives up kept candidate j.

        J - J_new = G/K^2 - (G + delta_G)/(K+u)^2
                  = [ G * u * (2K + u) / K^2  -  delta_G ] / (K+u)^2

    The second form is what this computes. The first would subtract two nearly-equal
    large numbers -- both terms are O(G/K^2) while their difference is O(u/K) smaller --
    and lose most of its significant digits; under `_bf16` (~3 decimal digits) it loses
    the answer entirely. The grouped form has no such cancellation: `G*u*(2K+u)/K^2` and
    `delta_G` are both already first-order-small.

    G, K: (..., q_blk). delta_G, u: (..., q_blk, k_blk). Returns (..., q_blk, k_blk)."""
    K_ = K.unsqueeze(-1)
    K_new = (K_ + u).clamp(min=eps)
    return (G.unsqueeze(-1) * u * (2.0 * K_ + u) / K_.pow(2) - delta_G) / K_new.pow(2)


def bake_M(model):
    """model: a loaded (wired) VGGT aggregator (`models.vggt.load_vggt`'s return).
    Returns (n_blocks, nh, hd, hd) -- one Mahalanobis matrix per (layer, head),
    sliced from each global block's `attn.proj.weight` ((dim, nh*hd), PyTorch
    Linear's (out, in) layout -- column h*hd:(h+1)*hd is W_O^h).
    Bias is irrelevant (dropped): M_h only ever multiplies difference vectors, and
    `W_O(a) - W_O(b) = W_O_h(a - b)` cancels any additive bias term."""
    Ms = []
    with torch.no_grad():
        for block in global_blocks(model):
            W_O = block.attn.proj.weight  # (dim, nh*hd)
            nh = block.attn.num_heads
            hd = W_O.shape[1] // nh
            heads = []
            for h in range(nh):
                W_O_h = W_O[:, h * hd:(h + 1) * hd]  # (dim, hd)
                heads.append(W_O_h.T @ W_O_h)  # (hd, hd)
            Ms.append(torch.stack(heads, dim=0))  # (nh, hd, hd)
    return torch.stack(Ms, dim=0)  # (n_blocks, nh, hd, hd)


def bake_and_attach_M(model):
    """Bakes `bake_M(model)` and attaches each layer's (nh, hd, hd) slice as
    `block.attn.M`, so `models.attention.ress_attention_forward`'s
    `block_criterion="projected"` branch (the per-head or
    joint W_O-projected block cutoff) can read it without threading M through every
    call site. Idempotent -- safe to call again after re-loading a model."""
    M = bake_M(model)
    for i, block in enumerate(global_blocks(model)):
        block.attn.M = M[i].to(next(block.attn.parameters()).device)
    return M














def quadratic_deviation(v_bar_query, v_bar, M):
    """v_bar_query: (..., nh, q_blk, hd) = attn_pooled @ v_bar, the dense
    per-query-block reference (`...` is an optional leading batch dim -- present in
    the live model path, since `models.attention.predict_attention` keeps B,
    absent in unit tests / toy instances; the ellipsis einsum below handles both
    without a special case, found necessary when `block_criterion="projected"`
    crashed on the live-model 5D case, tests/test_projected.py's 4D-only toy case
    had not caught the missing batch-dim support). v_bar: (..., nh, k_blk, hd).
    M: (nh, hd, hd) (never has a batch dim -- scene-independent, `bake_and_attach_M`).
    Returns sqrt(e^T M_h e) for e = v_bar_query_i - v_bar_j, per (head, q_blk,
    k_blk): (..., nh, q_blk, k_blk). Clamped at 0 before sqrt -- M is PSD by
    construction (W_O_h^T W_O_h) so this only guards float rounding, not a real
    negative eigenvalue.

    Expands e^T M e = vq^T M vq - 2 vq^T M vk + vk^T M vk (M symmetric) instead of
    forming e = vq - vk directly -- the direct form needs a (..., nh, q_blk, k_blk,
    hd) intermediate (both for e itself and for M @ e) that measured as the
    dominant cost of `block_criterion="projected"` at 100-view scale (hd~64x
    bigger than the (..., nh, q_blk, k_blk) result actually needed downstream, e.g.
    `selection.beta_budget.compute_vector_sum_cost`'s docstring hit an 8.74 GiB
    OOM from the same shape). This form's largest intermediate is that same
    (..., nh, q_blk, k_blk) result -- verified numerically identical to the direct
    form (max abs diff ~6e-6, fp32) before swapping in."""
    Mvq = torch.einsum("hij,...hqj->...hqi", M, v_bar_query)  # (..., nh, q_blk, hd)
    Mvk = torch.einsum("hij,...hkj->...hki", M, v_bar)  # (..., nh, k_blk, hd)
    term_q = (Mvq * v_bar_query).sum(dim=-1)  # (..., nh, q_blk)
    term_k = (Mvk * v_bar).sum(dim=-1)  # (..., nh, k_blk)
    cross = torch.einsum("...hqi,...hki->...hqk", Mvq, v_bar)  # (..., nh, q_blk, k_blk)
    quad = (term_q.unsqueeze(-1) - 2.0 * cross + term_k.unsqueeze(-2)).clamp(min=0)
    return quad.sqrt()


# ONE objective, applied at every scope without exception, and with no discrepancy between
# the formula here and the one the kernel realises. The sparse kernel re-softmaxes over
# whatever it keeps, so
# dropping the set D shifts this row's pooled output by EXACTLY
#
#     eps = g / K,   g = Sum_{j in D} p_ij (v̄_i - V_j),   K = 1 - Sum_{j in D} p_ij
#
# and the quantity to minimize is `J = ||g||^2_M / K^2 = ||eps||^2_M`. There is no second
# criterion anywhere in this module: the "renormalized score" is J's |D| = 1 special case,
#
#     J({j}) = p_ij^2 ||d_ij||^2_M / (1 - p_ij)^2 = ( s_ij / (1 - p_ij) )^2,
#
# so ranking candidates by `s/(1-p)` for the initial one-shot cut and by the exact
# `J - J_new` marginals inside the loop are the SAME objective read at two set sizes --
# not two conventions that have to be reconciled. What must never happen is folding the
# 1/(1-p) factor into `d` itself and then feeding that into the loop's algebra: `s^2` there
# has to stay `d^T M d` for the cumulative `g = Sum_j d_j` it is combined with, and
# `1/(1-p_ij)` vs `1/K` would be the same re-softmax counted twice.
# `RESS_RENORM=0` restores the unrenormalized cut for A/B.
_RENORM_DEFAULT = os.environ.get("RESS_RENORM", "1") != "0"


def dense_reference(attn_pooled, v_bar, n_cand=None):
    """Split a pooled map laid out on the KERNEL's key-block grid into the part the
    selector may drop and the part it may not: the sink tokens belong in the formula, not
    outside it.

    The sparse kernel attends over `cat([k_patch, k_special])` and appends the trailing
    special ("sink") key-blocks to every row's mask as always-kept
    (`models.attention.build_block_mask`'s `sink_blocks`). Earlier revisions had the scoring
    math was handed `predict_attention(q_patch, k_patch)` instead -- a softmax over the
    PATCH keys only, i.e. the patch-CONDITIONAL distribution -- so the sinks were absent
    from `P`, from the reference `v̄`, and from the kept mass. Block COUNT made that look
    negligible (~1 of ~258 blocks at 12 views), but attention sinks are exactly the
    phenomenon where few keys carry disproportionate mass:
    The sink share was measured at 4.9% of the pooled mass
    on average over VGGT's global layers, up to 10.9% for a whole layer, and over 0.92
    for individual query blocks.

    Given the pooled map over ALL blocks (candidates first, sinks last) and the candidate
    count, this returns the four quantities every formula below is built from:

        P            (..., nh, q_blk, n_cand)  true probabilities of the DROPPABLE blocks
        V            (..., nh, n_cand, hd)     their block-mean values
        v_bar_query  (..., nh, q_blk, hd)      Sum over ALL blocks of p*V == the dense output
        total_mass   (..., nh, q_blk)          Sum over ALL blocks of p == 1

    Two things change substantively, and neither is a new formula -- see
    `restore_swap_rank_by`, whose algebra is carried over verbatim:

    - `v_bar_query` is the true dense pooled output, so the deviation `v̄_i - V_j` is
      measured against a reference that includes the sink values. Rescaling alone would
      not do this: with `alpha = 1 - beta` the patch mass, `v̄ = alpha*v̄_patch + beta*v̄_sink`,
      and the extra term tilts every candidate's deviation, not just their common scale.
    - `K` starts at `total_mass` and can only fall to `beta`, never to 0. The sink mass is
      the part of the row the selector cannot lose, so a row with heavy sink attention is
      correctly judged less damaged by the same patch drop -- and `J = ||g||^2_M/K^2` loses
      the 0/0 degeneracy it had when a row's entire candidate set could be dropped.

    `n_cand=None` means "no sink blocks", which reproduces the earlier behaviour
    exactly (`total_mass` is then the row sum of a full softmax, i.e. 1) -- that is the
    path the unit tests and the `RESS_SINK_MASS=0` A/B take."""
    if n_cand is None:
        n_cand = attn_pooled.shape[-1]
    return (attn_pooled[..., :n_cand], v_bar[..., :n_cand, :],
            attn_pooled @ v_bar, attn_pooled.sum(dim=-1))


def projected_score(attn_pooled, v_bar, M, n_cand=None, renormalize=None):
    """The per-candidate criterion `sqrt(J({j}))` -- the exact drift left behind by
    dropping candidate `j` ALONE, measured in the metric its head reaches the residual
    stream through:

        s_ij / (1 - p_ij),      s_ij = p_ij * ||W_O^h (v̄_i - V_j)|| = p_ij * sqrt(d^T M d)

    (`sqrt` rather than `J` itself only because the two order candidates identically and
    `s` is the quantity `restore_swap_rank_by`'s marginals reuse.)

    attn_pooled: (..., nh, q_blk, k_blk); v_bar: (..., nh, k_blk, hd); M: (nh, hd, hd)
    (see `bake_M`). `n_cand` marks how many leading key-blocks are DROPPABLE candidates,
    the rest being always-kept sinks -- see `dense_reference`. Returns
    (..., nh, q_blk, n_cand).

    `renormalize=False` drops the `1/(1-p_ij)` factor, leaving the bare `s_ij`. That is
    not an alternative criterion, it is a building block of this one; the flag exists for
    the ablation and for the round algebra, which needs `s^2 == d^T M d` exactly (see this
    module's `_RENORM_DEFAULT` comment). `None` takes the default, i.e. ON.

    The clamp below only guards float rounding: with true probabilities
    `1 - p_ij >= beta_i + Sum_{other candidates} p >= 0`, and it was reachable at 0 only
    while `p` was the patch-conditional distribution of a single-candidate row."""
    if renormalize is None:
        renormalize = _RENORM_DEFAULT
    P, V, v_bar_query, _ = dense_reference(attn_pooled, v_bar, n_cand)
    score = P * quadratic_deviation(v_bar_query, V, M)
    if renormalize:
        score = score / (1.0 - P).clamp(min=1e-4)
    return score






def restore_swap_rank_by(attn_pooled, v_bar, M, num_to_select, restore_frac=0.5,
                          n_iters=1, dtype=torch.float32, track_best=False,
                          big_dtype=None, decay=0.5, renormalize=None, n_cand=None,
                          maximize=False, invert_cut=False):
    """Drift-correction via RESTORE-and-SWAP. It replaced an earlier "drop more to
    cancel" design, which overshot catastrophically at high sparsity. A `block_criterion` alternative to plain `projected_score`, same
    contract (which k-blocks fill an ALREADY-decided per-head budget
    `num_to_select`):

    0. Full one-shot drop: keep the top `num_to_select` candidates by plain
       `projected_score`, drop the rest (`total_drop = k_blk - num_to_select`) --
       identical to `block_criterion="projected"` alone. This is the STARTING
       kept/drop state for iteration 1.

    Then `n_iters` rounds (default 1, matching the original single-shot design;
    `n_iters=5` is what the ReSS paper reports), each round given the CURRENT
    kept/drop state from the previous round:

    1. Compute the CURRENTLY realized net drift `g = Sum_{k in current drop-set}
       e_k` (`e_k = P_k(v̄_query - v̄_k)`, exactly `compute_vector_sum_cost`'s own
       `g` -- no extra forward pass, analytic from tensors already computed this
       layer, RE-DERIVED fresh from the current state every round, not just once).
    2. RESTORE `round(restore_frac/n_iters * num_to_select)` candidates FROM the
       current drop-set back to kept -- whichever ones, if removed from the
       drop-set, shrink ||g||^2_M the most. Removing k changes g -> g - e_k, exact
       squared-norm delta = `-2*(e_k.Mg) + s_k^2` (`||e_k||^2_M == s_k^2` exactly,
       verified numerically against a direct before/after computation); restore
       the most-negative-delta candidates.
    3. Compensate by newly dropping the same count from the CURRENT kept set, by
       plain lowest `projected_score` (no drift criterion -- deliberately, to
       avoid `two_pass_rank_by`'s exact batch-overshoot failure mode on this side
       of the swap too).
    4. This round's result becomes next round's starting kept/drop state.

    Splitting `restore_frac` evenly across `n_iters` smaller steps and
    RE-EVALUATING `g` from the actual current state before each step is the fix
    for `two_pass_rank_by`/this function's own single-shot version both sharing
    the same failure mode: a single big static batch picked against a FIXED `g`
    can overshoot when many candidates happen to align similarly. Doing it in
    `n_iters` smaller, re-evaluated steps lets each step correct only what the
    PREVIOUS step actually left behind, rather than guessing the whole correction
    at once.

    `maximize`/`invert_cut` (the ReSS paper's "Sanity Check: Drift Maximization") run this
    same machinery BACKWARDS, as the negative control for
    the claim that `J` is the causal quantity rather than a proxy for attention magnitude:

    - `maximize=True` flips the sign of the objective the ROUNDS optimize -- restore
      whichever dropped candidates raise `J` most, compensate with whichever kept ones
      raise it most, and (under `track_best`) keep the round that reached the LARGEST `J`.
      The initial cut is untouched, so this arm and Ours start from the identical top-k
      `projected_score` state and take the identical number of swaps; the ONLY difference
      between them is the sign. That isolation is the point: keeping the attention prior
      fixed on both sides is what makes the resulting quality gap evidence about `J`
      itself, which a whole-pipeline reversal cannot provide (see `invert_cut`).
    - `invert_cut=True` additionally flips the initial cut to keep the LOWEST-scoring
      candidates. Combined with `maximize` this is the full reversal, and it is the
      OUTER bracket of the experiment, not its headline: maximizing `J` means dropping
      the highest-`P` blocks, driving `K` toward the sink mass, so this arm degenerates
      into "keep only near-zero-attention blocks" -- catastrophic for a reason that says
      nothing about the residual-space metric specifically.

    Both leave every other handle (head budget, sink protection, swap schedule, measured
    sparsity) exactly where Ours has it. `maximize` also forces the eager round body:
    `scoring.fused_round`'s kernel has the minimization direction compiled in, and a
    silently-still-minimizing fused path would make the control agree with Ours.

    `restore_frac` (total, across all iterations) scales `num_to_select` (the KEPT
    budget), NOT `total_drop` -- scaling off `total_drop` collapses at high
    sparsity: once sparsity > 2/3, `total_drop > 2*num_to_select`, so a single-shot
    `restore_frac=0.5*total_drop` exceeds `num_to_select` and clamps to it,
    swapping out the ENTIRE original kept set for drift-canceling-but-otherwise-
    low-score candidates (observed: pose AUC30 collapsed to ~0.03,
    near-random, at sparsity~0.73). Scaling off `num_to_select` instead bounds the
    total swap to at most `restore_frac` of the kept budget regardless of how
    large `total_drop` grows.

    schedule="geometric" (default) is the decaying per-round budget described above.
    schedule="fixed" instead spends a CONSTANT
    `fixed_frac` of `num_to_select` every round forever (no `restore_frac` cap) --
    this is close in spirit to the equal-split schedule this function's own history
    (above) already found oscillates round-over-round rather than converging.

    schedule="pair_argmin" (following a real-data diagnostic
    -- the round-by-round diagnostic on the production `_ccdrift` recipe
    -- that found round 0 (geometric decay's largest single step, 50% of `restore_frac`
    in one un-re-evaluated batch) routinely roughly DOUBLES `||g||^2_M` before later,
    shrinking rounds walk it back down: "spikes then decays", not monotone. Root cause
    diagnosed as the same batch-overshoot failure mode already found for
    `pooled_restore_swap_select` -- restore's and
    compensate's per-candidate deltas are each computed independently (holding
    everything else fixed), so a batch that moves many candidates at once ignores the
    CROSS terms between them. Requires `restore_criterion="magnitude"` and
    `compensate_criterion="drift"` (raises otherwise) and does not support
    `sequential_compensate` (the mechanism below already accounts for the specific
    restore/compensate interaction that flag targets, for every candidate it actually
    considers pairing).

    Replaces the schedule-decided `restore_frac`/`per_iter_count` entirely with a
    per-row SELF-SIZING swap, computed once per round with no retry loop:

    1. Rank the drop-set by `Delta_k` (exact restore delta, ascending -- most
       beneficial first, same `restore_quality` ranking already used) and the
       kept-set by `Delta'_j` (exact compensate delta, ascending -- least harmful
       first, same `compensate_quality` ranking `compensate_criterion="drift"`
       already computes) independently, then pair them 1:1 by rank position:
       `(k_1,j_1), (k_2,j_2), ...` up to `W = min(|drop-set|, |kept-set|)` pairs.
    2. For `m` in `[1, W]`, compute the EXACT delta of applying pairs `1..m`
       TOGETHER against the round's starting `g`. FIRST ATTEMPT (abandoned same day): summing each pair's own INDEPENDENT single-pair delta
       `swap_delta_i = Delta_k_i + Delta'_j_i - 2*(d_k_i . M d_j_i)` (exact for pair
       i ALONE, via `||g-d_k+d_j||^2_M - ||g||^2_M`'s quadratic expansion) --
       real-data validation found this catastrophically WRONG for m>1: DTU round
       traces got 5-10x WORSE, not better (round9 ~0.25-0.33 vs round0 ~0.03-0.07),
       because top-ranked candidates turned out to be strongly M-correlated with
       each other (measured mean pairwise M-cosine ~0.64 among the top 10 pairs'
       net vectors `n_i = d_j_i - d_k_i` on a synthetic check), not independent --
       summing SCALAR per-pair deltas discards the cross terms `n_i . M n_i'`
       between DIFFERENT pairs entirely, and those cross terms are large and
       same-signed (not noise that cancels) precisely because a "most beneficial
       first" ranking selects candidates that all point the same useful direction,
       i.e. redundant with each other. FIX: cumsum the net
       VECTORS first, take the quadratic form once per `m`, not the other way
       around -- `N(m) = sum_{i<=m} n_i` (a `(..., W, hd)`-sized vector cumsum,
       cheap, same size class as the per-pair `hd`-sized gathers already needed
       for the ranking), then `total(m) = 2*N(m).Mg + N(m)^T M N(m)` -- this is
       `||g+N(m)||^2_M - ||g||^2_M` EXACTLY, for every `m` simultaneously, no
       cross-pair approximation left (verified: matches a direct brute-force
       before/after `||g||^2_M` recomputation to float64 precision at every `m`
       tested). Still only O(W*hd) work (a vector cumsum + one `hd`-sized quadratic
       form per m), not O(W^2) pairwise dot products, and the PAIRING itself
       (deciding WHICH k_i goes with WHICH j_i) is still by rank position, not an
       optimal assignment search (item (B)
       flags the O(m^2) assignment problem as a separate, costlier idea not
       implemented here) -- only the SIZING decision needed to become exact.
    3. `drift_at(m) = g_norm_sq_start + total(m)` for `m` in `[0, W]` (m=0 = do
       nothing this round, ALWAYS a valid candidate, `total(0)=0`). Adopt
       `m* = argmin_m drift_at(m)` -- the swap count is whatever the data says is
       best THIS round, not a pre-decided fraction, and (now that `total(m)` is
       exact rather than an independent-sum approximation) truly can never make the
       round's `||g||^2_M` worse than not swapping at all. No external
       `restore_frac`/schedule cap on `m`'s search range -- the search is bounded by the
       drift magnitude actually reached, not by an arbitrary count ceiling.

    Residual approximation: the SIZING decision (`m*`) is now exact for the pairing
    it's given, but the PAIRING/ordering itself (which `k_i` goes with which `j_i`,
    and in what priority) is still the cheap single-candidate-marginal heuristic
    from step 1, not a jointly-optimal assignment -- a different pairing order could
    in principle reach a lower `drift_at` for the same `m`. `track_best` (recommended
    kept on) remains a residual safety net against round-to-round drift from THAT
    remaining gap, not against the sizing itself (which no longer needs it).
    `track_best=True` (must be paired with `schedule="fixed"` to matter -- geometric
    decay's cumulative spend is bounded and already converges, so there is no
    oscillation for it to fix) is the user's proposed mitigation for that: instead
    of returning whatever state the LAST round happens to land on, every round's
    resulting `||g||^2_M` (per (..., nh, q_blk) row) is compared against the best
    seen so far (including the pre-loop initial one-shot cut) and the row keeps
    whichever round's kept/drop state achieved the lowest drift, not necessarily the
    final one -- turns oscillation from a liability into "extra free samples of the
    kept/drop state space", at the cost of one extra `g`/`Mg` evaluation after the
    loop ends (to also score the final round's state, which the loop's own
    before-this-round diagnostic point never gets to see).

    `restore_criterion="magnitude"` (default) is the `||g||^2_M`-reduction ranking
    described above. `restore_criterion="angle"` exists because magnitude genuinely
    matters -- what is added to the residual stream carries its size, and an earlier claim
    that LayerNorm erases it was wrong -- but a DISCRETE block-selection process is a
    clumsy tool for fixing a
    continuous quantity; split the job instead) ranks candidates by how much
    restoring them would raise `cos_M(v̄_approx, v̄_query)` (`v̄_approx = v̄_query - g`,
    the row's current sparse approximation of the full/dense pooled output;
    `v̄_query` itself, already computed unmasked every call, IS the "full" reference
    -- no extra dense pass needed) instead of by how much they'd shrink `||g||^2_M`.
    Exact same "per-candidate-independent marginal" character as the magnitude
    version (both are closed-form deltas from a SINGLE candidate's restoration,
    holding everything else fixed): restoring k changes `v̄_approx -> v̄_approx + e_k`
    (removing e_k's contribution from `g` is exactly adding it back to `v̄_approx`,
    since `v̄_approx = v̄_query - g`), and
    `cos_M(v̄_approx+e_k, v̄_query) = num_k / sqrt(denom_sq_k * ||v̄_query||^2_M)` where
    `num_k = v̄_approx·M·v̄_query + e_k·M·v̄_query` and
    `denom_sq_k = ||v̄_approx||^2_M + 2*e_k·M·v̄_approx + s_k^2` -- both `e_k·M·v̄_query`
    and `e_k·M·v̄_approx` reuse the SAME bilinear-expansion trick the magnitude
    version's `raw_dot` already uses (just with `v̄_query` in place of `Mg` for the
    first, algebraically `raw_dot - raw_dot_vs_query` for the second), so no new
    per-candidate cost. `||v̄_query||^2_M` is a per-row CONSTANT (same every round,
    unlike `Mg`), so it's precomputed once, outside the loop, unlike the magnitude
    version's fully-round-dependent terms -- and since it's a positive constant
    common to every candidate's denominator, it doesn't affect the ranking and is
    dropped from `angle_quality` entirely (only matters for an actual cosine VALUE,
    not for picking the argmax). Compensate (the drop side) is controlled separately
    by `compensate_criterion`, independent of `restore_criterion` -- see that
    parameter's own paragraph below.

    `compensate_criterion="score"` (default, unchanged behavior) drops the
    currently-kept candidates with the lowest plain `projected_score`, same
    overshoot-avoidance rationale the magnitude-mode docstring above already gives
    (deliberately drift-BLIND, so compensate can't itself overshoot the way
    `two_pass_rank_by` did). `compensate_criterion="drift"` exists because restore is
    drift-aware
    but compensate wasn't, an asymmetry with no principled justification beyond
    "it happened to avoid one specific failure mode") instead picks the `m`
    currently-KEPT candidates whose ADDITIONAL removal would raise `||g||^2_M` the
    LEAST -- the exact mirror of the restore step's own criterion. Dropping kept
    candidate `k` changes `g -> g + d_k` (`d_k = P_k*e_k`, the same vector the
    restore step's `Mg`/`raw_dot` already use), so the exact delta is
    `Delta'_k = 2*(d_k . Mg) + s_k^2 = -2*raw_dot + s_k^2` (`raw_dot = -(d_k . Mg)`,
    the SAME tensor computed once per round for either `restore_criterion` branch --
    no extra matmul, just a second ranking of an already-computed quantity). Compensate
    picks the `m` smallest-`Delta'_k` kept candidates, i.e. ranks by `-Delta'_k =
    2*raw_dot - s_k^2` descending. This is the untested half of the "certified
    monotone decrease" argument by analogy with OMP/local search -- symmetric criteria
    alone do NOT by themselves guarantee a round
    never increases `||g||^2_M` (both sides are still evaluated against a FIXED
    snapshot of `g`/`Mg`, so the same batch-interaction risk `two_pass_rank_by`
    hit applies to `compensate_criterion="drift"` too, potentially compounded by
    running on both sides of the swap at once) -- this flag exists to measure that
    empirically, not because the risk has been ruled out analytically.

    `sequential_compensate=True` follows from a diagnostic that confirmed a
    `compensate_criterion="score"` round routinely UNDOES more than half of what
    that same round's restore just did, measured as a "net surviving swap count"
    well below the nominal budget, while `compensate_criterion="drift"` undoes
    much less but still isn't immune per the paragraph above) re-derives `g`/`Mg`
    from the state AFTER this round's restore has been applied (instead of the
    STALE pre-restore snapshot every other mode uses) before deciding what to
    compensate-drop, and lets the compensate step's candidate pool include the
    just-restored blocks (a data-driven "should we actually keep this one" check,
    not blind re-cancellation, since it acts on freshly-accurate `g`, not on
    `compensate_criterion="score"`'s g-blind ranking). Only changes anything when
    paired with `compensate_criterion="drift"` -- `"score"` never reads `g` at
    all, so a mid-round refresh has nothing to act on. Costs one extra `g`/`Mg`
    evaluation per round (same shape as `track_best`'s own post-loop one).

    `return_norm_correction=True` additionally returns a per-(..., nh, q_blk) scalar
    `norm_correction = ||v̄_query||_2 / ||v̄_approx_final||_2` (PLAIN Euclidean norm,
    not `||·||_M` -- this corrects the actual output TENSOR's magnitude, a physical
    quantity in value-space, not a `W_O`-projected abstraction), computed ONCE after
    the loop from the FINAL kept/drop state (one extra `g` evaluation, same pattern
    `track_best`'s post-loop score already uses). Meant for the caller to multiply
    directly into the real (fine-grained, token-level) attention output before it
    hits the residual stream -- `angle`-mode restore-swap only fixes v̄_approx's
    DIRECTION; this scalar is the intended complementary fix for the MAGNITUDE gap
    `angle` mode deliberately ignores. Clamped to [0.2, 5.0] (safety bound against a
    near-zero `v̄_approx_final` blowing the ratio up on a degenerate row -- v̄_approx
    is a pooled BLOCK-level statistic used here as a cheap proxy for the real
    token-level output's magnitude, not a proof the correction is exact).

    `renormalize` (None => on, per `_RENORM_DEFAULT`; user) is not a second
    criterion layered on top of this one -- it is this one, evaluated on a single-candidate
    drop set. The initial one-shot cut compares candidates one at a time, and for
    `D = {j}` the objective `J = ||g||^2_M/K^2` is exactly `(s_ij/(1-p_ij))^2`; the rounds
    then evaluate the same `J` for the whole set. The factor is therefore applied at every
    scope, but never twice to the same quantity: it decorates the CUT's ranking key, while
    the loop reaches the identical correction through `1/K` on the cumulative `g`. Folding
    it into `d` instead would make `s^2 != d^T M d` and the round marginals simply wrong
    (`tests/test_kept_mass_norm.py::test_quality_matches_bruteforce` catches that;
    `tests/test_projected.py::test_renormalized_cut_is_the_single_drop_objective` pins the
    equivalence that makes the two scopes one objective). Earlier revisions had this flag feed
    the renormalized score into the round algebra as well, which was inconsistent once
    `kept_mass_norm` landed; nothing enabled it, so it never shipped that way.

    `n_cand` marks how many leading
    key-blocks of `attn_pooled`/`v_bar` are DROPPABLE candidates; the rest are the kernel's
    always-kept sink blocks, which participate in the reference `v̄` and in the kept mass
    but are never selected against. See `dense_reference`. `None` = no sinks (the pre-fix
    behaviour, and what the unit tests drive).

    `kept_mass_norm=True` (block_criterion suffix "_knorm") is exactly the
    correction `renormalize` deferred, now derived and applied to the CUMULATIVE `g`.
    The sparse kernel re-softmaxes over the kept blocks, so this layer's realized output
    shift is not `g` but

        delta = g / K,      K = Sum_{j kept} P_ij   (the kept probability mass, 1 - S)

    -- `g` is that shift scaled DOWN by the kept mass. A budget-vs-drift sweep
    found the consequence on the head-BUDGET axis to be fatal (`||g||^2_M` vanishes as a
    head is starved to zero, so an exact allocator starves 69-75% of the heads and calls
    it a 99.8% drift reduction -- the mechanism behind the `waterfill`
    collapse). On the ORDERING axis, which is what this function decides, the budget is
    fixed so the effect is milder but not absent: `K` still varies from query row to query
    row, so minimizing `||g||^2_M` systematically UNDER-weights exactly the rows that lost
    the most mass -- the most damaged ones. This flag optimizes

        J = ||g||^2_M / K^2      instead of      ||g||^2_M

    everywhere the round body compares states: both marginal qualities, `track_best`'s
    per-row winner, and the floor's `beneficial_threshold` (still 0, since J-reduction is
    still "positive quality"). Restoring candidate k moves `g -> g - d_k` AND `K -> K + P_k`;
    dropping kept j moves `g -> g + d_j` and `K -> K - P_j`, so the exact deltas become

        J - J_new = [ G * P (2K +- P) / K^2  -  Delta_G ] / (K +- P)^2

    (+ for restore, - for compensate; `G = ||g||^2_M`, `Delta_G` the same exact squared-norm
    delta the unnormalized path already computes). Written in that grouped form rather than
    as `G/K^2 - G_new/K_new^2` deliberately -- the latter subtracts two nearly-equal large
    numbers and loses most of its significant digits, catastrophically so under `_bf16`.
    To first order in `P` this is `(old_quality + 2*P_k*G/K)/K^2` for restore and
    `(old_quality - 2*P_j*G/K)/K^2` for compensate, i.e. the correction adds a preference
    for restoring HIGH-mass candidates and giving up LOW-mass ones -- recovering attention
    mass is now worth something on its own, which the unnormalized objective cannot see.

    Not supported with `restore_criterion="angle"` (that criterion optimizes a direction
    ratio, not `||g||^2_M`, so there is nothing to renormalize) nor with
    `schedule="pair_argmin"` (its exact joint-delta cumsum is built for the quadratic form
    only); both raise. `scoring.fused_round` DOES cover this objective: `G`
    and `K` are per-row scalars, so the kernel takes two extra scalar loads and still makes
    one pass over the row -- without that, the eager fallback cost more than the method
    gained (+1.74% -> +5.14% forward overhead vs SparseVGGT).

    `cross_head=True` (block_criterion suffix "_xhead") replaces the per-head
    objective `Sum_h ||g_h||^2_{M_h}` with the RESIDUAL-STREAM one,

        || Sum_h W_O^h g_h ||^2      (call the inner vector D)

    i.e. it stops treating the heads as independent problems. They are not: they all write
    into the same residual stream, and this project's own diagnostic recorded the
    cross-head off-block-diagonal Gram energy EXCEEDING the diagonal at every layer checked
    (1.2-3.0x), so `Sum_h ||g_h||^2` discards more than it keeps. A candidate whose
    contribution opposes the other heads' current drift can then be genuinely free -- even
    beneficial -- to drop, which the block-diagonal objective cannot represent.

    The delta algebra is UNCHANGED in form, which is why this is a flag and not a new
    function. Restoring candidate k of head h moves `D -> D - D_k` with `D_k = W_O^h d_k`,
    so the exact delta is `-2<D_k, D> + ||D_k||^2`; and

        <D_k, D> = <d_k, (W_O^h)^T D> ,      ||D_k||^2 = d_k^T M_h d_k = s_k^2

    so substituting `gamma_h = (W_O^h)^T D` for `M_h g_h` reproduces every downstream
    formula verbatim -- `raw_dot`, both qualities, the selection, even `s_sq`. No
    per-candidate residual vector is ever materialized (that (nh, k_blk, dim) projection is
    what forced `residual_aware_restore_swap_rank_by` to be QUERY-POOLED), so this stays
    per-query-row, and `term_k` keeps its shape/dtype so `scoring.fused_round` needs no
    change either.

    Cost: two extra einsums per round, `nh*q_blk*hd*dim` each (~0.85 GFLOP at 100-view
    shapes) against the round's existing ~2.7 GFLOP -- and nothing extra on the `k_blk`
    axis, which is what the round's memory traffic is made of.

    Requires `W_O` ((nh, dim, hd), `bake_W_O`). `track_best` under this flag compares each
    head's CONTRIBUTION `<g_h, gamma_h>` to `||D||^2` (they sum to `||D||^2` exactly), so
    it can legitimately go negative for a head that is cancelling the others.

    attn_pooled: (..., nh, q_blk, k_blk). v_bar: (..., nh, k_blk, hd).
    M: (nh, hd, hd). num_to_select: (..., nh, q_blk) long.

    Returns rank_by (..., nh, q_blk, k_blk): final keep/drop decision encoded as a
    large keep-vs-drop offset on top of the base score, for
    `models.attention.build_block_mask`'s top-`num_to_select` selection to
    reproduce exactly. If `return_norm_correction=True`, returns
    `(rank_by, norm_correction)` instead."""
    # Fixed by derivation rather than by measurement, so there is no branch for the
    # alternative (see `scoring.criterion`'s module docstring):
    #   kept_mass_norm -- the kernel re-softmaxes over kept blocks, so the realized
    #     output error is exactly g/K and the objective is J = ||g||^2_M / K^2.
    #   compensate_criterion="drift" -- restore and compensate are the same exact
    #     marginal applied in opposite directions.
    # `big_dtype` is the dtype of the
    # (..., nh, q_blk, k_blk)-sized tensors -- `P`, `s`, `raw_dot`, the quality
    # tensors -- which dominate this function's memory traffic; `dtype` remains the
    # ACCUMULATION dtype for the small (hd-sized) `g`/`Mg`/norm quantities, which
    # stay fp32 regardless. None (default) = same as `dtype`, i.e. the exact prior
    # all-fp32 behavior. torch.bfloat16 measured ~1.8x faster per round at real
    # 100-view shapes (7.08 -> 3.85 ms) -- and costs little precision in principle
    # since `attn_pooled`/`v_patch` arrive from a bf16 autocast forward pass anyway
    # (upcasting them here never recovered information the model didn't have), while
    # PyTorch still accumulates bf16 `sum`/matmul in fp32 internally. Still gated
    # behind an opt-in rather than made the default: the cancellation-heavy
    # `term_k - term_q` step could in principle behave differently, so it needs
    # eval-level re-validation before it can be trusted (block_criterion "_bf16").
    big_dtype = big_dtype or dtype
    M = M.to(dtype)
    # Sink-aware split: `P`/`v_bar` become the DROPPABLE candidate blocks,
    # while `v_bar_query_big`/`total_mass` are formed over ALL key-blocks the kernel
    # attends to -- the always-kept sinks included. Everything downstream is unchanged in
    # FORM; only these inputs move. See `dense_reference`.
    P, v_bar, v_bar_query_big, total_mass = dense_reference(
        attn_pooled.to(big_dtype), v_bar.to(big_dtype), n_cand)
    total_mass = total_mass.to(dtype)
    k_blk = P.shape[-1]

    # ONE objective, read at two set sizes (see `_RENORM_DEFAULT`). `s` is the bare
    # `p*||d||_M` because the round body's exact marginals are `Delta_G = +-2 d^T M g + s^2`,
    # where `s^2` has to be `d^T M d` for the cumulative `g = sum_j d_j` it is combined with;
    # `s_cut = s/(1-p)` is `sqrt(J)` for a single-candidate drop, which is precisely what the
    # one-shot cut below compares. Substituting `s_cut` into the algebra instead would count
    # the same re-softmax twice, once as `1/(1-p_ij)` and once as `1/K`.
    s = P * quadratic_deviation(v_bar_query_big, v_bar, M.to(big_dtype))
    s_cut = (s / (1.0 - P).clamp(min=1e-4)
             if (_RENORM_DEFAULT if renormalize is None else renormalize) else s)
    s_sq = s.pow(2)  # hoisted out of the round loop: `s` never changes across
    # rounds, but every round recomputed `s.pow(2)` for BOTH the restore and compensate
    # quality -- a full (..., nh, q_blk, k_blk) elementwise pass each, twice per round.
    total_drop = (k_blk - num_to_select).clamp(min=0)  # (..., nh, q_blk)

    # ONE sync per call (see `_top_m_mask`'s `k_max` docstring): every
    # per-round count below is a deterministic fraction of `num_to_select`, so this
    # single Python int bounds all of them -- no further `.item()` anywhere in the loop.
    max_nts = int(num_to_select.amax().item()) if num_to_select.numel() else 0

    # Iteration 0: full one-shot drop set by plain score (top `num_to_select` kept).
    # `invert_cut` negates the ranking key only -- same budget, same candidate set, the
    # LOWEST-`sqrt(J)` blocks kept instead of the highest.
    current_drop_mask = ~_top_m_mask(-s_cut if invert_cut else s_cut,
                                      (k_blk - total_drop), k_max=max_nts)

    thrash_initial_drop_mask = current_drop_mask.clone() if _thrash_diag_records is not None else None
    thrash_gross = 0.0

    v_bar_query = v_bar_query_big.to(dtype)  # (..., nh, q_blk, hd) -- the DENSE pooled output
    # (over candidates AND sinks), same every round. Kept in the fp32 accumulation dtype --
    # it is hd-sized, not k_blk-wide, and feeds every `g`/`Mg` computation; see the
    # `big_dtype` comment above.

    # The fused round kernel (`scoring.fused_round`) reproduces the round body exactly;
    # `_FUSED_ROUND` (env `RESS_FUSED_ROUND=0`) is the escape hatch back to the eager
    # path, which stays as the readable reference and as the fallback for shapes the
    # kernel declines (see `fused_round.can_use`).
    from . import fused_round as _fused_round
    # `maximize` has no kernel: `fused_round_step` picks the round's restore/compensate
    # sets by the minimizing direction internally, so the anti arms must take the eager
    # path (slower, and that is fine -- these are analysis-only runs).
    _fused_round_ok = _FUSED_ROUND and not maximize

    # `decay` (block_criterion suffix "_dc<NN>") was hardcoded 0.5 until now.
    # It sets HOW the total spend is distributed over rounds, which is a different axis
    # from `restore_frac`, which sets how much total there is:
    #
    #     round i spends  restore_frac * decay * (1 - decay)^i
    #     cumulative      restore_frac * (1 - (1-decay)^n_iters)
    #
    # At decay=0.5, n_iters=5, ROUND 0 ALONE IS 51.7% of the whole spend and it is applied
    # in one batch with no re-evaluation -- exactly the step `scripts/restore_swap_drift_
    # diagnostic.py` measured as roughly DOUBLING ||g||^2_M before later rounds walk it back
    # ("spikes then decays", see `schedule="pair_argmin"`'s paragraph above). Lowering
    # `decay` flattens the schedule and shrinks that first spike while keeping most of the
    # total; lowering `restore_frac` instead shrinks EVERY round proportionally and the
    # total with it, which measured monotonically WORSE (rf 0.5 -> 0.05: DTU Chamfer
    # 0.672 -> 0.723). So the two knobs are not substitutes and the flatter-schedule
    # direction is untested.
    #
    # Geometric decay, NOT an equal `restore_frac/n_iters`
    # split: round i uses HALF of whatever restore budget is still "remaining"
    # (round 0 spends 50% of `restore_frac`, round 1 spends 50% of what's left
    # -- 25% of the original total -- round 2 12.5%, ...), so cumulative spend
    # after `n_iters` rounds is `restore_frac * (1 - 0.5**n_iters)`, approaching
    # but never quite reaching `restore_frac`. Later rounds' corrections shrink
    # toward zero instead of staying constant-sized, which directly targets the
    # oscillation this function was observed to have with equal-sized rounds
    # (||g|| dropped sharply round 1, then oscillated rather than continuing to
    # converge) -- a shrinking step can't re-inject as much fresh
    # drift each round as a constant-sized one can.
    remaining_frac = restore_frac

    def _apply_metric(g):
        """`M_h g_h`, the drift measured in the metric this head's output reaches the
        residual stream through (`M_h = (W_O^h)^T W_O^h`, or identity under `_nowo`)."""
        return torch.einsum("hij,...hqj->...hqi", M, g)

    # `total_mass` (computed above by `dense_reference`) is the row's FULL probability mass,
    # candidates plus sinks -- 1 by construction, since the pooled map is a softmax over the
    # kernel's whole key-block grid. Before the sink fix it was `P.sum(-1)` over a
    # patch-only softmax, which is also 1, but for the wrong reason: the sinks were missing
    # from the distribution rather than accounted for. The difference shows up in `K` below,
    # which can now fall no lower than the sink mass instead of all the way to 0.
    _K_EPS = KEPT_MASS_EPS
    # Working dtype for `kept_mass_norm`'s k_blk-wide quality arithmetic. fp32, and the
    # A/B says that is free: at DTU scan1 / 100 views the bf16 version measured +4.90%
    # forward-pass overhead vs sparse_vggt against fp32's +5.14% -- 0.24pp, inside the
    # run-to-run spread. The cost is the extra elementwise PASSES the grouped quality
    # formula makes over the (nh, q_blk, k_blk) tensor, not the width of its operands, so
    # narrowing them buys nothing and only risks the ranking. `RESS_KNORM_BF16=1` keeps
    # the A/B reproducible.
    _knorm_dtype = big_dtype if os.environ.get("RESS_KNORM_BF16") == "1" else dtype

    def _row_objective(drop_mask):
        """The scalar this function minimizes per row, for a given drop-mask -- factored
        out so `track_best` can score the pre-loop initial cut and the
        post-loop final state the same way the in-loop check scores every round:
        `J = ||g||^2_M / K^2`."""
        Pd = P * drop_mask.to(big_dtype)
        sum_Pd = Pd.sum(dim=-1)
        masked_vbar_d = Pd @ v_bar
        g = sum_Pd.unsqueeze(-1).to(dtype) * v_bar_query - masked_vbar_d.to(dtype)
        obj = (g * _apply_metric(g)).sum(dim=-1)
        return obj / (total_mass - sum_Pd.to(dtype)).clamp(min=_K_EPS).pow(2)

    best_g_norm_sq = _row_objective(current_drop_mask) if track_best else None
    best_drop_mask = current_drop_mask.clone() if track_best else None

    this_call_diag = None
    if _diag_records is not None:
        this_call_diag = []
        _diag_records.append(this_call_diag)
    prev_g, prev_g_norm_sq = None, None  # for the diagnostic's round-to-round M-cosine
    ref_g, ref_g_norm_sq = None, None  # FIXED round-0 g: the comparison must be against the
    # full (round-0) drift, not against the previous round -- consecutive-
    # round cosine only shows LOCAL round-to-round stability; it can read ~1.0 late (rounds settled into a
    # near-tied 2-cycle, see the diagnostic script's convergence-identity check) while still having drifted
    # far from the ORIGINAL (round-0, pre-restore-swap) drift direction -- this compares every round back to
    # that one fixed baseline instead, so "did we end up somewhere related to where we started" is answered
    # directly rather than inferred from a chain of local comparisons.

    for _ in range(n_iters):
        this_round_frac = remaining_frac * decay
        remaining_frac -= this_round_frac
        per_iter_count = (this_round_frac * num_to_select.to(dtype)).round().long()

        # Analytic upper bound on every count this round takes a `topk` of, so no
        # `.item()` sync is needed inside the loop (see `_top_m_mask`'s `k_max`
        # docstring). `per_iter_count = round(this_round_frac * num_to_select)`
        # elementwise and `round` is monotone, so `round(frac * max_nts)` bounds it;
        # `+1` covers the ">=1 floor" applied below.
        round_k_max = min(k_blk, max(1, round(this_round_frac * max_nts) + 1))

        Pd = P * current_drop_mask.to(big_dtype)
        sum_Pd = Pd.sum(dim=-1)
        masked_vbar_d = Pd @ v_bar
        # `g`/`Mg` (and everything derived from them) stay in the fp32 accumulation
        # dtype even when `big_dtype` is bf16 -- see the `big_dtype` comment above.
        g = (sum_Pd.unsqueeze(-1).to(dtype) * v_bar_query - masked_vbar_d.to(dtype))  # (..., nh, q_blk, hd)

        Mg = _apply_metric(g)                          # (..., nh, q_blk, hd)
        g_norm_sq = (g * Mg).sum(dim=-1)  # ||g||^2_M per (..., nh, q_blk), BEFORE this round's decision
        # K = mass still kept, and the round's current objective J = ||g||^2_M / K^2.
        # `g_norm_sq` above stays the RAW norm so the diagnostic block below keeps
        # reporting the same quantity it always has.
        kept_mass = (total_mass - sum_Pd.to(dtype)).clamp(min=_K_EPS)   # (..., nh, q_blk)
        row_obj = g_norm_sq / kept_mass.pow(2)
        if track_best:
            better = row_obj > best_g_norm_sq if maximize else row_obj < best_g_norm_sq
            best_g_norm_sq = torch.where(better, row_obj, best_g_norm_sq)
            best_drop_mask = torch.where(better.unsqueeze(-1), current_drop_mask, best_drop_mask)
        fused_new_drop = None
        term_q = (v_bar_query * Mg).sum(dim=-1)  # (..., nh, q_blk)
        Mg_big = Mg.to(big_dtype)
        term_k = torch.einsum("...hkj,...hqj->...hqk", v_bar, Mg_big)  # (..., nh, q_blk, k_blk)
        if _fused_round_ok and _fused_round.can_use(P, s_sq, term_k, current_drop_mask,
                                                     big_dtype,
                                                     _knorm_dtype):
            # Everything from `raw_dot` down to this round's new drop-set in ONE kernel
            # (`scoring.fused_round`): the eager version below walks the
            # (..., nh, q_blk, k_blk) tensor ~10 times (raw_dot, two qualities, two
            # masked_fills, two selections, three boolean updates) where the kernel walks
            # it once. `term_k` above is deliberately still cuBLAS's einsum, so the GEMM
            # accumulation order is untouched and the kernel only has to reproduce the
            # elementwise rounding -- verified bit-identical (new drop mask AND
            # restore_count) on every real dumped layer, and end-to-end on a full forward.
            fused_new_drop, restore_count = _fused_round.fused_round_step(
                P, term_k, s_sq, term_q, current_drop_mask, per_iter_count, big_dtype,
                g_norm_sq=g_norm_sq,
                kept_mass=kept_mass,
                knorm_dtype=_knorm_dtype,
                k_eps=_K_EPS)
            raw_dot = None  # not materialized on this path
        else:
            raw_dot = P * (term_k - term_q.to(big_dtype).unsqueeze(-1))  # (..., nh, q_blk, k_blk) = -(e_k . M g)

            # Restoring k removes d_k from g (`Delta_G = 2*raw_dot + s^2`) and returns
            # P_k to the kept mass (`u = +P_k`); `kept_mass_norm_quality` turns that pair
            # into the exact reduction in J. `_knorm_dtype` is fp32 -- see where it is set.
            restore_quality = kept_mass_norm_quality(
                g_norm_sq.to(_knorm_dtype), kept_mass.to(_knorm_dtype),
                2.0 * raw_dot.to(_knorm_dtype) + s_sq.to(_knorm_dtype),
                P.to(_knorm_dtype))
            if maximize:
                restore_quality = -restore_quality
            restore_quality = restore_quality.masked_fill(~current_drop_mask, float("-inf"))
        beneficial_threshold = 0.0  # delta<0 (i.e. restore_quality>0) means restoring HELPS
        # -- and under `maximize` the negation above already redefined "helps" as "raises
        # J", so the floor's own semantics carry over with no second sign to track.
        if fused_new_drop is not None:
            # `scoring.fused_round` already produced this round's restore+compensate result
            # (and its `restore_count`, which the thrash/diag blocks below still read).
            pass
        else:
            # Floor at >=1 swapped block/row, so that the iteration keeps reducing drift even
            # at high sparsity: geometric decay alone rounds `per_iter_count`
            # to 0 well before `remaining_frac`'s cumulative budget is exhausted -- confirmed via
            # the per-round count trace (e.g. s=0.72 hits count=0 by
            # round 7/10) -- so every later round silently no-ops regardless of `n_iters`, and WORSE at high
            # sparsity since num_to_select (hence per_iter_count) starts smaller there too. Floored ONLY on rows
            # where restoring the single best candidate would actually improve `restore_quality` past
            # `beneficial_threshold` (0 for magnitude-mode's delta, the current no-restore ratio for angle-mode) --
            # rows with no genuinely beneficial candidate left still get restore_count=0 exactly as before, so
            # this does not force a swap once there is nothing left to gain (avoids reintroducing
            # `two_pass_rank_by`'s blind-overshoot failure mode).
            has_beneficial_candidate = restore_quality.amax(dim=-1) > beneficial_threshold
            per_iter_count = torch.where(has_beneficial_candidate, per_iter_count.clamp(min=1), per_iter_count)

            current_kept_count = k_blk - current_drop_mask.sum(dim=-1)
            restore_count = torch.minimum(per_iter_count, current_drop_mask.sum(dim=-1))
            restore_count = torch.minimum(restore_count, current_kept_count)
            restore_count = torch.where(has_beneficial_candidate, restore_count, torch.zeros_like(restore_count))
            restore_mask = _top_m_mask(restore_quality, restore_count, k_max=round_k_max)

            drop_mask_after_restore = current_drop_mask & ~restore_mask
            current_kept_mask = ~current_drop_mask
            # Compensate. Delta'_j = ||g+d_j||^2_M - ||g||^2_M = -2*raw_dot + s_j^2, and the
            # kept mass drops by P_j -- the same exact marginal as the restore side, applied in
            # the opposite direction (which is why there is no separate "score" criterion: the
            # two sides have to be scored in the same currency). Pick the `restore_count` KEPT
            # candidates whose removal costs least, i.e. the largest quality.
            compensate_quality = kept_mass_norm_quality(
                g_norm_sq.to(_knorm_dtype), kept_mass.to(_knorm_dtype),
                s_sq.to(_knorm_dtype) - 2.0 * raw_dot.to(_knorm_dtype),
                -P.to(_knorm_dtype))
            if maximize:
                compensate_quality = -compensate_quality
            compensate_quality = compensate_quality.masked_fill(~current_kept_mask, float("-inf"))
            extra_drop_mask = _top_m_mask(compensate_quality, restore_count, k_max=round_k_max)

        if thrash_initial_drop_mask is not None:
            thrash_gross += restore_count.sum().item()
        if this_call_diag is not None:
            # M-weighted cosine similarity between THIS round's g and the PREVIOUS round's g
            # (the question being: does drift's realized MAGNITUDE decrease reflect real
            # convergence, or is the restore/compensate swap just cancelling a different-DIRECTION
            # g each round while total ||g||^2_M happens to trend down on average -- i.e. "chasing"
            # rather than converging). cos_M = (g_prev . M g_curr) / (||g_prev||_M * ||g_curr||_M);
            # reuses the already-computed `Mg` (no extra matmul). NaN for round 0 (no previous g) and
            # for degenerate all-zero rows (row is fully kept, drop-set empty -- 0/0).
            if prev_g is None:
                cos_m_prev = torch.full_like(g_norm_sq, float("nan"))
            else:
                cross_diag = (prev_g * Mg).sum(dim=-1)
                cos_m_prev = cross_diag / (prev_g_norm_sq.clamp(min=1e-12).sqrt() * g_norm_sq.clamp(min=1e-12).sqrt())
            if ref_g is None:
                cos_m_ref = torch.full_like(g_norm_sq, float("nan"))
                ref_g, ref_g_norm_sq = g.detach(), g_norm_sq.detach()
            else:
                cross_ref = (ref_g * Mg).sum(dim=-1)
                cos_m_ref = cross_ref / (ref_g_norm_sq.clamp(min=1e-12).sqrt() * g_norm_sq.clamp(min=1e-12).sqrt())
            prev_g, prev_g_norm_sq = g.detach(), g_norm_sq.detach()
            # per_iter_count logged AFTER the floor (this_call_diag's whole purpose is diagnosing the
            # floor's effect -- see the floor's own comment above -- so it must reflect what actually
            # got swapped, not the pre-floor requested count).
            this_call_diag.append((g_norm_sq.detach().float().mean().item(), restore_count.float().mean().item(),
                                    cos_m_prev.detach().float().nanmean().item(),
                                    cos_m_ref.detach().float().nanmean().item()))

        current_drop_mask = (fused_new_drop if fused_new_drop is not None
                              else drop_mask_after_restore | extra_drop_mask)

    if track_best:
        # The loop's own in-round check only ever scores the state BEFORE that round's
        # decision -- the LAST round's resulting state never got compared, so score it
        # here once more before picking the final per-row winner.
        g_norm_sq = _row_objective(current_drop_mask)
        better = g_norm_sq > best_g_norm_sq if maximize else g_norm_sq < best_g_norm_sq
        best_drop_mask = torch.where(better.unsqueeze(-1), current_drop_mask, best_drop_mask)
        current_drop_mask = best_drop_mask

    if thrash_initial_drop_mask is not None:
        thrash_net = (thrash_initial_drop_mask != current_drop_mask).float().sum().item()
        _thrash_diag_records.append({
            "gross": thrash_gross, "net": thrash_net,
        })

    # Kept as a 0-dim TENSOR, not a Python float: `.item()` here was one
    # more GPU->CPU sync per attention layer, and the value is only ever broadcast-added
    # to `s` -- nothing needs it on the host. Same numeric result either way.
    KEEP_OFFSET = (s.detach().abs().amax() + 1.0) if s.numel() > 0 else 1.0
    rank_by = torch.where(current_drop_mask, s, s + KEEP_OFFSET)

    return rank_by


class ProjectedScorer(BlockScorer):
    def __init__(self, M):
        self.M = M  # (n_blocks, nh, hd, hd) or (nh, hd, hd) for a single layer

    def score(self, attn_pooled, v_bar, ctx=None):
        M = self.M
        if ctx is not None and "layer_idx" in ctx and M.dim() == 4:
            M = M[ctx["layer_idx"]]
        return projected_score(attn_pooled, v_bar, M)
