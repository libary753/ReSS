"""ReSS block-sparse global attention forward pass.

This module replaces the global-attention forward of a wired backbone. It predicts a
pooled block-level attention map, hands that map to a scorer and a budget to decide which
key blocks each (head, query-block) keeps, and runs the resulting mask through the
SpargeAttn block-sparse CUDA kernel.

The budget decision deliberately does NOT live here. This module only ever receives an
already-decided per-head block budget (`ress.budget`), so adding an allocation rule does
not mean editing the forward pass -- which is what makes HeSS and the uniform baseline the
same code path with a different budget object.

The block-map construction and the kernel call follow the SparseVGGT reference
implementation (https://github.com/brianwang00001/sparse-vggt, `models/attention.py` and
`utils/sparse_wrapper.py`); the scoring, the budget hooks and the restore-swap path are
this project's. That repository carries no licence; see this project's README.

`spas_sage_attn` is called directly, and no part of it is wrapped or re-implemented.
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from spas_sage_attn._qattn import (
    qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold as _sparge_kernel,
)
from spas_sage_attn.quant_per_block import per_block_int8
from spas_sage_attn.utils import block_map_lut_triton, fill_block_map_triton, hyperparameter_check

from ress.models.qkv import combine_special_and_patch, compute_patch_qkv, project_out
from ress.scoring import criterion as _criterion

KS_Q = 128
KS_K = 64
KBLK = 64  # SpargeAttn's hardcoded key-block size (sparse_wrapper.py block_sparse_attn_cuda)
MIN_NUM_TO_SELECT = 2  # floor on real (non-sink) key-blocks per query row, see
# ress_attention_forward's clamp -- single-block selections were found to trigger
# NaN in SpargeAttn's int8 quantization under some allocator+aspect-ratio combos.

# TWO independent axes, on TWO switches. They were one flag until the eval made
# it clear that "how much did each cost" is a question the ReSS paper has to answer
# separately,
# and a single kill-switch could only ever price them together. They ended up with OPPOSITE
# defaults, which is the whole reason they had to be split.
#
# `RESS_SINK_MASS=1` -- pool the attention map over the kernel's real key
#   `cat([k_patch, k_special])` instead of over `k_patch` alone, so the scoring math sees the
#   sink blocks' probability mass. DEFAULT OFF, i.e. `P` is deliberately the
#   patch-CONDITIONAL distribution. This is NOT an approximation we settled for: the layer's
#   output error decomposes EXACTLY into an aggregation error (the part the selector steers)
#   and a mixture shift toward the sink value that is rank-1 per row and depends on the drop
#   set only through the total mass dropped -- i.e. unsteerable at a fixed budget. Minimizing
#   only the first term is what the patch-conditional map computes, and it is an upper bound
#   on the steerable damage (see `scoring.projected.dense_reference`).
#
#   Turning it ON was measured to cost ScanNet++ (20 scenes, 100 views, sparsity 0.745)
#   F1 0.628 -> 0.575 and pose AUC@30 0.919 -> 0.843, with 2 of 20 scenes COLLAPSING to F1
#   0.02/0.27: in rows where the sink mass beta approaches 1, the reference is dragged onto
#   the sink value, the 1/(1-P) protection of a dominant candidate goes flat, and K's floor
#   rises -- all three at once. DTU (22 scenes, 49 views) mildly prefers it ON
#   (Chamfer 0.765 vs 0.783, pose tied to 3e-4); the asymmetry in magnitude, and the absence
#   of a catastrophic tail, is what decided the default. Set =1 to reproduce that arm.
#
# `RESS_SINK_PROTECT=0` -- go back to SparseVGGT's candidate rule, `ceil(T_patch/64)`, which
#   protects only blocks lying ENTIRELY past the patch keys. With it OFF the boundary block
#   holding the patch tail plus the leading special keys is an ordinary drop candidate, which
#   leaks 48 of 60 special keys at VGGT/12-view and all of them on a one-special-token-per-
#   view backbone. This is a CORRECTNESS axis (default ON, and measured performance-neutral:
#   F1 within +-0.003 with the sign split across scenes); see
#   `tests/test_special_key_protection.py`.
#
# All four combinations are coherent: `n_cand` always marks how many leading blocks of the
# pooled map are candidates, and whatever follows is protected, whichever pooling produced it.
_SINK_MASS = os.environ.get("RESS_SINK_MASS", "0") != "0"
_SINK_PROTECT = os.environ.get("RESS_SINK_PROTECT", "1") != "0"


def predict_attention(query, key, ks_q=KS_Q, ks_k=KS_K, pool_mode="avg"):
    """Pooled approximate attention map P^h = softmax(P_b(Q^h) P_b(K^h)^T).
    query: (B, nh, Tq, C), key: (B, nh, Tk, C) -> pooled_prob: (B, nh, q_blk, k_blk)."""
    assert pool_mode in ("max", "avg")
    pooling_fn = {"max": F.max_pool1d, "avg": F.avg_pool1d}[pool_mode]
    B, nh, _Tq, C = query.shape
    _, _, _Tk, _ = key.shape

    q = rearrange(query, "B nh Tq C -> (B nh) C Tq")
    pooled_q = pooling_fn(q, kernel_size=ks_q, ceil_mode=True)
    pooled_q = rearrange(pooled_q, "(B nh) C Tq -> B nh Tq C", B=B, nh=nh)

    k = rearrange(key, "B nh Tk C -> (B nh) C Tk")
    pooled_k = pooling_fn(k, kernel_size=ks_k, ceil_mode=True)
    pooled_k = rearrange(pooled_k, "(B nh) C Tk -> B nh Tk C", B=B, nh=nh)

    scale = 1 / math.sqrt(C)
    pooled_score = pooled_q @ pooled_k.transpose(-1, -2) * scale
    return F.softmax(pooled_score, dim=-1)


def block_mean_value(value, ks_k=KS_K):
    """V̄: per-key-block mean of V, needed for the ΔO block-approximation closed form.
    value: (B, nh, Tk, hd) -> (B, nh, k_blk, hd)."""
    B, nh, _Tk, _hd = value.shape
    v = rearrange(value, "B nh Tk hd -> (B nh) hd Tk")
    pooled_v = F.avg_pool1d(v, kernel_size=ks_k, ceil_mode=True)
    return rearrange(pooled_v, "(B nh) hd kb -> B nh kb hd", B=B, nh=nh)


def same_frame_block_mask(q_blk, k_blk, N, H, W, device):
    """(q_blk, k_blk) bool -- True where query-block q and key-block k overlap the
    SAME frame(s) in the frame-major-flattened N*(H*W) patch token sequence (frame
    f occupies token positions [f*H*W, (f+1)*H*W)). Blocks are formed by pooling
    that flattened sequence with a fixed kernel (KS_Q for query, KS_K for key,
    ceil_mode=True) -- since H*W is not generally a multiple of KS_Q/KS_K, a block
    CAN straddle a frame boundary, so this computes genuine interval overlap (both
    frames a straddling block touches count), not a single frame index per block.

    "Own-frame" protection: every query can always attend to at least its own frame's keys,
    regardless of what any block_criterion/selector decided. Query-block-DEPENDENT
    (unlike `sink_blocks`, which protects the SAME fixed key-blocks for every
    query), so it's applied as a post-hoc BUDGET-NEUTRAL SWAP on `final_map`
    (see `ress_attention_forward`'s `protect_own_frame` docstring) rather than
    folded into `sink_blocks` itself. Pure position arithmetic (N/H/W only, no tensor
    data), same for every layer given one input's shape -- cheap to recompute per
    call, no caching needed."""
    HW = H * W
    T = N * HW
    q_start = torch.arange(q_blk, device=device) * KS_Q
    q_end = (q_start + KS_Q - 1).clamp(max=T - 1)
    q_frame_start = q_start // HW
    q_frame_end = q_end // HW

    k_start = torch.arange(k_blk, device=device) * KS_K
    k_end = (k_start + KS_K - 1).clamp(max=T - 1)
    k_frame_start = k_start // HW
    k_frame_end = k_end // HW

    # interval overlap: NOT (q's frame range ends before k's starts, OR vice versa)
    no_overlap = (q_frame_end.unsqueeze(1) < k_frame_start.unsqueeze(0)) | \
                 (k_frame_end.unsqueeze(0) < q_frame_start.unsqueeze(1))
    return ~no_overlap


def build_block_mask(pooled_score, num_to_select, sink_blocks, rank_by=None):
    """Ported from `get_block_mask`'s tail (sparse_wrapper.py L167-202), taking an
    already-decided per-(head,query-block) token budget instead of computing it
    in-place from sparse_ratio/cdf_threshold/head_attr_scores/beta -- that decision
    now lives in `ress.allocation`.

    pooled_score: (B, nh, q_blk, k_blk); num_to_select: (B, nh, q_blk) long tensor.
    rank_by: optional (B, nh, q_blk, k_blk) alternative ranking key. This is the second
    of the method's two independent handles -- HOW MANY blocks a head keeps (the budget)
    and WHICH ONES it keeps (this) -- and separating them is what lets HeSS and ReSS be
    compared one axis at a time. None keeps the baseline (probability-descending, i.e.
    highest pooled_score kept first); passing `damage.damage_score(pooled_score,
    v_bar)` instead switches to damage-ascending cutoff (keep highest predicted-harm
    blocks first) -- an INDEPENDENT knob from `num_to_select` (the head-budget
    handle), so e.g. "uniform head budget + damage-ordered blocks" isolates the
    block-cutoff criterion's own contribution.
    Returns: (B, nh, q_blk, k_blk + sink_blocks) bool.
    """
    key = pooled_score if rank_by is None else rank_by
    sorted_score = torch.sort(key, dim=-1, descending=True)
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if sink_blocks > 0:
        shape = list(final_map.shape)
        shape[-1] = sink_blocks
        trailing_ones = torch.ones(shape, device=final_map.device).bool()
        final_map = torch.cat([final_map, trailing_ones], dim=-1)
    return final_map


_RANDOM_SEED = 20260826  # `block_criterion="random"`; dated, like `methods._SHUFFLE_SEED`
_random_gens = {}


def _random_gen(device):
    """One RNG per device for `block_criterion="random"`, seeded once at first use and
    then advanced by every draw. Deterministic given the same call ORDER, which a fixed
    (method, dataset, scene, grid point) run has -- so a re-run reproduces its own number
    -- but deliberately NOT the same pattern in every layer, which reseeding per call
    would give and which would make the arm a fixed mask rather than a random one.

    A random ranking key is not the same thing as a random KEPT SET drawn uniformly: it
    is, because `build_block_mask` takes each row's top `num_to_select` of the key and
    the key is exchangeable, so every subset of that size is equally likely. The budget
    and the sinks are untouched, which is the whole point of this arm."""
    gen = _random_gens.get(device)
    if gen is None:
        gen = torch.Generator(device=device).manual_seed(_RANDOM_SEED)
        _random_gens[device] = gen
    return gen


def _record_realized_drift(M, attn_full, v_bar_full, final_map, n_cand):
    """Append this layer's mean realized `J = ||g||^2_M / K^2` to whatever list
    `scoring.projected.record_realized_drift()` has active, or do nothing (the production
    path). `v_bar_full` is the caller's memoizing closure, not a tensor, so an inactive
    recorder does not force the pooled values to be materialized.

    The import is function-local on purpose: `scoring.projected` imports `models.wiring`,
    which imports this module, so a module-level import here would close that cycle."""
    from ress.scoring import projected as _projected

    if _projected._realized_records is None or M is None:
        return
    with torch.no_grad():
        J = _projected.realized_objective(attn_full, v_bar_full(), M,
                                           final_map[..., :n_cand], n_cand=n_cand)
    _projected._realized_records.append(J.float().mean().item())


def run_block_sparse_kernel(query, key, value, final_map, dtype=torch.float16, out_dtype=torch.float32):
    """Ported from `block_sparse_attn_cuda`'s kernel-calling tail (sparse_wrapper.py
    L257-299), given an already-built boolean block mask instead of building one
    in-place. query/key/value: (B, nh, T*, hd). final_map: (B, nh, q_blk, k_blk+sink).
    Returns: (o, sparsity)."""
    _is_causal = 0
    pvthreshd = hyperparameter_check(1e10, query.size(-3), query.device)
    lut, valid_block_num = block_map_lut_triton(final_map)

    query, key, value = (
        query.contiguous().to(dtype),
        key.contiguous().to(dtype),
        value.contiguous().to(dtype),
    )
    km = key.mean(dim=-2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(query, key - km)
    q_scale, k_scale = q_scale.squeeze(-1), k_scale.squeeze(-1)

    hd = query.shape[-1]
    scale = 1.0 / (hd**0.5)
    o = torch.empty_like(query)
    _sparge_kernel(
        q_int8, k_int8, value, o, lut, valid_block_num, pvthreshd,
        q_scale, k_scale, 1, _is_causal, 1, scale, 0,
    )
    sparsity = 1 - final_map.float().mean().item()
    return o.to(out_dtype), sparsity


@dataclass
class AttentionAux:
    """Per-layer auxiliary values exposed for ReSS damage computation. Populated
    only when `aux_store` is passed to `ress_attention_forward` (probe/analysis
    calls); the plain runtime path (aux_store=None) skips populating these to avoid
    materializing q_patch/attn_pooled/v_bar every layer at inference time."""

    attn_pooled: torch.Tensor = None  # P^h over the DROPPABLE candidate blocks, (B, nh, q_blk, n_cand)
    v_bar: torch.Tensor = None  # V̄ for those blocks, (B, nh, n_cand, hd)
    attn_full: torch.Tensor = None  # P^h over ALL key-blocks incl. sinks, (B, nh, q_blk, k_blk_total)
    v_bar_full: torch.Tensor = None  # V̄ for those, (B, nh, k_blk_total, hd)
    n_cand: int = None  # how many leading blocks of the `_full` pair are candidates; the
    # rest are the kernel's always-kept sink blocks. `attn_pooled`/`v_bar` are exactly the
    # leading `n_cand` slices, so a consumer that only ranks candidates needs no change --
    # but note the values are now the TRUE probabilities restricted to the candidates (row
    # sum = 1 - sink mass), not a patch-conditional softmax (row sum = 1).
    q_patch: torch.Tensor = None
    shape: dict = None
    sparsity: float = None
    output: torch.Tensor = None  # O_h, (B, nh, N*P, hd) -- the tensor damage.py hooks the gradient of


def compute_baseline_num_to_select(pooled_score, sparse_ratio=None, cdf_threshold=None, eps=1e-5):
    """Ported from `get_block_mask`'s num_to_select computation (sparse_wrapper.py
    L156-186, minus the head_attr_scores reallocation, which is a separate step --
    `allocation.proportional.adjust_attention_selection`): baseline per-query-block
    budget from sparse_ratio (uniform topk floor) and/or cdf_threshold (adaptive,
    per-query cumulative-probability cutoff). Needs the ACTUAL attn_pooled from this
    input (cdf selection is content-dependent), unlike ReSS's own water-filling
    budgets which are precomputed externally from a probe pass -- see
    `ress_attention_forward`'s `budget_fn` hook for how this plugs in for a
    single-pass (HeSS-style) forward."""
    B, nh, q_blk, k_blk = pooled_score.shape
    num_to_select = None
    if cdf_threshold is not None:
        sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
        cdf = torch.cumsum(sorted_score.values, dim=-1)
        cdfthreshd = torch.as_tensor(cdf_threshold, device=pooled_score.device).view(1, 1, 1, 1)
        cdfthreshd = (cdfthreshd + eps).expand(B, nh, q_blk, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdfthreshd, right=True).squeeze(-1)
    if sparse_ratio is not None:
        topk = int(k_blk * (1 - sparse_ratio))
        if num_to_select is None:
            num_to_select = torch.full((B, nh, q_blk), topk, device=pooled_score.device)
        else:
            num_to_select = torch.clamp(num_to_select, min=topk)
    assert num_to_select is not None, "at least one of sparse_ratio/cdf_threshold required"
    return num_to_select


def ress_attention_forward(
    self,
    x,
    pos,
    num_to_select=None,
    budget_fn=None,
    perhead_budget_fn=None,
    joint_select_fn=None,
    block_criterion="probability",
    pool_mode="avg",
    num_special_tokens=5,
    num_heads=16,
    shape=None,
    aux_store: dict | None = None,
    aux_full: bool = True,
    dense: bool = False,
    retain_output_grad: bool = False,
    pooled_fallback: bool = False,
    protect_own_frame: bool = False,
):
    """Replacement forward for a VGGT global-attention block.

    Args:
        num_to_select: (B, nh, q_blk) long tensor, per-(head,query-block) key-block
            budget (from `ress.budget`, precomputed offline/from a prior probe
            pass -- ReSS's own two-pass runtime). Required unless `dense=True`,
            `budget_fn`, or `joint_select_fn` is given.
        budget_fn: alternative to `num_to_select` for single-pass (HeSS-style)
            use: callable(attn_pooled) -> num_to_select, invoked with THIS layer's
            just-computed P^h (content-dependent budget decisions, e.g.
            `compute_baseline_num_to_select` + `budget._proportional.adjust_attention_selection`,
            for single-pass static-score inference).
        perhead_budget_fn: per-head budget hook, takes
            precedence over `budget_fn` when both are set: callable(attn_pooled,
            v_bar) -> num_to_select, same (B, nh, q_blk) contract as `num_to_select`
            itself, but needs V̄ (per-key-block mean value) too since the per-head
            budget there is a cost computed FROM v_bar (`selection.beta_budget`),
            not just the pooled attention map.
        pooled_fallback: reserved for a not-yet-implemented mixed-resolution
            fallback (dropped key-blocks contribute a pooled token instead of
            vanishing)
            -- currently a no-op; every caller in this codebase passes False.
        joint_select_fn: one-shot joint-selection path, mutually exclusive with
            `num_to_select`/`budget_fn`: callable(attn_pooled, v_bar, sink_blocks) ->
            final_map ((nh, q_blk, k_blk+sink_blocks) bool, True=KEEP, sink already
            appended). See `runtime.joint.set_joint_budget` for the per-layer wiring
            (a `scoring.BlockScorer` + `selection.topk_joint.TopKJointSelector`
            closure) and `selection.topk_joint`'s module docstring for why this
            bypasses `num_to_select`/`build_block_mask` entirely instead of
            expressing joint selection as a `budget_fn` variant.
        block_criterion: "probability" (the baseline order, used by SparseVGGT and HeSS),
            "damage" (the second handle described above: rank candidate
            key-blocks by `damage.damage_score(attn_pooled, v_bar)` instead of raw
            attention probability -- independent of `num_to_select`/`budget_fn`,
            the head-budget handle), "norm" (ablation: `damage.norm_score`,
            ranks by ||v|| alone, no distance-to-mean term), or "projected"
            (`projected.projected_score`,
            P·||W_O^h e|| -- requires `scoring.projected.bake_and_attach_M(model)`
            to have set `self.M` on every attn module first), or
            "random" (negative-control REFERENCE for the drift-maximization sanity
            check: an i.i.d. uniform ranking key, so the head budget and
            the sink protection are unchanged and only WHICH blocks fill the budget
            becomes uninformed. It is what puts the minimizing and maximizing arms on
            a scale -- "better than Ours" and "worse than the anti arm" are both
            claims about a middle, and without one measured the experiment only shows
            that two extremes differ). The maximizing arms themselves are restore-swap
            names carrying `_antiswap`/`_anti`; see `scoring.criterion`'s module
            docstring. Ignored when `joint_select_fn` is given (the scorer passed to
            that closure plays the same role).

            An earlier `"projected_inverted"` documented here was deleted in the
            an earlier cleanup and is NOT accepted -- `_anti` supersedes it, and unlike
            it does not conflate the score's sign with dropping the restore-swap rounds
            entirely.
        aux_store: if not None, populated with an `AttentionAux` (P^h, V̄, q_patch,
            shape) for ReSS damage/probe computation. `output` always requires this
            path to run with `torch.enable_grad()` around the caller if a gradient
            probe is being backed through it.
        aux_full: if False, `aux_store["value"]` only gets `sparsity`/`shape` (all
            other `AttentionAux` fields left None) -- for callers (e.g. sweep scripts
            that only read `.sparsity` off every layer) that don't need attn_pooled/
            v_bar/q_patch/output. Those full-size tensors, if populated, stay alive
            in `aux_store` for every one of the model's global blocks simultaneously
            across a single forward pass (nothing frees layer i's entry when layer
            i+1 runs) -- a real memory multiplier at high view counts that has
            nothing to do with the sparse attention computation itself. Ignored (has
            no effect) when `aux_store is None`.
        dense: if True, ignore num_to_select and run exact dense SDPA (used by
            `dense`/uniform-vs-none baselines and analysis paths that need the exact,
            unsparsified O_h for reference).
        protect_own_frame: when True, guarantees
            every query-block can always attend to at least its own frame's
            key-blocks (`same_frame_block_mask`), via a BUDGET-NEUTRAL swap
            against whatever `final_map` the path above built: for each missing
            own-frame block, the lowest-scoring currently-kept NON-own-frame
            block is swapped out, one-for-one, so the total kept count per row
            (and hence measured sparsity) is unchanged from the no-protection
            case -- unlike an earlier version of this flag, which OR'd the mask
            in for free and measurably lowered sparsity. Default False -- every
            existing method's behavior/measured numbers are unchanged unless this
            is explicitly opted into. Ignored when `dense=True` (nothing to mask).
    """
    qkv = compute_patch_qkv(self, x, pos, num_special_tokens=num_special_tokens, num_heads=num_heads,
                             shape=shape)
    q_patch, k_patch, v_patch = qkv.q_patch, qkv.k_patch, qkv.v_patch
    shp = qkv.shape
    B, N, P, hd, nh = shp["B"], shp["N"], shp["P"], shp["head_dim"], shp["num_heads"]

    if qkv.q_special is not None:
        key = torch.cat([k_patch, qkv.k_special], dim=-2)
        value = torch.cat([v_patch, qkv.v_special], dim=-2)
        # mathematically == F.scaled_dot_product_attention(q_special, k, v) with the
        # original (unsplit) k/v: same key/value set, softmax is order-invariant.
        x_special = F.scaled_dot_product_attention(qkv.q_special, key, value)
    else:
        x_special = None
        key, value = k_patch, v_patch

    # Pooled map on the kernel's OWN key-block grid. The kernel attends over
    # `key = [k_patch | k_special]` and blocks THAT tensor by 64, keeping the trailing
    # special blocks unconditionally, so pooling `k_patch` alone (what this line did until
    # produced the patch-CONDITIONAL distribution: the sinks were missing from
    # the probabilities, from the reference V̄, and from the kept mass. Measured sink share
    # of the pooled mass: 4.9% on average over VGGT's global layers, up to 10.9% for a whole
    # layer and >0.92 for individual query blocks (measured on CO3Dv2).
    #
    # Pooling the concatenated key also puts the last candidate block on the kernel's real
    # boundary: when the patch keys do not end on a multiple of 64 that block STRADDLES,
    # holding the patch tail plus the leading special keys, which the patch-only pooling
    # could not represent at all.
    #
    # `n_cand` is the number of DROPPABLE blocks; the rest are the always-kept tail.
    #
    # A block is a candidate only if it holds patch keys EXCLUSIVELY. That is stricter
    # than the upstream rule and fixes a real boundary bug inherited from SparseVGGT
    # (SparseVGGT's own `get_block_mask`), whose intent is
    # explicit -- `# Always select special tokens/blocks` -- but whose implementation
    # protects only `sink_blocks = ceil(Tk/64) - ceil(T_patch/64)`, i.e. blocks lying
    # ENTIRELY past the patch keys. When `T_patch` is not a multiple of 64 the boundary
    # block holds the patch tail AND the leading special keys, is counted inside
    # `orig_kblk`, and is therefore an ordinary drop candidate. At VGGT / 12 views /
    # 518^2 / patch 14 that leaks 20 of 60 special keys; on a backbone with one special
    # token per view (DA3) `sink_blocks` comes out 0 and ALL of them leak, while its pose
    # head reads exactly those tokens.
    #
    # Taking the FLOOR moves that straddling block into the protected tail, which also
    # keeps it out of the scoring math -- its pooled key averages patch and special keys
    # together and so represents neither, making any score computed on it unreliable in
    # either direction. Where `T_patch` IS a multiple of 64, floor == ceil and nothing
    # changes. Without special tokens there is no boundary, so the ceiling is correct.
    has_special = qkv.q_special is not None
    # WHAT the map is pooled over (`_SINK_MASS`) and WHICH of its blocks may be dropped
    # (`_SINK_PROTECT`) are independent -- see the two flags' comment at the top of this
    # module. `_SINK_MASS=0` reproduces the patch-conditional map; `_SINK_PROTECT=0`
    # reproduces SparseVGGT's leaking candidate rule.
    attn_full = predict_attention(
        query=q_patch, key=(key if _SINK_MASS else k_patch), pool_mode=pool_mode)
    if has_special and _SINK_PROTECT:
        n_cand = k_patch.shape[-2] // KS_K
    else:
        n_cand = math.ceil(k_patch.shape[-2] / KS_K)
    attn_pooled = attn_full[..., :n_cand]
    # Budgets read a PATCH-CONDITIONAL copy, the scoring reads the true probabilities.
    # Different consumers, different correct inputs:
    #
    # SparseVGGT defines its CDF budget on `predict_attention(q_patch, k_patch)` -- a
    # softmax over the patch keys alone, so the row sums to 1 and `cdf_threshold` means
    # "capture tau of the PATCH attention mass", with the special keys outside that
    # accounting entirely (`sparse_wrapper.py::get_block_mask`). Handing the budget the
    # sink-aware map instead makes the candidate row sum to alpha = 1 - beta ~ 0.93, so a
    # tau of 0.97 can never be reached and the budget silently inflates: measured -3.1pp
    # to -5.1pp of realized sparsity at the same grid point, i.e. the baseline stops being
    # the baseline. Renormalising back to 1 reproduces SparseVGGT exactly, whatever the
    # scoring does with the sinks.
    #
    # `sparse_ratio`-only budgets size off `k_blk` and are unaffected either way; this
    # matters for the CDF path, and for `HessBudget`, which reads the same axis.
    attn_cond = attn_pooled / attn_pooled.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    _v_bar_full = None

    def v_bar_full():
        """`block_mean_value` over the same grid, memoized -- several branches below want
        it and the dense path wants none of them."""
        nonlocal _v_bar_full
        if _v_bar_full is None:
            _v_bar_full = block_mean_value(value if _SINK_MASS else v_patch)
        return _v_bar_full

    if dense:
        x_patch = F.scaled_dot_product_attention(q_patch, key, value)
        sparsity = 0.0
    else:
        Tk = key.shape[-2]
        orig_kblk = attn_pooled.shape[-1]
        total_kblk = math.ceil(Tk / KBLK)
        sink_blocks = total_kblk - orig_kblk
        assert not _SINK_MASS or attn_full.shape[-1] == total_kblk, (
            f"pooled grid {attn_full.shape[-1]} != kernel grid {total_kblk} -- "
            "predict_attention's ks_k and the kernel's KBLK must agree")
        norm_correction = None  # set only by "restore_swap_angle*" (block_criterion branch below) --
        # hoisted above the joint_select_fn/block_criterion split (fix: was declared only
        # inside the block_criterion `else` branch, so the joint_select_fn path -- pooled_restore_swap's
        # own hook -- hit an UnboundLocalError at this function's later `if norm_correction is not None`
        # check; pre-existing bug, unrelated to any pooled_restore_swap_select change, just never
        # exercised through a real forward pass since "restore_swap_angle"/norm_correction was added).
        if joint_select_fn is not None:
            # A single budget shared jointly across heads per
            # query-block, built by `selection.topk_joint` (already includes its own
            # MIN_NUM_TO_SELECT-equivalent floor -- see that module's docstring for
            # why the same NaN-under-starvation risk applies here). Bypasses the
            # per-head `num_to_select`/`build_block_mask` path entirely; produces the
            # same `final_map` contract so `run_block_sparse_kernel` needs no change.
            final_map = joint_select_fn(attn_cond, v_bar_full()[..., :n_cand, :], sink_blocks)
        else:
            if perhead_budget_fn is not None:
                num_to_select = perhead_budget_fn(attn_cond, v_bar_full()[..., :n_cand, :])
            elif budget_fn is not None:
                num_to_select = budget_fn(attn_cond)
            assert num_to_select is not None, "num_to_select (or budget_fn/perhead_budget_fn) required unless dense=True or joint_select_fn is given"
            # Never let a query row select fewer than MIN_NUM_TO_SELECT real key-blocks --
            # found empirically (PROGRESS.md §Phase 2.6d) that a head pinned to a single
            # 64-key block (whichever allocator produced that -- static beta-reallocation
            # or a precomputed per-input tensor, both land here) can make SpargeAttn's
            # per_block_int8 quantization scale degenerate on that one low-variance block,
            # producing NaN downstream -- worse when sink_blocks (the always-kept special-
            # token blocks) also happens to be 0 for a given image aspect ratio, so there's
            # no fallback context at all. Clamping to >=2 blocks (or orig_kblk itself, if
            # smaller) costs a few extra keys per row but keeps the kernel in a regime that
            # hasn't been observed to diverge.
            num_to_select = num_to_select.clamp(min=min(MIN_NUM_TO_SELECT, orig_kblk))
            rank_by = None
            if block_criterion == "projected":
                # The metric correction at the heart of ReSS: P·||W_O^h e|| instead
                # of the raw head-space P·||e|| `damage` uses. Needs `self.M` (this head's
                # (hd,hd) Mahalanobis matrix) pre-attached -- see
                # `scoring.projected.bake_and_attach_M` (scene-independent, baked once per
                # model load from `self.proj.weight`).
                from ress.scoring import projected as _projected

                assert getattr(self, "M", None) is not None, (
                    "block_criterion='projected' requires scoring.projected.bake_and_attach_M(model) "
                    "to have been called first (self.M not found on this attention module)"
                )
                rank_by = _projected.projected_score(attn_full, v_bar_full(), self.M,
                                                      n_cand=n_cand)
            elif _criterion.is_restore_swap(block_criterion):
                # Restore-and-swap drift correction, which replaced an earlier "two_pass"
                # design after that one overshot at high sparsity: full one-shot drop by
                # the plain per-candidate score, then restore whichever dropped
                # candidates contributed most to the realized drift, compensated by
                # newly dropping an equal count of currently-KEPT candidates so the
                # budget stays exact -- see `scoring.projected.restore_swap_rank_by`'s
                # docstring for the derivation and for each option's own history.
                #
                # The name grammar (n_iters, W_o on/off, bf16, track_best) is parsed by
                # `scoring.criterion.parse_restore_swap`.
                from ress.scoring import projected as _projected

                spec = _criterion.parse_restore_swap(block_criterion)
                if spec.use_wo:
                    assert getattr(self, "M", None) is not None, (
                        f"block_criterion={block_criterion!r} requires scoring.projected.bake_and_attach_M(model) "
                        "to have been called first (self.M not found on this attention module)"
                    )
                    M = self.M
                else:
                    # M=identity reduces projected_score EXACTLY to `scoring.damage.damage_score`
                    # (tests/test_projected.py::test_projected_score_matches_damage_score_when_M_is_identity),
                    # so the same restore/swap machinery runs on Rung 1's head-space drift
                    # instead of the residual-space one -- no `self.M`/bake_and_attach_M
                    # prerequisite at all. This is the default recipe for backbones other than
                    # VGGT, and matches the "damage_restore_swap*" spelling exactly.
                    M = torch.eye(hd, device=attn_pooled.device, dtype=attn_pooled.dtype).expand(nh, hd, hd)
                rank_by = _projected.restore_swap_rank_by(
                    attn_full, v_bar_full(), M, num_to_select, n_cand=n_cand,
                    n_iters=spec.n_iters, track_best=spec.track_best,
                    big_dtype=spec.big_dtype, restore_frac=spec.restore_frac,
                    decay=spec.decay,
                    maximize=spec.maximize, invert_cut=spec.invert_cut,
                )
            elif block_criterion == "random":
                rank_by = torch.rand(attn_cond.shape, device=attn_cond.device,
                                      dtype=torch.float32, generator=_random_gen(attn_cond.device))
            elif block_criterion != "probability":
                raise ValueError(block_criterion)
            final_map = build_block_mask(attn_cond, num_to_select, sink_blocks, rank_by=rank_by)

        if protect_own_frame:
            # Keeps the realized sparsity from sagging: BUDGET-
            # NEUTRAL swap, not a free OR-in -- a plain OR (the original
            # implementation) adds own-frame blocks ON TOP of num_to_select's
            # budget, which measurably lowers sparsity (confirmed empirically:
            # 0.7229->0.7057 at a fixed sparse_ratio). Instead, for every
            # (head, query-block) row: for each own-frame key-block NOT already
            # kept ("missing"), swap out the LOWEST-attn_pooled-score currently-
            # kept NON-own-frame block to make room -- one-for-one, so the total
            # kept count per row (and hence measured sparsity) is EXACTLY
            # unchanged from the no-protection case. If a row doesn't have enough
            # swappable (non-own-frame, kept) candidates to fully cover its
            # missing own-frame blocks (only possible when num_to_select is
            # smaller than one frame's own block count -- very high sparsity),
            # only as many as are swappable get swapped in, prioritized by
            # highest attn_pooled score among the missing ones.
            q_blk = attn_pooled.shape[-2]
            same_frame = same_frame_block_mask(q_blk, orig_kblk, shp["N"], shp["H"], shp["W"], attn_pooled.device)
            same_frame_b = same_frame.unsqueeze(0).unsqueeze(0).expand_as(final_map[..., :orig_kblk])

            kept_patch = final_map[..., :orig_kblk]
            missing = same_frame_b & ~kept_patch  # own-frame, not currently kept
            swappable = kept_patch & ~same_frame_b  # currently kept, NOT own-frame -- eligible to swap out
            n_missing = missing.sum(dim=-1)
            n_swap = torch.minimum(n_missing, swappable.sum(dim=-1))

            swap_out_priority = attn_pooled.masked_fill(~swappable, float("inf"))  # lowest score swapped out first
            rank_swap_out = torch.argsort(torch.argsort(swap_out_priority, dim=-1, descending=False), dim=-1)
            swap_out_mask = rank_swap_out < n_swap.unsqueeze(-1)

            missing_priority = attn_pooled.masked_fill(~missing, float("-inf"))  # highest score swapped in first
            rank_missing = torch.argsort(torch.argsort(missing_priority, dim=-1, descending=True), dim=-1)
            swap_in_mask = rank_missing < n_swap.unsqueeze(-1)

            new_kept_patch = (kept_patch & ~swap_out_mask) | swap_in_mask
            final_map = torch.cat([new_kept_patch, final_map[..., orig_kblk:]], dim=-1)

        # Opt-in diagnostic, and placed after every handle that can still move the mask
        # (`protect_own_frame` included) so it measures the objective the KERNEL was
        # actually handed, not the one some intermediate stage was aiming at.
        _record_realized_drift(getattr(self, "M", None), attn_full, v_bar_full, final_map, n_cand)

        x_patch, sparsity = run_block_sparse_kernel(q_patch, key, value, final_map)

        if norm_correction is not None:
            # Upsample the per-(nh, q_blk) correction to per-token (nh, Tq_patch) by repeating each
            # block's scalar across its KS_Q tokens -- exact inverse of `predict_attention`'s own
            # avg_pool1d(kernel_size=KS_Q, ceil_mode=True) pooling, so this lines up with the same
            # block boundaries `attn_pooled`/`final_map` were computed on (last block may be partial,
            # hence the truncation to Tq_patch rather than assuming an exact multiple of KS_Q).
            Tq_patch = x_patch.shape[-2]
            token_correction = torch.repeat_interleave(norm_correction, KS_Q, dim=-1)[..., :Tq_patch]
            x_patch = x_patch * token_correction.unsqueeze(-1).to(x_patch.dtype)

    if aux_store is not None:
        if aux_full:
            aux_store["value"] = AttentionAux(
                attn_pooled=attn_pooled,
                v_bar=v_bar_full()[..., :n_cand, :],
                attn_full=attn_full,
                v_bar_full=v_bar_full(),
                n_cand=n_cand,
                q_patch=q_patch,
                shape=shp,
                sparsity=sparsity,
            )
        else:
            aux_store["value"] = AttentionAux(shape=shp, sparsity=sparsity)

    x_out = combine_special_and_patch(x_special, x_patch, shp)
    if retain_output_grad:
        x_out.retain_grad()  # x_out is a non-leaf intermediate; probes.py needs its .grad after backward

    if aux_store is not None and aux_full:
        aux_store["value"].output = x_out  # (B, nh, N*P, hd), hook target for damage.py

    return project_out(self, x_out, B, N, P, nh, hd)
