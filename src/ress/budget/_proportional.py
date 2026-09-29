"""Reallocating a fixed total block budget across heads in proportion to their scores.

Each head's share is its score times the total, then capped at what it can actually use
(a head cannot keep more blocks than exist), and the surplus freed by capping is
redistributed over the uncapped heads -- iterated until nothing more is capped.

Note this is NOT water-filling. Water-filling equalises marginal damage; this equalises
score-weighted share and then repairs the overflow. Both are reasonable, they are not the
same allocation, and HeSS is defined by this one.

Private (leading underscore): `budget/hess.py` is the entry point callers use.
"""

import torch


# Moved here from the deleted `selection.beta_budget` (clean-code pass): this
# is the only surviving caller.
def _largest_remainder_round(m_float, target, fixed=None, pinned_value=None):
    """m_float: (nh, q_blk) >= 0, target column sum `target` (int or (q_blk,)
    tensor) COUNTING ONLY the non-fixed rows. Returns (nh, q_blk) long, columns
    summing to exactly `target` (+ any `pinned_value` at fixed rows) --
    largest-remainder apportionment.

    `fixed`: (nh, q_blk) bool, rows to exclude from the deficit bump entirely (their
    remainder priority is forced below every real [0,1) remainder, so a deficit of
    at most `(~fixed).sum(dim=0)` -- always true here, `target` is exactly the sum
    over non-fixed rows -- can never reach into a fixed row). `pinned_value`: value
    to use for fixed rows in the output (defaults to 0, i.e. `fixed` unused/no rows
    pinned -- the initial, no-clamping-yet call)."""
    if not torch.is_tensor(target):
        target = torch.full(m_float.shape[1:], target, device=m_float.device, dtype=torch.long)
    m_floor = torch.floor(m_float).clamp(min=0)
    m_floor_int = m_floor.long()
    deficit = (target - m_floor_int.sum(dim=0)).clamp(min=0)  # (q_blk,)
    remainder = m_float - m_floor
    if fixed is not None:
        remainder = torch.where(fixed, torch.full_like(remainder, -1.0), remainder)
    order = torch.argsort(remainder, dim=0, descending=True)
    rank = torch.argsort(order, dim=0)
    bump = rank < deficit.unsqueeze(0)
    result = m_floor_int + bump.long()
    if fixed is not None and pinned_value is not None:
        result = torch.where(fixed, pinned_value, result)
    return result



def adjust_attention_selection(num_to_select, head_scores, beta, max_val, max_iters=1000, floor_frac=None):
    """num_to_select: (..., nh, q_blk) float/long, baseline per-head budget.
    head_scores: (nh,) importance score. beta: interpolation strength (0=no
    change, 1=full proportional reallocation -- HeSS's default). max_val: cap.
    Returns: (..., nh, q_blk) long, capped + surplus-redistributed budget.

    floor_frac: None (default) reproduces HeSS exactly -- pure proportional
    share of `total` by `head_scores`, no per-head lower bound (a head with a low
    score can be driven toward 0). When set, it isolates
    whether HeSS's per-head-allocation handle benefits from a floor independent of
    which block-cutoff order it's paired with, mirroring
    `selection.topk_joint.floor_from_frac`'s convention/formula), every head is
    first guaranteed `floor_frac` of its uniform per-column share
    `c_uniform = total // nh` BEFORE the remaining budget is split proportionally
    by `head_scores` -- i.e. `ideal = floor + (total - nh*floor) * scores_norm`
    instead of `ideal = total * scores_norm`. Still beta-blended with
    `num_to_select` exactly as before, then fed through the same cap-and-
    redistribute-surplus loop below (which only ever MOVES budget from a
    capped head to an uncapped one, never removes budget from a head below
    max_val, so a floor established here survives that loop intact).

    Guards against a real infinite-loop found empirically
    (phase26_c1_reexamination.py hung indefinitely): if `head_scores` sums to ~0
    for some layer (e.g. a layer whose gradient w.r.t. the probe objective is
    exactly zero -- found for VGGT's last global block specifically, whose direct
    attention-sublayer output has ~no marginal effect on `pose_enc`, since
    `camera_head` only reads the final aggregated-token snapshot; any offline
    score built from that gradient, like `moa_offline_calibrate`, inherits an
    all-zero row for that layer), `scores_norm = head_scores / head_scores.sum()`
    is 0/0 = NaN, which then propagates through `ideal`/`adjusted`, and the while
    loop's `excess.sum() <= 0` check is FALSE for NaN forever (NaN comparisons are
    always false) -- so it never terminates. Falls back to uniform reallocation
    (beta effectively 0 for that call) when head_scores is degenerate, and caps
    the loop at `max_iters` as defense-in-depth against any other NaN source.
    """
    score_sum = head_scores.sum()
    if not torch.isfinite(score_sum) or score_sum <= 0:
        scores_norm = torch.full_like(head_scores, 1.0 / head_scores.numel())
    else:
        scores_norm = head_scores / score_sum
    scores_reshaped = scores_norm.reshape(*([1] * (num_to_select.dim() - 2)), -1, 1)

    total = num_to_select.sum(dim=-2, keepdim=True)
    if floor_frac is None:
        ideal = scores_reshaped * total
    else:
        nh = head_scores.numel()
        c_uniform = torch.div(total, nh, rounding_mode="floor")
        floor_per_head = floor_frac * c_uniform  # (..., 1, q_blk), same value broadcast to every head
        remaining = (total - nh * floor_per_head).clamp(min=0)
        ideal = floor_per_head + remaining * scores_reshaped
    adjusted = (1 - beta) * num_to_select + beta * ideal

    for _ in range(max_iters):
        excess = (adjusted - max_val).clamp(min=0)
        if not torch.isfinite(excess).all() or excess.sum() <= 0:
            break
        total_excess = excess.sum(dim=-2, keepdim=True)
        adjusted = adjusted.clamp(max=max_val)
        eligible = adjusted < max_val
        weights = adjusted * eligible
        weights_norm = weights / (weights.sum(dim=-2, keepdim=True) + 1e-9)
        adjusted = adjusted + weights_norm * total_excess
    else:
        raise RuntimeError(f"adjust_attention_selection did not converge in {max_iters} iterations")

    return adjusted.clamp(max=max_val).floor().long()


def adjust_query_block_selection(num_to_select, row_scores, beta, max_val, floor_frac=None):
    """Query-block-axis mirror of `adjust_attention_selection`: conserves
    `num_to_select.sum(dim=-1)` (per head, across query-blocks) instead of
    `num_to_select.sum(dim=-2)` (across heads). This is the damage-based
    replacement for `cdf_threshold`'s per-row budget adaptivity (see
    ): unlike CDF, which only ever RAISES a row's
    budget (uncontrolled sparsity inflation), this redistributes a FIXED
    total by `row_scores` (e.g. `scoring.damage.row_damage_at_budget`'s
    gradient-free, direction-aware hardness), so harder rows get more of the
    SAME total instead of free extra budget.

    num_to_select, row_scores: (..., nh, q_blk) -- unlike `head_scores` (a
    fixed (nh,) vector), row hardness genuinely varies per query-block, so
    `row_scores` has the SAME shape as `num_to_select`, no reshape/broadcast
    needed. beta: 0 = num_to_select unchanged EXACTLY (same convention as
    `adjust_attention_selection`); 1 = full conserving proportional-by-
    hardness reallocation. max_val: cap per row (k_blk). floor_frac: same
    convention as `adjust_attention_selection`'s own (guarantee floor_frac *
    uniform per-row share before splitting the rest by row_scores).

    Degenerate-score guard: unlike `adjust_attention_selection` (one (nh,)
    vector, one global degeneracy check), here EACH HEAD's own row_scores can
    independently sum to ~0 (e.g. a head whose baseline budget is already 0
    everywhere) -- that head falls back to a uniform reallocation for its
    OWN rows only, other heads unaffected (the degeneracy check broadcasts
    per (..., nh) slice, not globally).

    Unlike `adjust_attention_selection`'s plain `.floor().long()` (which does
    NOT guarantee exact conservation -- floor of a proportional split can
    under-count the target sum), this rounds via largest-remainder
    apportionment (`selection.beta_budget._largest_remainder_round`, already
    used/tested there) with an iterative pin-at-cap + redistribute-surplus
    loop mirroring `selection.beta_budget._allocate_budget`'s proven
    cap/redistribute mechanism (rather than `adjust_attention_selection`'s
    own inexact float loop + floor) -- so `result.sum(dim=-1) ==
    num_to_select.sum(dim=-1)` holds EXACTLY, not just approximately, while
    still respecting `result <= max_val` everywhere."""
    q_blk = num_to_select.shape[-1]

    score_sum = row_scores.sum(dim=-1, keepdim=True)  # (..., nh, 1)
    degenerate = ~torch.isfinite(score_sum) | (score_sum <= 0)
    uniform_scores = torch.full_like(row_scores, 1.0 / q_blk)
    scores_norm = torch.where(degenerate, uniform_scores, row_scores / score_sum.clamp(min=1e-30))

    total = num_to_select.sum(dim=-1, keepdim=True)  # (..., nh, 1)
    if floor_frac is None:
        ideal = scores_norm * total
    else:
        c_uniform = torch.div(total, q_blk, rounding_mode="floor")
        floor_per_row = floor_frac * c_uniform
        remaining = (total - q_blk * floor_per_row).clamp(min=0)
        ideal = floor_per_row + remaining * scores_norm

    adjusted_float = (1 - beta) * num_to_select.to(ideal.dtype) + beta * ideal

    # `_largest_remainder_round`'s convention: axis 0 = items to round/conserve,
    # remaining axes = independent columns -- move the query-block axis (the
    # item axis here, unlike `_allocate_budget`'s head axis) to position 0.
    m_float = adjusted_float.movedim(-1, 0)  # (q_blk, ..., nh)
    target_total = total.squeeze(-1)  # (..., nh), already matches m_float.shape[1:]

    fixed = torch.zeros_like(m_float, dtype=torch.bool)
    m_int = _largest_remainder_round(m_float, target_total, fixed)

    for _ in range(q_blk):
        newly_over = (m_int > max_val) & ~fixed
        if not newly_over.any():
            break
        fixed = fixed | newly_over
        m_int = torch.where(fixed, torch.full_like(m_int, max_val), m_int)
        free = ~fixed
        if not free.any():
            break
        pinned_sum = torch.where(fixed, torch.full_like(m_int, max_val), torch.zeros_like(m_int)).sum(dim=0)
        remaining_total = target_total - pinned_sum
        free_weight = torch.where(free, m_float, torch.zeros_like(m_float))
        free_sum = free_weight.sum(dim=0, keepdim=True)
        n_free = free.sum(dim=0, keepdim=True).clamp(min=1).to(m_float.dtype)
        uniform_free = free.to(m_float.dtype) / n_free
        norm_weight = torch.where(free_sum > 1e-30, free_weight / free_sum.clamp(min=1e-30), uniform_free)
        free_float = torch.where(free, remaining_total.unsqueeze(0).to(m_float.dtype) * norm_weight,
                                  torch.zeros_like(m_float))
        m_int = _largest_remainder_round(free_float, remaining_total, fixed, pinned_value=m_int)
    else:
        raise RuntimeError(f"adjust_query_block_selection did not converge within {q_blk} pin rounds")

    assert (m_int.sum(dim=0) == target_total).all(), "adjust_query_block_selection: budget not conserved"
    assert (m_int <= max_val).all(), "adjust_query_block_selection: cap violated"
    return m_int.movedim(0, -1).long()


def adjust_joint_selection(num_to_select, cell_scores, beta, max_val, floor_frac=None):
    """Joint (head, query-block) reallocation: conserves ONLY the grand total
    `num_to_select.sum(dim=(-2, -1))` (the whole layer's budget), not each
    individual per-head or per-query-block sub-total -- the composition the
    user chose after `adjust_attention_selection` (mechanism A, conserves
    per-query-block column totals) and `adjust_query_block_selection`
    (mechanism B, conserves per-head row totals) turned out to conserve
    DIFFERENT margins of the same (nh, q_blk) matrix, so naively running one
    after the other silently breaks whichever guarantee ran first (a
    matrix-margins conflict -- see docs discussion).

    `cell_scores` (typically `scoring.damage.row_damage_at_budget`'s output)
    already varies over BOTH the head axis and the query-block axis, so a
    single flat proportional split by this score is simultaneously
    head-aware and query-block-aware: heads whose cells are systematically
    harder automatically pull a larger share of the total in the split,
    while individual hard query-blocks within a head are still prioritized.
    No separate per-head importance vector (Fisher or otherwise) is needed,
    and no extra forward pass -- `cell_scores` is computed the same cheap way
    `adjust_query_block_selection` already does. This makes mechanism A
    unnecessary in this joint mode: it's superseded by a generalized
    (whole-grid) version of mechanism B's own exact-conservation machinery.

    num_to_select, cell_scores: (..., nh, q_blk). beta, max_val, floor_frac:
    same conventions as `adjust_query_block_selection`. Returns (..., nh,
    q_blk) long, with `result.sum(dim=(-2,-1)) ==
    num_to_select.sum(dim=(-2,-1))` exactly and `result <= max_val`
    everywhere.

    Implementation: flattens (nh, q_blk) into a single axis of size nh*q_blk
    and delegates entirely to `adjust_query_block_selection` (unchanged) --
    that function conserves `num_to_select.sum(dim=-1)` per "row"; with only
    one row left after flattening, it conserves the sum over the entire
    grid instead. `max_val` is a plain scalar cap (= k_blk), unaffected by
    the reshape since every cell shares the same cap regardless of axis."""
    nh, q_blk = num_to_select.shape[-2:]
    flat_shape = (*num_to_select.shape[:-2], 1, nh * q_blk)
    flat_counts = num_to_select.reshape(flat_shape)
    flat_scores = cell_scores.reshape(flat_shape)
    flat_out = adjust_query_block_selection(flat_counts, flat_scores, beta, max_val, floor_frac=floor_frac)
    return flat_out.reshape(*num_to_select.shape[:-2], nh, q_blk)


def adjust_head_selection(num_to_select, cell_scores, beta, max_val, floor_frac=None):
    """Head-axis reallocation using scores that vary per (head, query-block)
    cell, conserving EACH QUERY-BLOCK'S OWN column total (sum over heads) --
    the per-query-block-localized generalization of
    `adjust_attention_selection` (mechanism A), which only accepts a single
    (nh,) head_scores vector broadcast identically to every query-block
    column. Here the budget is fixed per query-block
    first, and each query-block independently decides its own head split
    from its own `cell_scores` row -- unlike mechanism A's one
    layer-wide head-importance vector.

    num_to_select, cell_scores: (..., nh, q_blk). beta, max_val, floor_frac:
    same conventions as `adjust_query_block_selection`. Returns (..., nh,
    q_blk) long, with `result.sum(dim=-2) == num_to_select.sum(dim=-2)`
    exactly (each query-block's column total conserved) and `result <=
    max_val` everywhere.

    Implementation: swaps the last two axes (nh <-> q_blk) and delegates to
    `adjust_query_block_selection` (unchanged) -- that function conserves
    `num_to_select.sum(dim=-1)` per "row"; after the transpose, its "row" is
    the ORIGINAL query-block and its "columns" are the ORIGINAL heads, so it
    conserves per-query-block column totals and reallocates across heads
    instead. `max_val` is a plain scalar cap (= k_blk), unaffected by the
    transpose since every cell shares the same cap regardless of axis."""
    transposed_counts = num_to_select.transpose(-2, -1)
    transposed_scores = cell_scores.transpose(-2, -1)
    transposed_out = adjust_query_block_selection(transposed_counts, transposed_scores, beta, max_val,
                                                    floor_frac=floor_frac)
    return transposed_out.transpose(-2, -1)
