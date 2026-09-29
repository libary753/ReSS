"""One Triton kernel for a whole restore-swap round's (..., nh, q_blk, k_blk)-sized
work.

`scoring.projected.restore_swap_rank_by`'s round body walks the big tensor about a
dozen times -- `raw_dot`, two quality tensors, two `masked_fill`s, two selections,
three boolean mask updates -- each a separate HBM round trip on 42 MB (bf16, at the
real 100-view shape nh=16, q_blk=810, k_blk=1619), so the round is bound by memory
traffic rather than by arithmetic.

This kernel keeps the row in registers and does all of it in one pass:

    raw_dot   = P * (term_k - term_q)
    restore_q = (-2*raw_dot - s_sq)  masked to -inf off the drop-set
    restore_count from (max restore_q > 0), the schedule count, and the row's own
                  drop/kept populations -- the same gating the eager path applies
    restore   = top-`restore_count` of restore_q
    compens_q = ( 2*raw_dot - s_sq)  masked to -inf off the kept-set
    compens   = top-`restore_count` of compens_q
    new_drop  = (drop & ~restore) | compens

so only the NEW drop mask is written; `raw_dot` and both quality tensors never exist
in memory.

BIT-EXACTNESS. `term_k` is NOT recomputed here -- it stays cuBLAS's einsum output, so
the GEMM's accumulation order is untouched. Every elementwise step then rounds to
`big_dtype` at exactly the point PyTorch does (it computes bf16 elementwise ops in
fp32 and rounds the result), and the selection is the same exact top-m as
`fused_topm`. The wired-in path is A/B verified against eager to produce identical
predictions; `RESS_FUSED_ROUND=0` restores the eager round body.

Restricted to the configuration the official recipe uses -- magnitude restore
criterion, drift compensate, geometric/fixed schedule, no `sequential_compensate`,
no `adaptive_restore_count`. `can_use()` gates on all of it; anything else keeps the
eager path.

`kept_mass_norm` (`_knorm`) is covered too, via the `KNORM` flag. It swaps
both quality expressions for `J = ||g||^2_M / K^2`'s exact single-move delta and needs
only two more PER-ROW scalars (`G`, `K`), so the memory traffic -- the thing this kernel
exists to cut -- is unchanged. That matters disproportionately: on the eager path the
same objective cost +5.14% forward overhead vs SparseVGGT against the recipe's +1.74%,
which was most of what the method gained.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

# Row-length bound: one program holds a whole row (padded to the next power of two) in
# registers, so k_blk -- which grows with view count, k_blk ~ 16.2 * n_views at VGGT's
# 1036 tokens/frame and KS_K=64 -- cannot grow without limit. Above this the dispatch
# falls back to the eager path (`can_use` returns False); it never fails.
#
# 4096 (~253 views) is a DELIBERATELY CONSERVATIVE setting, not a measured cliff. The
# kernel was verified identical to eager and still faster well past it (this
# kernel, synthetic rows at nh=16, q=128):
#
#     k_blk   ~views   eager      kernel     speedup
#     4096      253    1.224 ms   0.398 ms   3.07x
#     6476      400    2.634 ms   2.481 ms   1.06x   (pads to BLOCK=8192, 26% wasted)
#     8192      506    3.239 ms   1.707 ms   1.90x
#
# So raising MAX_K to 8192 is justified by measurement if runs go past ~250 views --
# left at 4096 since nothing here runs that wide.
# Beyond 8192 is unmeasured: the row stops fitting in registers, so expect spills or a
# compile failure, and the bound should stay.
MAX_K = 4096


if HAVE_TRITON:

    @triton.jit
    def _round_to_big(x, BF16: tl.constexpr):
        """Round an fp32 value to `big_dtype` and back, matching what PyTorch stores
        after each bf16 elementwise op (fp32 math, bf16 result).

        ONE return statement, deliberately: an early `return` inside the `if` compiled
        to a no-op rounding (the caller silently kept full fp32 precision), which is a
        silent-wrong-answer failure mode -- the kernel then diverged from eager by ~1 ulp
        per step, enough to flip rankings at the selection threshold on every real call.
        """
        y = x
        if BF16:
            y = x.to(tl.bfloat16).to(tl.float32)
        return y

    @triton.jit
    def _order_key(x, NBITS: tl.constexpr):
        """Order-preserving float -> unsigned key (see fused_topm for the derivation).
        Values that are exactly representable in bf16 keep all their information in the
        top 16 bits, so NBITS=16 halves the search."""
        b = x.to(tl.int32, bitcast=True)
        u = b.to(tl.uint32, bitcast=True)
        return tl.where(b >= 0, u | 0x80000000, ~u) >> (32 - NBITS)

    @triton.jit
    def _select_top_m(key, valid, m, NBITS: tl.constexpr):
        """Exact top-`m` mask of `key` within `valid`, ties broken by lowest index."""
        thr = tl.zeros([], dtype=tl.uint32)
        for i in tl.static_range(NBITS):
            cand = thr | (tl.full([], 1, tl.uint32) << (NBITS - 1 - i))
            c = tl.sum(tl.where((key >= cand) & valid, 1, 0).to(tl.int32))
            thr = tl.where(c >= m, cand, thr)
        gt = (key > thr) & valid
        n_gt = tl.sum(gt.to(tl.int32))
        eq = (key == thr) & valid
        rank_eq = tl.cumsum(eq.to(tl.int32), axis=0) - eq.to(tl.int32)
        return gt | (eq & (rank_eq < (m - n_gt)))

    @triton.jit
    def _knorm_quality(G, Kk, delta_G, u, K_EPS):
        """`scoring.projected.kept_mass_norm_quality`, operation-for-operation.

        `J - J_new = [G*u*(2K+u)/K^2 - delta_G] / (K+u)^2`, and the grouping is
        load-bearing (see that function): the ungrouped difference cancels away its own
        significant digits. Association order matches the eager expression exactly --
        `((G*u)*(2K+u))/(K*K)`, then subtract, then divide -- so this stays bit-exact."""
        K_new = tl.maximum(Kk + u, K_EPS)
        return (G * u * (2.0 * Kk + u) / (Kk * Kk) - delta_G) / (K_new * K_new)

    @triton.jit
    def _round_kernel(P, TERM_K, S_SQ, TERM_Q, DROP, PER_ITER, G_NORM, KEPT_MASS,
                      OUT_DROP, OUT_CNT, K, K_EPS,
                      BLOCK: tl.constexpr, NBITS: tl.constexpr, RAW_BF16: tl.constexpr,
                      Q_BF16: tl.constexpr, KNORM: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        valid = offs < K
        base = row * K + offs

        p = tl.load(P + base, mask=valid, other=0.0).to(tl.float32)
        tk = tl.load(TERM_K + base, mask=valid, other=0.0).to(tl.float32)
        ssq = tl.load(S_SQ + base, mask=valid, other=0.0).to(tl.float32)
        drop = tl.load(DROP + base, mask=valid, other=0) != 0
        # Two extra per-ROW scalars under `kept_mass_norm` -- the whole reason this
        # objective fuses at no bandwidth cost. `G = ||g||^2_M` and `K` (kept probability
        # mass) are (..., q_blk), NOT (..., q_blk, k_blk), so they ride in as broadcast
        # registers and the kernel still makes exactly ONE pass over the row.
        g_norm = tl.load(G_NORM + row).to(tl.float32)
        kept_mass = tl.load(KEPT_MASS + row).to(tl.float32)
        tq = _round_to_big(tl.load(TERM_Q + row).to(tl.float32), RAW_BF16)

        # raw_dot = P * (term_k - term_q), rounding where torch rounds. The two dtype
        # flags are SEPARATE on purpose: `raw_dot` lands in `big_dtype` (P and term_k are
        # both big), but `s_sq` comes from `projected_score`, which is NOT necessarily
        # big_dtype -- with the official bf16 recipe it is fp32, so `-2*raw - s_sq`
        # PROMOTES and the quality tensors (and hence the selection's key width) are fp32,
        # not bf16. Assuming otherwise silently rounded the quality an extra time and
        # searched 16 bits of a 32-bit key, which flipped rankings near the threshold on
        # every real call while still looking correct in isolation.
        raw = _round_to_big(p * _round_to_big(tk - tq, RAW_BF16), RAW_BF16)

        # restore_quality, then -inf off the drop-set. Two objectives:
        #   plain  : -Delta_G                       (minimize ||g||^2_M)
        #   knorm  : J - J_new, with u = +P_k       (minimize ||g||^2_M / K^2)
        # Under KNORM the eager path does this whole chain in fp32 regardless of
        # `big_dtype` (`projected._knorm_dtype`), so there is no `_round_to_big` here --
        # `raw` alone is still rounded to big_dtype above, exactly as eager's
        # `raw_dot.to(dtype)` upcast of a bf16 tensor does.
        if KNORM:
            rq = _knorm_quality(g_norm, kept_mass, 2.0 * raw + ssq, p, K_EPS)
        else:
            rq = _round_to_big(_round_to_big(-2.0 * raw, RAW_BF16) - ssq, Q_BF16)
        rq = tl.where(drop & valid, rq, float("-inf"))

        # Same count gating as the eager path: the >=1 floor applies only where some
        # candidate is actually beneficial (restore_quality > 0), and the count can
        # never exceed either population it has to move between.
        beneficial = tl.max(tl.where(valid, rq, float("-inf"))) > 0.0
        n_drop = tl.sum((drop & valid).to(tl.int32))
        n_kept = tl.sum(((drop == 0) & valid).to(tl.int32))
        pic = tl.load(PER_ITER + row).to(tl.int32)
        pic = tl.where(beneficial, tl.maximum(pic, 1), pic)
        cnt = tl.minimum(tl.minimum(pic, n_drop), n_kept)
        cnt = tl.where(beneficial, cnt, 0)

        restore = _select_top_m(_order_key(rq, NBITS), drop & valid, cnt, NBITS)

        # compensate_quality, then -inf off the kept-set. `ssq - 2*raw` in that order
        # under KNORM, matching eager's `s_sq - 2.0*compensate_raw_dot` -- the operand
        # order is a rounding decision, not cosmetic.
        if KNORM:
            cq = _knorm_quality(g_norm, kept_mass, ssq - 2.0 * raw, -p, K_EPS)
        else:
            cq = _round_to_big(_round_to_big(2.0 * raw, RAW_BF16) - ssq, Q_BF16)
        kept = (drop == 0) & valid
        cq = tl.where(kept, cq, float("-inf"))
        compensate = _select_top_m(_order_key(cq, NBITS), kept, cnt, NBITS)

        new_drop = ((drop & (restore == 0)) | compensate) & valid
        tl.store(OUT_DROP + base, new_drop.to(tl.int8), mask=valid)
        tl.store(OUT_CNT + row, cnt)


_SUPPORTED = (torch.float32, torch.bfloat16)


def _step_dtypes(P, term_k, s_sq, big_dtype, knorm_dtype=None):
    """The dtypes torch's own promotion produces for the round's two intermediates:
    `raw_dot` (from P, term_k and the big_dtype-cast term_q) and the quality tensors
    (`+-2*raw_dot - s_sq`, which promotes against whatever `projected_score` returned).
    The kernel has to round exactly where these say, and size its selection key from
    the quality dtype.

    `knorm_dtype` (not None => `kept_mass_norm`): the eager path casts every operand of
    the normalized quality to that one dtype up front (`projected._knorm_dtype`, fp32 by
    default), so the quality dtype is simply it -- `s_sq`'s own width stops mattering.
    `raw_dot` is still produced in `big_dtype` first and then upcast, so `raw_dtype` is
    unchanged."""
    raw_dtype = torch.promote_types(torch.promote_types(P.dtype, term_k.dtype), big_dtype)
    q_dtype = knorm_dtype if knorm_dtype is not None else torch.promote_types(raw_dtype, s_sq.dtype)
    return raw_dtype, q_dtype


def can_use(P, s_sq, term_k, drop_mask, big_dtype, knorm_dtype=None):
    """Whether `fused_round_step` can serve this call (else keep the eager body)."""
    if not HAVE_TRITON or not P.is_cuda:
        return False
    if big_dtype not in _SUPPORTED:
        return False
    k_blk = P.shape[-1]
    if not (0 < k_blk <= MAX_K):
        return False
    if not (s_sq.shape == P.shape and term_k.shape == P.shape and drop_mask.shape == P.shape):
        return False
    return all(dt in _SUPPORTED
               for dt in _step_dtypes(P, term_k, s_sq, big_dtype, knorm_dtype))


def fused_round_step(P, term_k, s_sq, term_q, drop_mask, per_iter_count, big_dtype,
                      g_norm_sq=None, kept_mass=None, knorm_dtype=None, k_eps=1e-6):
    """One round's big-tensor work. Returns (new_drop_mask, restore_count).

    P/term_k/s_sq: (..., k_blk) float. term_q: (...) same leading shape.
    drop_mask: (..., k_blk) bool. per_iter_count: (...) integer, this round's
    schedule count BEFORE the beneficial-candidate gating (applied in-kernel).

    `g_norm_sq`/`kept_mass`/`knorm_dtype` together switch the objective from `||g||^2_M`
    to `J = ||g||^2_M / K^2` (`scoring.projected.restore_swap_rank_by(kept_mass_norm=True)`).
    Both are (...) per-row scalars, so they cost two scalar loads and change nothing about
    the kernel's memory traffic -- which is the whole point of fusing this objective
    rather than leaving it on the eager path.
    """
    knorm = knorm_dtype is not None
    assert knorm == (g_norm_sq is not None) == (kept_mass is not None), (
        "fused_round_step: g_norm_sq, kept_mass and knorm_dtype must be given together")
    k_blk = P.shape[-1]
    lead = P.shape[:-1]
    p = P.contiguous()
    tk = term_k.contiguous()
    ss = s_sq.contiguous()
    tq = term_q.contiguous().to(torch.float32)
    dm = drop_mask.contiguous()
    pic = per_iter_count.contiguous().to(torch.int32)
    raw_dtype, q_dtype = _step_dtypes(P, term_k, s_sq, big_dtype, knorm_dtype)
    zero = torch.zeros(lead, dtype=torch.float32, device=p.device)
    gn = g_norm_sq.contiguous().to(torch.float32) if knorm else zero
    km = kept_mass.contiguous().to(torch.float32) if knorm else zero

    n_rows = p.numel() // k_blk
    out_drop = torch.empty(p.shape, dtype=torch.uint8, device=p.device)
    out_cnt = torch.empty(lead, dtype=torch.int32, device=p.device)
    _round_kernel[(n_rows,)](
        p.view(-1), tk.view(-1), ss.view(-1), tq.view(-1), dm.view(-1).view(torch.uint8),
        pic.view(-1), gn.view(-1), km.view(-1),
        out_drop.view(-1), out_cnt.view(-1), k_blk, float(k_eps),
        BLOCK=triton.next_power_of_2(k_blk),
        NBITS=16 if q_dtype is torch.bfloat16 else 32,
        RAW_BF16=raw_dtype is torch.bfloat16, Q_BF16=q_dtype is torch.bfloat16,
        KNORM=knorm,
        num_warps=4,
    )
    return out_drop.view(torch.bool), out_cnt.to(torch.int64)
