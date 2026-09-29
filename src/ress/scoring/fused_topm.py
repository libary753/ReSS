"""Triton per-row exact top-`m` MASK, a drop-in replacement for
`scoring.projected._top_m_mask`.

Why this exists: at the real 100-view shapes ((nh, q_blk, k_blk) = (16, 810, 1619))
the two `_top_m_mask` calls per restore-swap round measured 2.33 ms of a 6.24 ms
round -- 37%, the single largest item -- while one streaming read of the same
(nh, q_blk, k_blk) tensor costs 0.071 ms. `torch.topk` is ~12x off the bandwidth
floor here because it fully materializes the top-`k_max` VALUES and INDICES (k_max
is the round's analytic upper bound, far larger than most rows' own `counts`) and
then throws almost all of them away in the `pos < counts` step.

This kernel never materializes indices. Per row it finds the exact `counts[row]`-th
largest value by binary search over the order-preserving float->uint bit mapping
(SRAM-resident row, no extra HBM traffic), then writes the mask in one pass:
    mask = (key > thr) | (key == thr & rank_among_ties < remaining)
so exactly `counts[row]` entries are True, ties broken by lowest index.

`torch.topk`'s tie order among exactly-equal values is unspecified, and
`_top_m_mask`'s own docstring already declares tie ORDER a don't-care ("both pick an
arbitrary winner among ties"). That matters more here than it looks: with
`big_dtype=torch.bfloat16` the quality tensors carry 8 mantissa bits, so exact ties
at the selection threshold are common rather than measure-zero -- `verify_against_eager`
below reports the realized disagreement rate so it is a measured quantity, not an
assumption.

bf16/fp16 inputs search 16 bits instead of 32 (their fp32 bit patterns have a zero
low half by construction), which halves the search loop.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton ships with torch on CUDA builds
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _topm_mask_kernel(Q, CNT, OUT, K, BLOCK: tl.constexpr, NBITS: tl.constexpr,
                          LARGEST: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        valid = offs < K
        x = tl.load(Q + row.to(tl.int64) * K + offs, mask=valid, other=0.0).to(tl.float32)
        m = tl.load(CNT + row).to(tl.int32)

        # Order-preserving float32 -> uint32: positives get the sign bit set, negatives
        # are complemented, so unsigned compare reproduces float compare exactly
        # (-inf included; NaN is not produced by the callers).
        b = x.to(tl.int32, bitcast=True)
        u = b.to(tl.uint32, bitcast=True)
        key = tl.where(b >= 0, u | 0x80000000, ~u)
        if not LARGEST:
            key = ~key  # complement reverses the order -> "top m" becomes "bottom m"
        # A 16-bit input's fp32 bits have a zero low half, but the negative branch's
        # complement turns that half into all-ones -- so the low half cannot simply be
        # ignored, it has to be SHIFTED OUT. The top NBITS still separate every distinct
        # input value (the mapping is a bijection on the 16-bit patterns), so this stays
        # exactly order-preserving while halving the search loop.
        key = key >> (32 - NBITS)
        key = tl.where(valid, key, 0)

        # Largest `thr` with popcount(key >= thr) >= m, built MSB-first.
        thr = tl.zeros([], dtype=tl.uint32)
        for i in tl.static_range(NBITS):
            cand = thr | (tl.full([], 1, tl.uint32) << (NBITS - 1 - i))
            c = tl.sum(tl.where((key >= cand) & valid, 1, 0).to(tl.int32))
            thr = tl.where(c >= m, cand, thr)

        gt = (key > thr) & valid
        n_gt = tl.sum(gt.to(tl.int32))
        eq = (key == thr) & valid
        # Exclusive prefix count among the tied entries -> take the first `m - n_gt`.
        rank_eq = tl.cumsum(eq.to(tl.int32), axis=0) - eq.to(tl.int32)
        sel = gt | (eq & (rank_eq < (m - n_gt)))
        tl.store(OUT + row.to(tl.int64) * K + offs, sel.to(tl.int8), mask=valid)


# One program holds a whole row (padded to the next power of two) in registers, so the
# row length is bounded: k_blk scales with view count (1619 at 100 views ~ 16.2 *
# n_views), and past this register pressure would spill. Callers dispatch back to the
# eager path above this -- it never fails.
#
# Same conservative 4096 (~253 views) as `fused_round.MAX_K`, and the same note applies:
# raising it to 8192 (~500 views) is measurement-backed THERE (see that module's table)
# and this kernel has the identical one-row-per-program structure, but was not itself
# benchmarked past k_blk=1619. Left at 4096; if runs go
# wider, raise both and re-measure this one too.
MAX_K = 4096
_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def can_use(quality):
    """Whether `top_m_mask` can serve this tensor (else the caller keeps the eager path)."""
    return (HAVE_TRITON and quality.is_cuda and quality.dtype in _SUPPORTED_DTYPES
            and 0 < quality.shape[-1] <= MAX_K)


def _num_bits(dtype):
    # bf16/fp16 widen to fp32 bit patterns whose low 16 bits are zero, so the search
    # only has to resolve the top half.
    return 16 if dtype in (torch.bfloat16, torch.float16) else 32


def top_m_mask(quality, counts, largest=True):
    """Boolean mask of each row's top-`counts[row]` entries of `quality` (last dim).

    Same contract as `scoring.projected._top_m_mask` minus its `k_max` argument
    (this kernel reads each row's own count directly, so no analytic upper bound is
    needed). quality: (..., k_blk) float on CUDA. counts: (...) integer.
    """
    assert quality.is_cuda, "fused top_m_mask is CUDA-only"
    k_blk = quality.shape[-1]
    q = quality.contiguous()
    counts = counts.clamp(min=0, max=k_blk).contiguous().to(torch.int32)
    n_rows = q.numel() // k_blk
    assert counts.numel() == n_rows, f"counts {counts.numel()} != rows {n_rows}"

    out = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
    if n_rows == 0:
        return out.view(torch.bool)
    block = triton.next_power_of_2(k_blk)
    # num_warps=4 measured best at the real k_blk=1619 shape (0.344 ms vs 0.397 at 8,
    # 0.440 at 2, 0.722 at 16): the search loop is NBITS block-reductions, so more warps
    # buy per-lane parallelism but pay more cross-warp reduction steps. A 256-bin
    # histogram search was also tried and is slower (0.509 ms) -- `tl.histogram`'s
    # shared-memory atomics cost more than the extra reduction passes they remove.
    _topm_mask_kernel[(n_rows,)](
        q.view(-1), counts.view(-1), out.view(-1), k_blk,
        BLOCK=block, NBITS=_num_bits(q.dtype), LARGEST=largest, num_warps=4,
    )
    return out.view(torch.bool)
