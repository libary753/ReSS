"""Uniform head budget: the baseline `sparse_vggt` runs on, and the budget ablation arm
for the restore-swap family (`uniform_restore_swap*`).

There is deliberately no reallocation step here. Running an all-ones head-score vector
through `adjust_attention_selection` is NOT the no-op it looks like: that function sums
`baseline` ACROSS heads per query block and splits the total back evenly, which forcibly
equalizes every head's count within a query block (measured on a real forward: per-head
baselines like [243, 162, 238, ...] collapsed to [188, 188, 188, ...], std 32.3 -> 0.0)
and destroys the per-head variation `cdf_threshold` produces, since different heads' mass
concentrates differently and reaching the same cumulative probability naturally takes a
different number of key blocks per head.

That also matches SparseVGGT's own reference implementation, which reallocates only when
head scores are supplied -- its baseline configuration uses the raw cdf/topk-derived
`num_to_select` as-is.
"""

import torch

from ress.models.attention import compute_baseline_num_to_select


class UniformBudget:
    """`n_blocks`/`n_heads` size the (layer, head) score table. It is an all-ones no-op,
    but its first dimension is indexed by layer, so it has to match the backbone (VGGT 24
    sparsifiable layers, Pi3 18)."""

    def __init__(self, n_blocks=24, n_heads=16):
        self.head_scores = torch.ones(n_blocks, n_heads)

    @property
    def needs_v_bar(self):
        return False

    def num_to_select(self, layer_idx, attn_pooled, sparse_ratio=None, cdf_threshold=None,
                       beta=1.0):
        return compute_baseline_num_to_select(attn_pooled, sparse_ratio, cdf_threshold)
