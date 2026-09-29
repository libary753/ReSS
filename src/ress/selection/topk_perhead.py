"""Per-head top-c selection: every head independently keeps its own highest-scoring
key blocks, however many its budget allows.

This is the selection rule all four modes use. It is the baseline in the sense that
nothing about it is novel -- what differs between the modes is the `scores` handed in
and the `num_to_select` that sizes the cut, not this sort-then-fill step.
"""

import torch

from ress.selection.base import BlockSelector


def build_block_mask_from_scores(scores, num_to_select, sink_blocks):
    """scores: (nh, q_blk, k_blk), higher = keep. num_to_select: (nh, q_blk) long
    tensor or int. Returns (nh, q_blk, k_blk + sink_blocks) bool, True = KEEP.

    Sort by score, then fill by rank until each (head, query-block) has spent its
    budget. Sink blocks are appended as always-kept and are not ranked."""
    if not torch.is_tensor(num_to_select):
        num_to_select = torch.full(scores.shape[:-1], num_to_select, device=scores.device, dtype=torch.long)
    order = torch.argsort(scores, dim=-1, descending=True)  # kept-first order
    rank = torch.argsort(order, dim=-1)
    keep = rank < num_to_select.unsqueeze(-1)  # True = keep

    if sink_blocks > 0:
        shape = list(keep.shape)
        shape[-1] = sink_blocks
        trailing_ones = torch.ones(shape, device=keep.device, dtype=torch.bool)
        keep = torch.cat([keep, trailing_ones], dim=-1)
    return keep


class TopKPerHeadSelector(BlockSelector):
    def select(self, scores, budget_spec, sink_blocks=0, ctx=None):
        if "perhead" not in budget_spec:
            raise ValueError(f"TopKPerHeadSelector requires budget_spec={{'perhead': ...}}, got {budget_spec.keys()}")
        return build_block_mask_from_scores(scores, budget_spec["perhead"], sink_blocks)
