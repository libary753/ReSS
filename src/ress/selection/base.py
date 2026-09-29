"""How many blocks to keep is one decision; which ones is another.

`BlockScorer` (`scoring/base.py`) says how important each key block is, a budget object
(`budget/`) says how many a head may keep, and `BlockSelector` here turns the two into a
mask. Earlier sparse-attention code folds all three into one function, which makes them
impossible to vary independently -- and varying them independently is exactly what
separates HeSS (budget changes) from ReSS (score changes) from the SparseVGGT baseline
(neither). Split this way, each combination is configuration rather than a new code path.
"""

from abc import ABC, abstractmethod


class BlockSelector(ABC):
    @abstractmethod
    def select(self, scores, budget_spec, sink_blocks=0, ctx=None):
        """scores: (nh, q_blk, k_blk), higher = more important to KEEP (see
        `scoring.base.BlockScorer`).

        budget_spec: exactly one of
          - {"perhead": c_h} -- c_h is an int or (nh, q_blk) tensor, independent
            per-(head, query-block) budget -- what every mode here uses.
          - {"joint_per_query": C, "floor_per_head": f} -- a single budget C shared
            jointly across all heads for each query-block, with
            each head guaranteed at least `f` kept blocks before the remaining
            `C - f*nh` compete globally.

        sink_blocks: number of always-kept trailing key-blocks (special/register
        tokens) to append -- these are NEVER counted against the budget.

        Returns: kept mask (nh, q_blk, k_blk + sink_blocks) bool, True = KEEP.
        This is exactly the `final_map` contract
        (`models.attention.run_block_sparse_kernel`'s only input besides q/k/v), so
        any BlockSelector's output can be fed straight to the sparse kernel with zero
        translation."""
        raise NotImplementedError
