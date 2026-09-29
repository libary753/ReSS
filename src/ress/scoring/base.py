"""Scoring key blocks, separately from deciding which to keep.

Attention probability is one possible score and is what the SparseVGGT baseline uses;
`scoring/projected.py` computes ReSS's residual-drift score instead. Keeping the
score behind an interface is what lets the two be swapped without touching selection --
see `selection/base.py` for the other half of that split.
"""

from abc import ABC, abstractmethod


class BlockScorer(ABC):
    @abstractmethod
    def score(self, attn_pooled, v_bar, ctx=None):
        """attn_pooled (P^h): (nh, q_blk, k_blk). v_bar (V̄): (nh, k_blk, hd).
        ctx: optional dict for scorer-specific extras (e.g. `scoring.projected`'s
        per-head M_h matrices).

        Returns per-(head, query-block, key-block) importance scores, same shape as
        attn_pooled: (nh, q_blk, k_blk). HIGHER = more important to KEEP (this is the
        sign convention used throughout: here, and in `selection.BlockSelector`,
        higher/True always means "keep", never "drop")."""
        raise NotImplementedError
