"""HeSS: a per-head block budget driven by calibrated head importance.

Importance is the Fisher trace of the reconstruction loss with respect to each head's
W_Q, accumulated offline over a CO3Dv2 calibration set. Heads the loss is more sensitive
to get a larger share of the same total block budget; the shipped tables in
`ress/calibration/` are that accumulation, and `scripts/calibrate_fisher_scores.py`
regenerates them.

Two implementation notes worth keeping:

- The Fisher trace is computed as ||grad||^2 rather than as trace(grad @ grad.T). The two
  are equal, but the outer product costs O(d^2) to build and then throws all of it away
  except the diagonal.
- HeSS Eq. (9) mixes a camera-loss and a point-cloud-loss table. `combine_lambda`
  implements it for continuous lambda in [0, 1], not just the three discrete settings the
  method is usually described with -- which matters here because the three backbones want
  three different values (see `ress/backbones.py`).
"""

import torch

from ress.budget._proportional import adjust_attention_selection
from ress.models.attention import compute_baseline_num_to_select
from ress.models.wiring import global_blocks


def accumulate_fisher_trace_scores(model, head_attribution_scores):
    """head_attribution_scores: (3, n_blocks, n_heads) float tensor, accumulated
    in-place (call after `loss.backward()`, matching HeSS's per-scene calibration
    loop -- only needed to regenerate calibration scores from scratch, not for
    everyday use of an already-shipped `head_attribution_scores.pt`)."""
    with torch.no_grad():
        for block_idx, block in enumerate(global_blocks(model)):
            grad = block.attn.qkv.weight.grad
            block.attn.qkv.weight.grad = None
            dim = grad.shape[-1]
            grad = torch.unbind(
                grad.reshape(3, block.attn.num_heads, dim // block.attn.num_heads, dim), dim=0
            )
            for qkv_idx in range(3):
                grad_qkv = grad[qkv_idx].reshape(grad[qkv_idx].shape[0], -1)
                score = (grad_qkv**2).sum(dim=-1)  # ||g||^2, == trace(g g^T)
                head_attribution_scores[qkv_idx, block_idx] += score.cpu()


def sum_normalize(scores):
    """Per-layer sum=1 normalization (Eq(7)-(8), score_scale_type="sum")."""
    return scores / scores.sum(dim=-1, keepdim=True)


def combine_lambda(score_cam, score_pc, lam):
    """HeSS(h) = λ·HeSS_cam(h) + (1-λ)·HeSS_pc(h) (Eq(9)), for continuous λ -- the method
    is usually described with only λ∈{0, 0.5, 1}. Inputs already sum-normalized
    per layer; renormalizes the combination too."""
    combined = lam * score_cam + (1 - lam) * score_pc
    return sum_normalize(combined)


class HessBudget:
    """Offline (calibration-set) per-head importance -> per-(head, query-block)
    budget, following HeSS's runtime recipe:
    baseline num_to_select from THIS input's own P^h (content-dependent cdf/sparse
    ratio cutoff), then proportionally reallocated by the calibrated head scores.
    """

    def __init__(self, score_cam_path, score_pc_path, lam=0.5, qkv_idx=0, floor_frac=None,
                  layer_slice=None):
        raw_cam = torch.load(score_cam_path, weights_only=False)[qkv_idx]
        raw_pc = torch.load(score_pc_path, weights_only=False)[qkv_idx]
        if layer_slice is not None:
            # A calibration table's layer axis indexes the BACKBONE's own block numbering,
            # which is not always the same as ReSS's "sparsifiable blocks, in order"
            # (`models.wiring.global_blocks`). VGGT's tables are already the 24 global
            # blocks (its aggregator keeps them in a separate ModuleList), but Pi3's
            # (an earlier Pi3 calibration accumulated over
            # `model.decoder` wholesale) covers all 36 decoder blocks, of which only the
            # odd ones do cross-frame attention -- hence `slice(1, None, 2)` there. Applied
            # before normalization, which is per-layer (`sum_normalize`, dim=-1) and so
            # unaffected by which layers are present.
            raw_cam, raw_pc = raw_cam[layer_slice], raw_pc[layer_slice]
        self.score_cam = sum_normalize(raw_cam)  # (n_blocks, n_heads)
        self.score_pc = sum_normalize(raw_pc)
        self.lam = lam
        self.head_scores = combine_lambda(self.score_cam, self.score_pc, self.lam)  # (n_blocks, n_heads)
        # None (default) reproduces HeSS exactly (no per-head lower bound, see
        # `budget._proportional.adjust_attention_selection`'s `floor_frac` docstring).
        # Set (e.g. 0.7, matching `runtime.beta_budget.set_beta_budget_v2`'s "ours"
        # config) to test whether HeSS's per-head allocation benefits from a floor
        # independent of block-cutoff order.
        self.floor_frac = floor_frac

    def num_to_select(self, layer_idx, attn_pooled, sparse_ratio=None, cdf_threshold=None, beta=1.0):
        """attn_pooled (P^h): (nh, q_blk, k_blk), this layer's just-computed pooled
        map. Returns (nh, q_blk) long -- pass as `budget_spec={"perhead": ...}` to a
        `BlockSelector`, or directly as `num_to_select` to
        `models.attention.ress_attention_forward`."""
        baseline = compute_baseline_num_to_select(attn_pooled, sparse_ratio, cdf_threshold)
        max_val = attn_pooled.shape[-1]
        scores_i = self.head_scores[layer_idx].to(attn_pooled.device)
        return adjust_attention_selection(baseline, scores_i, beta, max_val, floor_frac=self.floor_frac)
