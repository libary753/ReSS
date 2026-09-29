"""The public entry points: load a backbone, then put one attention mode on it.

Four modes, and they differ along exactly two axes -- WHICH BUDGET decides how many key
blocks each (head, query-block) keeps, and WHICH ORDER those blocks are ranked in:

    mode           budget                    block order
    ------------   -----------------------   --------------------------------
    dense          -- (no sparsification)    --
    sparse_vggt    uniform across heads      attention probability
    hess           calibrated per head       attention probability
    ress           calibrated per head       residual-drift restore-swap

So `hess` is `sparse_vggt` with the head budget reallocated by HeSS's calibrated
importance, and `ress` is `hess` with the block ranking replaced by ReSS's residual-
drift criterion. Reading down the table is the ablation between the two papers.

Every `apply_*` fully resets the previous mode, so you can switch modes on one loaded
model between forward passes without reloading.

    import ress

    model = ress.load_backbone("vggt")
    ress.apply_ress(model, sparse_ratio=0.1)
    predictions = model(images)

A model you loaded yourself works too, as long as ReSS's attention is wired into it and
you say which backbone it is:

    from ress.models.vggt import load_vggt
    model = load_vggt()                       # already wired and tagged
    ress.apply_hess(model, sparse_ratio=0.1, cdf_threshold=0.97)
"""

from ress.backbones import get_backbone, infer_backbone
from ress.budget.hess import HessBudget
from ress.budget.uniform import UniformBudget
from ress.runtime.static import set_static_budget
from ress.scoring.projected import bake_and_attach_M

# The restore-swap configuration every number in the ReSS paper was measured with. The name is
# a grammar (`scoring/criterion.py`): 5 restore/swap rounds, the drift metric accumulated
# in bf16, and "trackbest" keeping the best-scoring mask seen across rounds rather than
# the last one. `apply_ress`'s `n_iters` rewrites the iter field.
RESS_CRITERION = "restore_swap_iter{n_iters}_bf16_trackbest"


def load_backbone(name, device="cuda", **kwargs):
    """Load one of "vggt" / "pi3" / "da3" with ReSS's attention wired in.

    Downloads the published checkpoint through huggingface_hub on first use. The model
    comes back DENSE -- call one of the `apply_*` functions to sparsify it.
    """
    backbone = get_backbone(name)
    model = backbone.load(device=device, **kwargs)
    # So `apply_*` can find the right calibration table without being told again.
    model._ress_backbone = backbone.name
    return model


def _budgets(model, backbone, lam=None):
    """(uniform, hess) budgets for this model, built once and cached on it.

    `bake_and_attach_M` precomputes the per-(layer, head) metric M_h = (W_O^h)^T W_O^h
    that the drift criterion scores against. It depends only on the weights, so it is
    scene-independent and belongs here rather than in the forward pass.
    """
    key = ("_ress_budgets", lam)
    cached = getattr(model, "_ress_budget_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    if not getattr(model, "_ress_M_baked", False):
        bake_and_attach_M(model)
        model._ress_M_baked = True

    uniform = UniformBudget(n_blocks=backbone.n_global_blocks, n_heads=backbone.n_heads)
    score_pcd, score_cam_pos = backbone.calibration_paths()
    # Argument order matters: `calibration_paths` returns (pcd, cam_pos) and HessBudget
    # takes (cam, pc). At lam=0.5 the combination is symmetric, but Pi3 runs lam=0 and DA3
    # lam=1, where swapping them selects exactly the opposite table.
    hess = HessBudget(
        score_cam_pos, score_pcd,
        lam=backbone.hess_lam if lam is None else lam,
        qkv_idx=0,          # HeSS scores W_Q; the k/v rows of the table go unused
    )
    model._ress_budget_cache = (key, (uniform, hess))
    return uniform, hess


def _resolve(model, backbone):
    spec = get_backbone(backbone) if backbone is not None else infer_backbone(model)
    return spec


def apply_dense(model, backbone=None):
    """Turn sparsification off -- the unmodified backbone, for a reference forward."""
    spec = _resolve(model, backbone)
    uniform, _ = _budgets(model, spec)
    set_static_budget(model, uniform, 0.0, None, beta=1.0)
    return model


def apply_sparse_vggt(model, sparse_ratio, cdf_threshold=None, backbone=None):
    """The SparseVGGT baseline: one budget shared by every head, blocks ranked by
    attention probability.

    `sparse_ratio` is the fraction of key blocks to drop. `cdf_threshold`, if given, also
    requires the kept blocks to capture that much of the attention mass, and the stricter
    of the two wins -- that pairing is SparseVGGT's own grid.
    """
    spec = _resolve(model, backbone)
    uniform, _ = _budgets(model, spec)
    set_static_budget(model, uniform, sparse_ratio, cdf_threshold, beta=1.0)
    return model


def apply_hess(model, sparse_ratio, cdf_threshold=None, lam=None, backbone=None):
    """HeSS: SparseVGGT's block order, but the per-head budget reallocated by the
    calibrated Fisher head-importance table shipped in `ress/calibration/`.

    `lam` overrides HeSS Eq(9)'s mix of the camera and point-cloud tables; the default is
    the value this backbone was tuned at (see `backbones.py`).
    """
    spec = _resolve(model, backbone)
    _, hess = _budgets(model, spec, lam=lam)
    set_static_budget(model, hess, sparse_ratio, cdf_threshold, beta=1.0)
    return model


def apply_ress(model, sparse_ratio, n_iters=5, lam=None, backbone=None):
    """ReSS: HeSS's head budget, with blocks chosen to minimise the residual drift the
    dropped blocks would cause, refined over `n_iters` restore/swap rounds.

    `n_iters=5` is what the ReSS paper reports. The rounds are cheap relative to the forward
    pass, and the mask tracked is the best-scoring one seen, not the last.

    No `cdf_threshold`: this arm takes its budget from `sparse_ratio` alone. The CDF
    cutoff is SparseVGGT's own knob, kept on the two baselines it belongs to.
    """
    spec = _resolve(model, backbone)
    _, hess = _budgets(model, spec, lam=lam)
    set_static_budget(model, hess, sparse_ratio, None, beta=1.0,
                      block_criterion=RESS_CRITERION.format(n_iters=n_iters))
    return model
