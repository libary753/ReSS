"""Official implementation of ReSS and HeSS, released together.

    ReSS: Residual-Restoring Sparse Attention for 3D Vision Transformers
    HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT  (CVPR 2026)

HeSS decides how many key blocks each attention head keeps; ReSS decides which ones, given
that budget. They share this codebase because ReSS's head budget IS HeSS's.

    import ress

    model = ress.load_backbone("vggt")        # or "pi3", "da3"
    ress.apply_ress(model, sparse_ratio=0.1)
    predictions = model(images)

`apply_hess`, `apply_sparse_vggt` and `apply_dense` select the other three modes on the
same loaded model; `ress.api` documents how they differ.
"""

from ress.api import (
    apply_dense,
    apply_hess,
    apply_ress,
    apply_sparse_vggt,
    load_backbone,
)
from ress.backbones import BACKBONES, get_backbone

__all__ = [
    "BACKBONES",
    "apply_dense",
    "apply_hess",
    "apply_ress",
    "apply_sparse_vggt",
    "get_backbone",
    "load_backbone",
]
