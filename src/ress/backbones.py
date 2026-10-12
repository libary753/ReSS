"""The three backbones ReSS and HeSS are wired into, and the constants each one needs.

A backbone contributes what the method cannot infer from the module tree: how many
sparsifiable (cross-frame) blocks it has and how many heads those carry -- together the
shape of the calibrated head-importance table -- and the lambda HeSS Eq(9) combines its
two tables with, which is NOT the same for all three.

The last three fields are needed only by `scripts/calibrate_fisher_scores.py`: a uniform
prediction contract to take a loss against, the confidence threshold that makes that
model's own conf activation a no-op, and which blocks to gradient-checkpoint.

Adding a fourth backbone means adding an entry here plus a loader under `models/` that
calls `wire_ress_attention` over its cross-frame blocks.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

from ress._pathsetup import CALIBRATION_DIR


@dataclass(frozen=True)
class Backbone:
    name: str
    n_global_blocks: int          # sparsifiable (cross-frame) blocks
    n_heads: int                  # attention heads per such block
    hess_lam: float            # HeSS Eq(9): HeSS = lam*HeSS_cam + (1-lam)*HeSS_pc
    load: Callable                # (device=..., **kw) -> model, already wired
    # --- calibration only (see the module docstring) ---------------------------------
    predict: Callable             # (model, images) -> {depth, conf, extrinsics, intrinsics}
    depth_conf_thr: float         # the value at which this model's conf filter is a no-op
    checkpointable_blocks: Callable  # (model) -> blocks to gradient-checkpoint

    def calibration_paths(self):
        """(pcd, cam_pos) Fisher head-importance tables shipped for this backbone.

        Both are `(3, n_global_blocks, n_heads)`: the leading axis is q/k/v, and HeSS
        reads index 0 (W_Q). Regenerate with `scripts/calibrate_fisher_scores.py`.
        """
        return (
            os.path.join(CALIBRATION_DIR, f"fisher_scores_{self.name}_pcd.pt"),
            os.path.join(CALIBRATION_DIR, f"fisher_scores_{self.name}_cam_pos.pt"),
        )


def _load_vggt(**kw):
    from ress.models.vggt import load_vggt
    return load_vggt(**kw)


def _load_pi3(**kw):
    from ress.models.pi3 import load_pi3
    return load_pi3(**kw)


def _load_da3(**kw):
    from ress.models.da3 import load_da3
    return load_da3(**kw)


def _checkpointable_vggt(model):
    """The global blocks only.

    The patch embedding and the frame blocks are `load_vggt`'s to cover, via its
    `checkpointing=True` -- which is NOT its default, so a caller that wants the whole
    model checkpointed has to ask for it there as well as calling this
    (`scripts/calibrate_fisher_scores.py` does both)."""
    from ress.models.wiring import global_blocks
    return list(global_blocks(model))


def _checkpointable_pi3(model):
    from ress.models.pi3 import checkpointable_blocks
    return checkpointable_blocks(model)


def _checkpointable_da3(model):
    from ress.models.da3 import checkpointable_blocks
    return checkpointable_blocks(model)


def _predict_vggt(model, images):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    H, W = images.shape[-2:]
    predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
    return {
        "depth": predictions["depth"][0, ..., 0],
        "conf": predictions["depth_conf"][0],
        "extrinsics": extrinsic[0],
        "intrinsics": intrinsic[0],
    }


def _predict_pi3(model, images):
    """Pi3 predicts a camera-space point map plus camera-to-world poses rather than a
    (depth, pose encoding) pair, so all four fields are derived here: depth is the
    camera-space Z, extrinsics invert the poses, conf is the sigmoid of the logit, and K is
    fitted (Pi3 predicts none -- see `ress.geometry.pinhole`)."""
    import torch

    from ress.geometry.pinhole import fit_pinhole_intrinsics

    predictions = model(images.unsqueeze(0))  # Pi3 takes (B, N, 3, H, W)
    local_points = predictions["local_points"][0].float()          # (N, H, W, 3)
    conf = torch.sigmoid(predictions["conf"][0, ..., 0].float())   # (N, H, W)
    c2w = predictions["camera_poses"][0].float()                   # (N, 4, 4)
    return {
        "depth": local_points[..., 2],
        "conf": conf,
        "extrinsics": torch.linalg.inv(c2w)[:, :3, :4],
        "intrinsics": fit_pinhole_intrinsics(local_points, conf),
    }


def _predict_da3(model, images):
    """DA3 predicts all four fields directly, so this is a rename plus the 4x4 -> 3x4
    slice: its extrinsics are already world-to-camera."""
    out = model(images.unsqueeze(0))  # DA3 takes (B, N, 3, H, W)
    return {
        "depth": out["depth"][0],
        "conf": out["depth_conf"][0],
        "extrinsics": out["extrinsics"][0][:, :3, :4],
        "intrinsics": out["intrinsics"][0],
    }


BACKBONES = {
    "vggt": Backbone(
        name="vggt",
        n_global_blocks=24,
        n_heads=16,
        # HeSS paper S4.1: "we set lambda = 0.5 unless mentioned otherwise" (VGGT).
        hess_lam=0.5,
        load=_load_vggt,
        predict=_predict_vggt,
        # VGGT's conf activation is expp1 (1 + exp(x)), so 1.0 is its floor and therefore
        # the value at which the filter passes everything.
        depth_conf_thr=1.0,
        checkpointable_blocks=_checkpointable_vggt,
    ),
    "pi3": Backbone(
        name="pi3",
        n_global_blocks=18,
        n_heads=16,
        # HeSS paper S4.5 (Generalization, on Pi3): "setting lambda = 0 -- using only
        # HeSS_pc -- is sufficient and works best in practice". So Pi3's HeSS budget is
        # the point-cloud table alone, not the mean of the two VGGT uses.
        hess_lam=0.0,
        load=_load_pi3,
        predict=_predict_pi3,
        # Pi3's conf is a sigmoid in [0, 1]; a 1.0 threshold here would mask every pixel.
        depth_conf_thr=0.0,
        checkpointable_blocks=_checkpointable_pi3,
    ),
    "da3": Backbone(
        name="da3",
        # ViT-g depth 40 with alt_start 13 -> cross-frame blocks 13, 15, ..., 39.
        n_global_blocks=14,
        n_heads=24,
        # No published HeSS number exists for DA3 (HeSS predates it), so lambda was swept
        # the way the HeSS paper itself swept it for Pi3 and the best value adopted. It is
        # monotone in lambda on every metric here, landing at the opposite end from Pi3:
        # DA3 wants the camera table alone. Which loss carries head importance turns out
        # to be backbone-specific rather than universal.
        hess_lam=1.0,
        load=_load_da3,
        predict=_predict_da3,
        depth_conf_thr=1.0,   # same expp1 activation as VGGT
        checkpointable_blocks=_checkpointable_da3,
    ),
}


def get_backbone(name):
    try:
        return BACKBONES[name]
    except KeyError:
        raise ValueError(
            f"unknown backbone {name!r} -- known: {', '.join(sorted(BACKBONES))}"
        ) from None


def infer_backbone(model):
    """Best-effort backbone lookup for a model the caller loaded themselves.

    `apply_*` needs the table shape and lambda, and asking the user to repeat a name they
    already expressed by loading a particular model is a good way to get a silently
    mismatched calibration table. The tag is stamped by each loader in `models/`; a model
    built some other way has none, and then the caller does have to say.
    """
    name = getattr(model, "_ress_backbone", None)
    if name is None:
        raise ValueError(
            "cannot tell which backbone this model is -- pass backbone=\"vggt\"/\"pi3\"/"
            "\"da3\" explicitly, or load it with ress.load_backbone(), which tags it."
        )
    return get_backbone(name)
