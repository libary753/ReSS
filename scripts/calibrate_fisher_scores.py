"""Regenerate HeSS's Fisher head-importance tables from the CO3Dv2 calibration set.

The tables this writes already ship in `ress/calibration/`, so nothing about using HeSS or
ReSS requires running this. It is here so the calibration is reproducible rather than an
opaque artifact, and so the method can be extended to a backbone that has no published
table.

HeSS scores each head by the Fisher-information trace of its QKV weight GRADIENT with
respect to a real downstream loss, averaged over CO3Dv2's dev split -- a sensitivity
measure, not a heuristic, and what `ress.budget.hess.HessBudget` consumes. Two loss
variants are run as two separate passes producing two tables (`*_pcd`, `*_cam_pos`), which
HeSS Eq. (9) then combines at the backbone's own lambda.

The model runs DENSE here: the block-sparse kernel is not differentiable, and what is being
measured is the unsparsified model's head sensitivities.

Reproducibility depends on one thing the loader cannot own. `Co3Dv2.get_scan` draws its
views with a bare `torch.randperm` and returns None if any drawn view is below the quality
floor, so a seed fixed inside it would make a rejected scene reject forever. The seed is
therefore set here, per scene index, immediately before each draw -- which also makes the
draw identical across backbones, so per-backbone tables differ only by the backbone.
Different `--view-seed` values give measurably different tables; see the README.

Usage (one pass per loss per backbone; both feed one HessBudget):

    CO3DV2_ROOT=/path/to/co3dv2 python -m scripts.calibrate_fisher_scores \\
        --backbone vggt --loss pcd
    CO3DV2_ROOT=/path/to/co3dv2 python -m scripts.calibrate_fisher_scores \\
        --backbone vggt --loss cam_pos

Needs the `calibrate` extra plus a local pytorch3d build -- see the README.
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from ress._pathsetup import CALIBRATION_DIR, CO3DV2_DATA_DIR
from ress.backbones import get_backbone
from ress.budget.hess import accumulate_fisher_trace_scores
from ress.budget.uniform import UniformBudget
from ress.datasets.co3dv2 import Co3Dv2
from ress.geometry.dense_align import dense_correspondence_sim3
from ress.models.wiring import global_blocks
from ress.runtime.static import set_static_budget

NUM_VIEWS = 20      # HeSS's own calibration default
VIEW_SEED = 1234    # arbitrary but fixed; only its stability matters


def _as_44(Rt):
    """(N,3,4) or (N,4,4) -> (N,4,4)."""
    if Rt.shape[-2] == 4:
        return Rt
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=Rt.device, dtype=Rt.dtype)
    return torch.cat([Rt, bottom.view(1, 1, 4).expand(Rt.shape[0], -1, -1)], dim=1)


def unproject_depth(depth, K, Rt):
    """(N,H,W) depth + (N,3,3) K + (N,3,4|4,4) world-to-camera -> (N,H,W,3) world points,
    differentiable in depth/K/Rt. The pixel convention (integer centres) matches
    `ress.geometry.dense_align`, i.e. the alignment this feeds."""
    _N, H, W = depth.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pix = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (H, W, 3)
    rays = torch.einsum("nij,hwj->nhwi", torch.linalg.inv(K), pix)
    cam_pts = rays * depth[..., None]

    c2w = torch.linalg.inv(_as_44(Rt))
    return torch.einsum("nij,nhwj->nhwi", c2w[:, :3, :3], cam_pts) + c2w[:, :3, 3][:, None, None, :]


def camera_centers(Rt):
    """(N,3,4|4,4) world-to-camera -> (N,3) camera centres in the world frame."""
    return torch.linalg.inv(_as_44(Rt))[:, :3, 3]


def apply_sim3(points, s, R, t):
    """gt ~= s * R @ pred + t -- the convention `dense_correspondence_sim3` returns."""
    return s * (points @ R.transpose(-1, -2)) + t


def gt_arrays(scene_data, n_views):
    """GT depth/K/extrinsics/validity from a CO3Dv2 scene, in the [N,H,W] convention
    `dense_correspondence_sim3` expects."""
    depth = scene_data["depth"]
    depth = depth.squeeze(1) if depth.dim() == 4 else depth  # (N,1,H,W) -> (N,H,W)
    K = scene_data["K"]
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(n_views, -1, -1)
    return (depth.float(), K.float(), scene_data["Rt"].float(),
            scene_data["depth_mask"].reshape(depth.shape).bool())


def _apply_checkpointing(module):
    original_forward = module.forward

    def new_forward(*a, **kw):
        return torch.utils.checkpoint.checkpoint(original_forward, *a, use_reentrant=False, **kw)

    module.forward = new_forward


def fisher_scores_path(backbone_name, loss):
    return os.path.join(CALIBRATION_DIR, f"fisher_scores_{backbone_name}_{loss}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backbone", default="vggt", choices=["vggt", "pi3", "da3"])
    parser.add_argument("--loss", default="pcd", choices=["pcd", "cam_pos"])
    parser.add_argument("--out", default=None,
                        help="output .pt (default: overwrite the shipped table for this "
                             "(backbone, loss) pair)")
    parser.add_argument("--view-seed", type=int, default=VIEW_SEED,
                        help="base seed for the per-scene view draw. Regenerate every table "
                             "you intend to compare within one result at the SAME seed")
    parser.add_argument("--num-views", type=int, default=NUM_VIEWS)
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="skip gradient checkpointing; faster, and much likelier to OOM")
    parser.add_argument("--co3dv2-root", default=CO3DV2_DATA_DIR,
                        help="CO3Dv2 root (default: $CO3DV2_ROOT, else ./data/co3dv2)")
    args = parser.parse_args()

    backbone = get_backbone(args.backbone)
    out_path = args.out or fisher_scores_path(backbone.name, args.loss)

    dataset = Co3Dv2(args.co3dv2_root, split="dev")
    print(f"CO3Dv2 dev: {dataset.size()} scenes | backbone={backbone.name} "
          f"loss={args.loss} -> {out_path}")

    load_kw = dict(device="cuda", aux_output=False, aux_full=False)
    if backbone.name == "vggt" and not args.no_checkpointing:
        # `load_vggt` defaults to inference, where checkpointing buys nothing. This pass
        # runs a backward, so ask for it: that call covers patch_embed and the frame
        # blocks, and the loop below adds the global blocks. The other two backbones have
        # no such flag -- all of their blocks come from `checkpointable_blocks`.
        load_kw["checkpointing"] = True
    model = backbone.load(**load_kw)

    # Only the sparsifiable blocks' QKV weights need gradients -- they are the ONLY thing
    # `accumulate_fisher_trace_scores` reads. Leaving the rest trainable makes autograd
    # retain activations for the whole encoder and every block running before the first
    # sparsifiable one; with them frozen the graph does not even begin until that block.
    # This is the difference between the calibration fitting in memory and OOMing on a
    # third of the scenes -- and skips are not neutral, they drop the LARGEST scenes and
    # bias the table.
    model.requires_grad_(False)
    for block in global_blocks(model):
        block.attn.qkv.weight.requires_grad_(True)

    # Dense (exact SDPA) forward: the block-sparse kernel is not differentiable, and the
    # calibration measures the UNSPARSIFIED model's head sensitivities.
    set_static_budget(model, UniformBudget(n_blocks=backbone.n_global_blocks,
                                           n_heads=backbone.n_heads), 0.0, None, beta=1.0)
    for block in global_blocks(model):
        block.attn.dense = True
    if not args.no_checkpointing:
        # Every transformer block the backbone has, not just the sparsifiable ones. Safe
        # here because the gradients read below are on leaf parameters, not on recomputed
        # internal activations.
        for block in backbone.checkpointable_blocks(model):
            _apply_checkpointing(block)

    n_blocks = len(global_blocks(model))
    n_heads = global_blocks(model)[0].attn.num_heads
    scores = torch.zeros((3, n_blocks, n_heads))  # (qkv, layer, head)
    n_used = 0

    def scene_gradient(scene_data):
        """One scene's forward + loss + backward, accumulating into `scores`.

        Deliberately a FUNCTION rather than an inline loop body: every tensor here (the
        prediction dict, the unprojected clouds, the loss and its whole autograd graph)
        must be unreachable before the next scene's forward allocates, and loop locals are
        not -- they stay bound until reassigned on the next iteration. That alone is enough
        to OOM a large backbone on the third scene.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = backbone.predict(model, scene_data["images"].cuda())
        depth = pred["depth"].float()
        K = pred["intrinsics"].float()
        Rt = pred["extrinsics"].float()

        # A non-finite predicted depth that the loss mask later excludes is NOT harmless.
        # `x[mask]`'s backward writes a ZERO gradient at excluded positions, and the
        # square's backward then evaluates 2*(pred - gt) * 0 = inf * 0 = NaN. The loss
        # comes out finite while the gradient is NaN. Guarding the loss cannot fix it; the
        # value has to be gone BEFORE it enters the graph, which is what `torch.where`
        # does -- its backward routes gradient only to the selected branch.
        finite_depth = torch.isfinite(depth) & (depth > 0)
        depth = torch.where(finite_depth, depth, torch.zeros_like(depth))

        # K/Rt are inverted inside `unproject_depth`; a singular one makes every pixel of
        # that view non-finite, which no per-pixel mask can rescue. That is a broken
        # prediction, not an outlier, so it fails loudly.
        for name, mat in (("intrinsics", K), ("extrinsics", _as_44(Rt))):
            if not torch.isfinite(mat).all():
                raise RuntimeError(f"model returned non-finite {name}")
            if not torch.isfinite(torch.linalg.inv(mat)).all():
                raise RuntimeError(f"model returned singular {name} (inverse is non-finite)")

        gt_depth, gt_K, gt_Rt, gt_valid = gt_arrays(scene_data, depth.shape[0])
        valid = (gt_valid.cpu().numpy()
                 & (pred["conf"].detach().float().cpu().numpy() > backbone.depth_conf_thr)
                 & finite_depth.detach().cpu().numpy())

        # Under no_grad, so the transform enters the loss as a CONSTANT.
        with torch.no_grad():
            s, R, t = dense_correspondence_sim3(
                gt_depth.cpu().numpy(), gt_K.cpu().numpy(), gt_Rt.cpu().numpy(),
                depth.detach().float().cpu().numpy(), K.detach().float().cpu().numpy(),
                Rt.detach().float().cpu().numpy(), valid,
            )
        s_t = torch.as_tensor(s, device=depth.device, dtype=torch.float32)
        R_t = torch.as_tensor(R, device=depth.device, dtype=torch.float32)
        t_t = torch.as_tensor(t, device=depth.device, dtype=torch.float32).reshape(3)

        if args.loss == "pcd":
            pred_pts = apply_sim3(unproject_depth(depth, K, Rt), s_t, R_t, t_t)
            gt_pts = unproject_depth(gt_depth.to(depth.device), gt_K.to(depth.device),
                                     gt_Rt.to(depth.device))
            # A GT-"valid" pixel can still unproject to a non-finite point, and ONE such
            # pixel inside the mask makes the mean non-finite, the gradients NaN, and --
            # since scores accumulate by summation over scenes -- the WHOLE table NaN.
            mask = (torch.as_tensor(valid, device=depth.device)
                    & torch.isfinite(pred_pts).all(dim=-1)
                    & torch.isfinite(gt_pts).all(dim=-1))
            n_dropped = int(torch.as_tensor(valid, device=depth.device).sum() - mask.sum())
            if n_dropped:
                print(f"[INFO] dropped {n_dropped} non-finite valid pixels", flush=True)
            if mask.sum() == 0:
                raise RuntimeError("no valid pixels")
            # Index BEFORE subtracting, not after. `(pred - gt)**2 .sum(-1)[mask]` gives the
            # same loss but a NaN gradient: at a masked-out pixel grad_out is 0 while
            # (pred - gt) can be +-inf, and inf*0 = NaN, which then reaches EVERY parameter.
            # It does not matter that `gt_pts` needs no gradient -- its VALUE is what gets
            # multiplied. Gathering first means only finite, in-mask points enter the
            # arithmetic at all.
            diff = pred_pts[mask] - gt_pts[mask]
            loss = (diff ** 2).sum(dim=-1).mean()
        else:
            pred_c = apply_sim3(camera_centers(Rt), s_t, R_t, t_t)
            gt_c = camera_centers(gt_Rt.to(depth.device))
            finite = torch.isfinite(pred_c).all(dim=-1) & torch.isfinite(gt_c).all(dim=-1)
            if finite.sum() == 0:
                raise RuntimeError("no finite camera centres")
            loss = ((pred_c[finite] - gt_c[finite]) ** 2).sum(dim=-1).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss ({loss.item()}) -- refusing to poison the table")
        loss.backward()

        # Localise a non-finite gradient BEFORE it reaches `scores`, where the per-head sum
        # would hide which layer produced it.
        bad = [(i, blk.attn.qkv.weight.grad) for i, blk in enumerate(global_blocks(model))
               if blk.attn.qkv.weight.grad is not None
               and not torch.isfinite(blk.attn.qkv.weight.grad).all()]
        if bad:
            n_nan = sum(int(torch.isnan(g).sum()) for _, g in bad)
            n_inf = sum(int(torch.isinf(g).sum()) for _, g in bad)
            raise RuntimeError(
                f"non-finite gradient in layers {[i for i, _ in bad]} "
                f"({n_nan} NaN, {n_inf} Inf) -- the table would be NaN")

        accumulate_fisher_trace_scores(model, scores)
        if not torch.isfinite(scores).all():
            raise RuntimeError("non-finite gradient accumulated -- the table would be NaN")

    for idx in tqdm(range(dataset.size()),
                    desc=f"Fisher calibration ({backbone.name}/{args.loss})"):
        try:
            torch.manual_seed(args.view_seed + idx)   # see the module docstring
            scene_data = dataset.get_scan(idx, n_views=args.num_views)
        except Exception as ex:  # noqa: BLE001 -- a scene that will not LOAD is skipped;
            # a scene that loads and then fails is re-raised below, since that one biases
            # the table.
            print(f"[WARN] scan {idx} failed to load ({ex}), skipping")
            continue
        if scene_data is None:
            print(f"[WARN] scan {idx} returned None (low quality views), skipping")
            continue

        try:
            scene_gradient(scene_data)
            n_used += 1
        except Exception as ex:
            # Deliberately NOT swallowed: a dropped scene is not neutral (an OOM drops the
            # LARGEST scenes), so this fails loudly rather than quietly averaging over an
            # unrepresentative subset.
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"scan {idx} failed ({type(ex).__name__}: {ex}) -- calibration must cover "
                f"every scene; if this is an OOM, lower --num-views uniformly across "
                f"backbones rather than skipping scenes"
            ) from ex
        finally:
            del scene_data
            torch.cuda.empty_cache()

    assert n_used > 0, "no scene produced a usable gradient"
    scores /= n_used
    print(f"used {n_used}/{dataset.size()} scenes | shape {tuple(scores.shape)}")
    print(f"per-layer mean (q): {scores[0].mean(dim=-1)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(scores, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
