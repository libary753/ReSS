"""Depth-Anything-3 backbone wrapper (backbone-expansion experiments).

DA3 is a single DINOv2 tower rather than VGGT's two-stream aggregator, but from
`alt_start` onwards it runs the same alternating pattern under the hood
(`model/dinov2/vision_transformer.py::_get_intermediate_layers_not_chunked`):

    i >= alt_start and i % 2 == 1  ->  "global": rearrange(x, "b s n c -> b (s n) c")
    otherwise                      ->  "local":  rearrange(x, "b s n c -> (b s) n c")

so the sparsifiable set is exactly the odd blocks at or after `alt_start`. For the GIANT
preset (`configs/da3-giant.yaml`: vitg, depth 40, `alt_start=13`) that is blocks
13, 15, ..., 39 -- **14 layers**, versus VGGT's 24 and Pi3's 18.

Two DA3-specific details the shared code needs told about:

- **One special token per view, not five.** DA3 uses a single cls/camera token
  (`num_register_tokens=0` for this preset, and `alt_start` overwrites token 0 with the
  camera token), where VGGT and Pi3 both carry 5. `num_special_tokens` is passed
  explicitly to `models.wiring.wire_ress_attention`, so this is a parameter, not a
  special case.
- **Attention is called with `attn_mask=`.** `model/utils/block.py` calls
  `self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask)`; the wrapper accepts and
  asserts it (no DA3 call site in the sparsified layers passes a real mask).

- **Cross-view positions carry no patch grid.** DA3 passes `pos_nodiff` to exactly these
  blocks (all-ones for every patch token), so unlike VGGT/Pi3 the per-view token split
  cannot be read back out of `pos` -- see `_attach_shape_capture` below.

Everything else lines up with the other two backbones: RoPE and qk-norm are both active
from `rope_start`/`qknorm_start` = `alt_start` (so every sparsifiable block has them),
the token layout inside a global block is frame-major with the special token(s) first,
and vitg's 1536 dim / 24 heads gives head_dim 64, which the SpargeAttn kernel needs.

Calibrated budgets must be regenerated for this backbone (14x24 table, vs VGGT's 24x16)
-- see `scripts/calibrate_fisher_scores.py --backbone da3`.

Loading deliberately bypasses DA3's own `api.DepthAnything3` wrapper: importing that
module pulls in `utils.export`, which imports `moviepy` and `pycolmap` at module scope
for result-export paths this project never uses. Instead the network is assembled from
the CHECKPOINT'S OWN `config.json` (which embeds the full object graph) and its
`model.safetensors`, loaded with `strict=True` -- that is a stronger correctness check
than the wrapper gives, since it also guarantees the architecture matches the specific
`-1.1` checkpoint rather than whatever the vendored `configs/da3-giant.yaml` describes.
"""

import json

from ress._pathsetup import DA3_PRETRAINED
from ress.models.wiring import wire_ress_attention

NUM_SPECIAL_TOKENS = 1  # DA3's single cls/camera token per view (no register tokens)


def _vit(model):
    """The DinoVisionTransformer inside a `DepthAnything3Net`."""
    return model.backbone.pretrained


def global_block_indices(model):
    """Indices into the ViT's `blocks` that run cross-view ("global") attention -- the
    odd ones at or after `alt_start` (see module docstring)."""
    vit = _vit(model)
    assert vit.alt_start != -1, "this DA3 preset has no alternating (cross-view) attention"
    return [i for i in range(len(vit.blocks)) if i >= vit.alt_start and i % 2 == 1]


def _load_net(pretrained):
    """Build `DepthAnything3Net` from the checkpoint's own config + weights (see module
    docstring for why `api.DepthAnything3.from_pretrained` is not used)."""
    from depth_anything_3.cfg import create_object
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    with open(hf_hub_download(pretrained, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    net = create_object(OmegaConf.create(cfg))

    state = load_file(hf_hub_download(pretrained, "model.safetensors"))
    # The published checkpoint is saved from the `DepthAnything3` API wrapper, whose only
    # submodule is `self.model` -- strip that one level to load into the net directly.
    state = {k.removeprefix("model."): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)

    # The published DA3-GIANT-1.1 checkpoint genuinely omits the final conv of the DualDPT
    # AUX ("ray") head for levels 1-3 (`head.scratch.output_conv2_aux.{1,2,3}.2.*`, 6
    # tensors). That output is discarded before it can affect anything on our path: this
    # preset has a camera decoder, and `DepthAnything3Net._process_camera_estimation`
    # deletes `output.ray`/`ray_conf` whenever `cam_dec` is present (we never pass
    # `use_ray_pose=True`, the only mode that would consume them). Everything the eval
    # actually reads -- depth, depth_conf, extrinsics, intrinsics -- comes from fully
    # loaded weights. Asserted rather than silently tolerated so that a future checkpoint
    # missing anything ELSE still fails loudly.
    allowed_missing = {k for k in missing if k.startswith("head.scratch.output_conv2_aux.")}
    assert not unexpected, f"unexpected keys in DA3 checkpoint: {unexpected[:5]}"
    assert set(missing) == allowed_missing, (
        f"DA3 checkpoint is missing weights that ARE used at inference: "
        f"{sorted(set(missing) - allowed_missing)[:5]}"
    )
    return net


def _attach_shape_capture(model, blocks):
    """Record the true (views, patch-grid) shape on every sparsified attn module at the
    start of each forward.

    Necessary because DA3 hands its CROSS-VIEW blocks `pos_nodiff`
    (`vision_transformer._prepare_rope`): an all-ones position for every patch token,
    which deliberately removes spatial RoPE differences from cross-view attention. VGGT
    and Pi3 both pass the real `+1`-shifted (y,x) grid there, so `models.qkv.
    compute_patch_qkv` can recover H/W (and hence the per-view token count) straight from
    `pos` -- with DA3 that inference collapses to H=W=1 and splits the sequence into 2740
    "views" of 2 tokens each. Measured: sparsity came out 0.407 for a requested
    0.8, i.e. silently wrong rather than crashing, which is why `compute_patch_qkv` now
    also asserts `N*P == NP`."""
    vit = _vit(model)
    original = vit._get_intermediate_layers_not_chunked

    def _capture(x, *args, **kwargs):
        _, S, _, H, W = x.shape
        shape = {"N": S, "H": H // vit.patch_size, "W": W // vit.patch_size}
        for block in blocks:
            block.attn.ress_shape = shape
        return original(x, *args, **kwargs)

    vit._get_intermediate_layers_not_chunked = _capture


def checkpointable_blocks(model):
    """Every ViT block, not just the 14 sparsifiable cross-view ones -- see
    `models.pi3.checkpointable_blocks`."""
    return list(_vit(model).blocks)


def load_da3(pretrained=DA3_PRETRAINED, aux_output=False, aux_full=True, device="cuda"):
    """Load a DA3 preset and wire its cross-view blocks for ReSS sparse/dense attention.
    Same contract as `models.vggt.load_vggt` / `models.pi3.load_pi3`."""
    model = _load_net(pretrained)
    blocks = [_vit(model).blocks[i] for i in global_block_indices(model)]

    aux_output_store = {i: {} for i in range(len(blocks))} if aux_output else None
    wire_ress_attention(model, blocks, aux_output_store=aux_output_store, aux_full=aux_full,
                          num_special_tokens=NUM_SPECIAL_TOKENS)
    _attach_shape_capture(model, blocks)
    model.aux_output_store = aux_output_store
    model.to(device)
    model.eval()
    # Lets `ress.api`'s apply_* find the right calibration table and head count
    # without the caller naming the backbone a second time.
    model._ress_backbone = "da3"
    return model
