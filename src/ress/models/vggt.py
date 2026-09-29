"""VGGT wrapper: replaces each global block's attention forward with
`ress_attention_forward`.

`sparse_vggt_aggregator_forward` below follows the SparseVGGT reference implementation
(https://github.com/brianwang00001/sparse-vggt, `models/vggt.py`); the wiring around it is
this project's. That repository carries no licence; see this project's README.

The aggregator forward itself (frame/global block interleaving, the Hilbert permutation,
intermediate collection) is VGGT's own architecture, not part of either method -- it is
spelled out rather than called because the global-attention step inside it is what gets
replaced.
"""

from functools import partial
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.models.vggt import VGGT

from ress._pathsetup import VGGT_PRETRAINED
from ress.models.hilbert import hilbert_permute
from ress.models.wiring import global_blocks, wire_ress_attention

N_GLOBAL_BLOCKS = 24


def sparse_vggt_aggregator_forward(
    self,
    images: torch.Tensor,
    use_hilbert: bool = True,
    intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
):
    B, N, C_in, H, W = images.shape
    if C_in != 3:
        raise ValueError(f"Expected 3 input channels, got {C_in}")

    images = (images - self._resnet_mean) / self._resnet_std
    images = images.view(B * N, C_in, H, W)
    patch_tokens = self.patch_embed(images)
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]
    _, P, C = patch_tokens.shape

    camera_token = slice_expand_and_flatten(self.camera_token, B, N)
    register_token = slice_expand_and_flatten(self.register_token, B, N)
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

    H = H // self.patch_size
    W = W // self.patch_size
    pos = None
    if self.rope is not None:
        pos = self.position_getter(B * N, H, W, device=images.device)
    if self.patch_start_idx > 0 and pos is not None:
        pos = pos + 1
        pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)
    _, P, C = tokens.shape

    frame_idx = 0
    global_idx = 0
    output_list = []

    S = self.patch_start_idx
    if use_hilbert:
        tokens = hilbert_permute(tokens, H, W, S)
        pos = hilbert_permute(pos, H, W, S)

    concat_inter = None
    frame_intermediates, global_intermediates = [], []
    for layer_idx in range(self.aa_block_num):
        for attn_type in self.aa_order:
            if attn_type == "frame":
                tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                    tokens, B, N, P, C, frame_idx, pos=pos
                )
            elif attn_type == "global":
                tokens, global_idx, global_intermediates = self._process_global_attention(
                    tokens, B, N, P, C, global_idx, pos=pos
                )
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")
        if layer_idx in intermediate_layer_idx:
            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        else:
            output_list.append(None)
    del concat_inter, frame_intermediates, global_intermediates

    if use_hilbert:
        for i, inter_x in enumerate(output_list):
            if inter_x is not None:
                output_list[i] = hilbert_permute(inter_x, H, W, S, reverse=True)
    return output_list, self.patch_start_idx


def wire_vggt_aggregator(model, aggregator: Aggregator, aux_output_store: dict | None = None,
                          aux_full: bool = True):
    """Monkey-patch aggregator.forward, then hand the global blocks to the shared
    `models.wiring.wire_ress_attention` (which patches each block's attn.forward and
    registers the block list for `global_blocks(model)`). Only the aggregator-forward
    replacement below is VGGT-specific; the attention patching itself is shared with
    every other backbone.

    [bug fix] `sparse_vggt_aggregator_forward`'s own signature defaults
    to `use_hilbert=True`, and this used to bind it with no override -- so every
    ReSS forward pass silently ran with Hilbert-curve-permuted patch tokens. The
    SparseVGGT's own entry point defaults to `use_hilbert=False` -- found by diffing two
    fresh model instances' aggregator output tensor-for-tensor on an identical scene:
    layer 0 (and every later layer) matched bit-for-bit ONLY after forcing
    use_hilbert=False here.
    Hilbert curve reorders patch tokens spatially before block-sparse global
    attention's fixed-size block pooling -- dense/frame attention is permutation-
    equivariant so it barely matters there, but which patches get pooled into the
    same coarse "block" (and hence which blocks a query keeps/drops) is completely
    different under Hilbert vs raster order. This silently changed the actual
    sparse attention pattern (not just its representation) for every sweep
    result predating the fix -- it explained a ~5pt CO3Dv2
    uniform-baseline AUC gap vs the HeSS paper's own Table S1 number that survived
    exhaustive line-by-line/tensor-by-tensor auditing of everything else (beta
    reallocation, cdf/topk math, sink_blocks, predict_attention pooling, tokens
    split, kernel call args, dataset/seed, checkpointing, cudnn determinism,
    checkpoint weights) -- every one of those matched, and only this did not."""
    aggregator.forward = MethodType(partial(sparse_vggt_aggregator_forward, use_hilbert=False), aggregator)
    wire_ress_attention(model, aggregator.global_blocks, aux_output_store=aux_output_store,
                          aux_full=aux_full, num_special_tokens=aggregator.patch_start_idx)
    return aggregator


def _apply_checkpointing(module):
    """Gradient checkpointing, applied before any calibration or probe backward pass to bound
    activation memory across 24 global + 24 frame blocks) -- NOT optional for probe
    passes: a full-resolution (~20 view) scene's non-checkpointed forward+backward
    with `retain_grad()` on every layer's O_h was found to OOM a 47GB GPU (found
    empirically running phase26_c1_reexamination.py at full scene resolution)."""
    original_forward = module.forward

    def new_forward(*args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    module.forward = new_forward


def load_vggt(pretrained=VGGT_PRETRAINED, aux_output=False, aux_full=True, device="cuda", checkpointing=True):
    """Load a fresh VGGT-1B and wire it for ReSS sparse/dense global attention.
    aux_output=True allocates a per-layer dict to receive `AttentionAux` (P^h, V̄,
    q_patch) on every forward call -- needed for damage/probe computation, skipped
    for plain runtime inference. aux_full=False (only meaningful when aux_output=True)
    populates just `.sparsity`/`.shape` per layer instead of the full attn_pooled/
    v_bar/q_patch/output tensors -- use this for callers that only read `.sparsity`
    (e.g. sweep scripts): those big tensors otherwise stay alive for ALL
    N_GLOBAL_BLOCKS layers simultaneously by the end of one forward pass (nothing
    frees layer i's entry when layer i+1 runs), a real OOM risk at high view counts
    that has nothing to do with the sparse attention computation itself
    (measured on a 100-view CO3Dv2 scene). checkpointing=True
    (default) wraps patch_embed + every frame/global block in `torch.utils.checkpoint`
    -- negligible cost for
    inference-only calls (checkpoint only recomputes on a backward pass, i.e. probe
    passes), essential for any probe/calibration pass at real scene resolution."""
    model = VGGT.from_pretrained(pretrained)
    aux_output_store = {i: {} for i in range(N_GLOBAL_BLOCKS)} if aux_output else None
    wire_vggt_aggregator(model, model.aggregator, aux_output_store, aux_full=aux_full)
    model.aux_output_store = aux_output_store
    model.to(device)
    model.eval()
    if checkpointing:
        _apply_checkpointing(model.aggregator.patch_embed)
        for frame_block in model.aggregator.frame_blocks:
            _apply_checkpointing(frame_block)
        # NOTE: global_blocks are deliberately NOT checkpointed here, even though
        # `retain_grad()`
        # on an internal intermediate (`AttentionAux.output`, set inside
        # `ress_attention_forward`) does not reliably work when the enclosing
        # function is checkpointed: `torch.utils.checkpoint` (even with
        # use_reentrant=False) only guarantees gradients for the checkpointed
        # function's own INPUTS: internal tensors are recomputed fresh during
        # backward and are not the same objects a caller captured during the
        # original (memory-saving) forward pass, so a hook/retain_grad set on the
        # original is disconnected from the actual backward computation. Since
        # ReSS's probe mechanism fundamentally needs each global block's O_h
        # gradient, checkpointing patch_embed + frame_blocks only (still a real
        # memory saving) is the safe subset; probe-pass callers should also use a
        # moderate view count (see sensitivity.probes docstring / script configs).
    # Lets `ress.api`'s apply_* find the right calibration table and head count
    # without the caller naming the backbone a second time.
    model._ress_backbone = "vggt"
    return model


def set_layer_budgets(model, num_to_select_per_layer: torch.Tensor | None, dense: bool = False,
                       block_criterion: str = "probability"):
    """num_to_select_per_layer: (n_blocks, B, nh, q_blk) or None (if dense=True).
    Always resets retain_output_grad/budget_fn -- callers doing real inference after
    a probe pass (`set_probe_mode`) must not inherit its retain_grad()-under-no_grad
    state (that combination raises, since retain_grad requires requires_grad=True).
    block_criterion: "probability" (default), "damage", or "norm" -- the
    block-cutoff-order handle, independent of the per-head budget in
    num_to_select_per_layer -- the budget handle, independent of the block-order one."""
    for i, block in enumerate(global_blocks(model)):
        block.attn.dense = dense
        block.attn.num_to_select = None if dense else num_to_select_per_layer[i]
        block.attn.retain_output_grad = False
        block.attn.budget_fn = None
        block.attn.joint_select_fn = None
        block.attn.block_criterion = block_criterion


def set_probe_mode(model, retain_output_grad: bool):
    """Probe passes need dense=True (SDPA is differentiable, SpargeAttn kernel is
    not) AND retain_output_grad=True (O_h is a non-leaf
    intermediate; its .grad after backward is what `sensitivity.probes` reads)."""
    set_layer_budgets(model, None, dense=True)
    for block in global_blocks(model):
        block.attn.retain_output_grad = retain_output_grad
