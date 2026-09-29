"""Pi3 backbone wrapper: the same ReSS block-sparse global attention wired into Pi3
instead of VGGT.

Pi3 (`external/Pi3`) is architecturally close
enough to VGGT that nothing in `models/attention.py`, `budget/`, `scoring/` or
`runtime/` needs a Pi3-specific path -- only which blocks to patch differs:

- VGGT keeps frame and global attention in two separate ModuleLists
  (`aggregator.frame_blocks` / `aggregator.global_blocks`). Pi3 interleaves them in ONE
  `model.decoder` ModuleList of 36 `BlockRope`s: even indices see one frame at a time
  (`hidden` reshaped to (B*N, hw, C)), odd indices see all frames jointly
  ((B, N*hw, C)) -- so the 18 odd blocks are exactly the "global" ones ReSS sparsifies,
  and `Pi3.decode` needs no patching at all (it already produces the shapes each block
  expects; the reference implementation patched it only to insert an optional Hilbert
  permutation, which ReSS does not use -- see `models.vggt.wire_vggt_aggregator`).
- Its attention module (`FlashAttentionRope`) exposes the same
  `qkv`/`q_norm`/`k_norm`/`rope`/`num_heads`/`proj`/`proj_drop` attributes
  `models.qkv.compute_patch_qkv`/`project_out` read, applies RoPE with the same
  `PositionGetter` (y,x) convention (+1 shift, zeros for special tokens), and lays tokens
  out frame-major with `patch_start_idx=5` special tokens per frame -- identical to VGGT.
  Only the keyword differs (`xpos=` vs `pos=`), which `models.wiring`'s wrapper accepts.
- Decoder width matches VGGT-1B's global blocks (dim 1024, 16 heads -> head_dim 64), so
  the SpargeAttn kernel's constraints are satisfied unchanged.

Note the shape difference the sparsity math sees: Pi3 patches are 14x14 like VGGT, but
Pi3 has 18 sparsifiable layers to VGGT's 24, so the per-layer calibration table
(`budget.hess.HessBudget`'s CO3Dv2 Fisher scores) is regenerated for Pi3 rather than
reused -- see `scripts/calibrate_fisher_scores.py`.
"""

from ress._pathsetup import PI3_PRETRAINED
from ress.models.wiring import wire_ress_attention

N_GLOBAL_BLOCKS = 18


def global_block_indices(model):
    """Indices into `model.decoder` of the cross-frame ("global") blocks -- the odd ones
    (`Pi3.decode`'s `if i % 2 == 0: ... else: ...` reshape split)."""
    return list(range(1, len(model.decoder), 2))


def checkpointable_blocks(model):
    """Every transformer block worth wrapping in `torch.utils.checkpoint` for a backward
    pass (`scripts/calibrate_fisher_scores.py`): the whole decoder, not just the 18
    sparsifiable cross-frame blocks, plus the DINOv2 encoder. Inference never needs this
    (`load_pi3` deliberately takes no `checkpointing` flag); a 20-view backward does."""
    return list(model.decoder) + list(model.encoder.blocks)


def load_pi3(pretrained=PI3_PRETRAINED, aux_output=False, aux_full=True, device="cuda"):
    """Load a fresh Pi3 and wire its global decoder blocks for ReSS sparse/dense
    attention. Same contract as `models.vggt.load_vggt`: `aux_output=True` allocates a
    per-layer dict receiving `AttentionAux` on every forward call (`aux_full=False`
    populates only `.sparsity`/`.shape`, which is all the eval driver reads).

    No `checkpointing` flag (unlike `load_vggt`): that exists there for gradient probe
    passes, and every Pi3 path so far -- inference and the forward-only drift
    calibration -- runs under `no_grad`.
    """
    from pi3.models.pi3 import Pi3

    model = Pi3.from_pretrained(pretrained)
    blocks = [model.decoder[i] for i in global_block_indices(model)]
    assert len(blocks) == N_GLOBAL_BLOCKS, f"expected {N_GLOBAL_BLOCKS} global blocks, got {len(blocks)}"

    aux_output_store = {i: {} for i in range(len(blocks))} if aux_output else None
    wire_ress_attention(model, blocks, aux_output_store=aux_output_store, aux_full=aux_full,
                          num_special_tokens=model.patch_start_idx)
    model.aux_output_store = aux_output_store
    model.to(device)
    model.eval()
    # Lets `ress.api`'s apply_* find the right calibration table and head count
    # without the caller naming the backbone a second time.
    model._ress_backbone = "pi3"
    return model
