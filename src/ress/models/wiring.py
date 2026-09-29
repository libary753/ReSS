"""Backbone-agnostic access to (and patching of) the global-attention blocks ReSS
sparsifies.

Every ReSS runtime/scoring entry point (`runtime.static`, `runtime.joint`,
`runtime.beta_budget`, `scoring.projected.bake_and_attach_*`) needs the same thing:
"the list of blocks whose `.attn` I should configure, in layer order". That used to
be spelled `model.aggregator.global_blocks` inline at ~20 call sites, which hardcoded
VGGT's aggregator layout into code that is otherwise entirely backbone-independent
(the budget/score/restore math only ever touches `attn.qkv`/`attn.proj`/`attn.rope`,
which every VGGT-family backbone spells identically).

`global_blocks(model)` replaces those references. A wired model carries its own
`ress_global_blocks` list (set by `wire_ress_attention` at load time), so each
backbone decides for itself which of its blocks are "global":

- VGGT  (`models.vggt`): `model.aggregator.global_blocks` -- 24 blocks, interleaved
  with an equal number of frame blocks that are left dense.
- Pi3   (`models.pi3`): `model.decoder[1::2]` -- 18 of 36 decoder blocks; Pi3
  alternates per-frame (even) and cross-frame (odd) attention over the SAME
  `decoder` ModuleList instead of keeping two separate lists.

The `model.aggregator.global_blocks` fallback is kept for models that were never
passed through `wire_ress_attention` (plain `VGGT.from_pretrained`, and
`tests/test_runtime_mode_isolation.py`'s fake model), so this is a pure
generalization -- no existing caller changes behavior.
"""

from types import MethodType

from ress.models.attention import ress_attention_forward


def global_blocks(model):
    """The transformer blocks ReSS sparsifies, in layer order. Each has an `.attn`
    module with `qkv`/`q_norm`/`k_norm`/`rope`/`proj`/`proj_drop` (the attribute set
    `models.qkv` needs)."""
    blocks = getattr(model, "ress_global_blocks", None)
    if blocks is not None:
        return blocks
    return model.aggregator.global_blocks


def global_attns(model):
    """Convenience view of `global_blocks(model)`'s `.attn` modules -- these are what
    carry the per-call config attributes (`num_to_select`, `block_criterion`, `M`, ...)."""
    return [block.attn for block in global_blocks(model)]


def wire_ress_attention(model, blocks, aux_output_store=None, aux_full=True,
                          num_special_tokens=5):
    """Replace each block's `attn.forward` with `ress_attention_forward`, and record
    `blocks` on the model for `global_blocks` to find.

    Per-call sparsity config (`num_to_select`/`dense`/`block_criterion`/...) is read
    off each attn module as a plain attribute at forward time (set by `runtime/*.py`
    right before each forward call) rather than baked in here, since ReSS budgets are
    recomputed per input/chunk.

    The wrapper accepts the positional encoding under either name -- VGGT's blocks call
    `self.attn(self.norm1(x), pos=pos)`, Pi3's `BlockRope` calls
    `self.attn(self.norm1(x), xpos=xpos)` -- so no per-backbone forward shim is needed.
    `attn_bias`/`attn_mask` are accepted (and must be None) only because Pi3's
    `FlashAttentionRope` and DA3's `Attention` respectively have them in their signatures;
    no call site in the sparsified layers passes one. They are asserted rather than
    ignored: a real mask would silently not be applied, since `ress_attention_forward`'s
    block-sparse path builds its own mask from the budget.

    `num_heads` is read off each attn module itself (every backbone here sets it in
    `__init__`), so mixed-width backbones would work without extra wiring;
    `num_special_tokens` is a per-model constant (`patch_start_idx`) and is passed in.
    """
    blocks = list(blocks)
    for i, block in enumerate(blocks):
        store_i = aux_output_store[i] if aux_output_store is not None else None

        def _fwd(self, x, pos=None, xpos=None, attn_bias=None, attn_mask=None,
                  _store=store_i, _attn=block.attn):
            assert attn_bias is None and attn_mask is None, (
                "ress_attention_forward builds its own block mask and cannot honor an "
                "externally supplied attn_bias/attn_mask"
            )
            return ress_attention_forward(
                _attn,
                x,
                pos if pos is not None else xpos,
                num_to_select=getattr(_attn, "num_to_select", None),
                budget_fn=getattr(_attn, "budget_fn", None),
                perhead_budget_fn=getattr(_attn, "perhead_budget_fn", None),
                joint_select_fn=getattr(_attn, "joint_select_fn", None),
                block_criterion=getattr(_attn, "block_criterion", "probability"),
                num_special_tokens=num_special_tokens,
                num_heads=_attn.num_heads,
                shape=getattr(_attn, "ress_shape", None),
                aux_store=_store,
                aux_full=aux_full,
                dense=getattr(_attn, "dense", False),
                retain_output_grad=getattr(_attn, "retain_output_grad", False),
                pooled_fallback=getattr(_attn, "pooled_fallback", False),
                protect_own_frame=getattr(_attn, "protect_own_frame", False),
            )
            # `protect_special_tokens` used to be passed here. It was never set
            # by any caller, so it never ran; the guarantee it was meant to provide is now
            # unconditional -- `ress_attention_forward` excludes any block holding a special
            # key from the candidate axis in the first place.

        block.attn.forward = MethodType(_fwd, block.attn)

    model.ress_global_blocks = blocks
    return blocks
