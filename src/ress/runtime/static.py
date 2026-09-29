"""Wires a budget object into every global-attention block of a loaded model.

This is the single place any mode is switched on, and it owns the reset: each call clears
every per-call knob the previous mode set, so switching modes between forward passes
cannot leave a stale hook behind that silently wins.

The budget itself stays behind an interface -- anything with
`.num_to_select(layer_idx, attn_pooled, sparse_ratio, cdf_threshold, beta) -> (nh, q_blk)
long` plugs in -- so adding an allocation rule does not mean touching this loop.

Budgets here are CONTENT-DEPENDENT and cannot be precomputed: HeSS's cutoff is read off
the attention map of the input actually being processed. So what gets installed is a
callback, invoked with each layer's just-computed `attn_pooled` from inside
`models.attention.ress_attention_forward`.
"""

from ress.models.wiring import global_blocks


def set_static_budget(model, budget, sparse_ratio=None, cdf_threshold=None, beta=1.0,
                       block_criterion="probability", pooled_fallback=False,
                       protect_own_frame=False):
    """budget: an object with `.num_to_select(layer_idx, attn_pooled, sparse_ratio,
    cdf_threshold, beta) -> (nh, q_blk) long` (see `budget.hess.HessBudget` /
    `budget.uniform.UniformBudget`). block_criterion: "probability" (default,
    HeSS-equivalent block cutoff order == `selection.topk_perhead` with
    `scoring.prob`), "damage"/"norm" (isolates the block-cutoff-order handle
    independent of the head budget -- the two handles are deliberately separate).
    pooled_fallback: see `models.attention.ress_attention_forward`'s own docstring
    -- an independent knob from `block_criterion` (mixed-resolution attention:
    dropped blocks fall back to a pooled token instead of vanishing), explicitly
    reset to False here like every other per-call knob this function owns, so a
    prior call with `pooled_fallback=True` doesn't silently leak into a later
    method that didn't ask for it.
    protect_own_frame: see `models.attention.ress_attention_forward`'s own
    docstring ("diagonal" own-frame protection) -- default False
    preserves every existing method's exact prior behavior/measured numbers."""
    needs_v_bar = getattr(budget, "needs_v_bar", False)
    # Sniffed opt-in (`budget.uniform.UniformBudget`'s
    # `query_beta` dial): a budget whose `.num_to_select` needs V̄ (to score
    # per-row hardness, `scoring.damage.row_damage_at_budget`) is wired via the
    # existing `perhead_budget_fn(attn_pooled, v_bar)` hook (already used by
    # `runtime.beta_budget`, takes precedence over `budget_fn` in
    # `ress_attention_forward`) instead of the plain `budget_fn(attn_pooled)`
    # hook every other budget object uses -- every budget that never sets
    # `needs_v_bar` (HessBudget, plain UniformBudget()) is wired exactly as
    # before, byte-for-byte.
    for i, block in enumerate(global_blocks(model)):
        block.attn.dense = False
        block.attn.num_to_select = None
        block.attn.retain_output_grad = False
        block.attn.block_criterion = block_criterion
        block.attn.pooled_fallback = pooled_fallback
        block.attn.protect_own_frame = protect_own_frame
        # MUST clear a stale joint_select_fn/perhead_budget_fn from a prior
        # runtime.joint.set_joint_budget/runtime.beta_budget.set_beta_budget call --
        # ress_attention_forward checks `if joint_select_fn is not None` BEFORE
        # falling back to num_to_select/budget_fn (and `budget_fn` before
        # `perhead_budget_fn`), so leaving either set silently makes every method
        # configured afterwards re-evaluate the OLD config instead of the one this
        # call just set up (found via a real sweep: 6 supposedly-different baseline
        # methods all returned bit-identical AUC/sparsity once set_joint_budget had
        # been called once).
        block.attn.joint_select_fn = None
        block.attn.perhead_budget_fn = None
        block.attn.budget_fn = None

        if needs_v_bar:
            def _perhead_budget_fn(attn_pooled, v_bar, _layer=i):
                return budget.num_to_select(_layer, attn_pooled, sparse_ratio, cdf_threshold, beta, v_bar=v_bar)

            block.attn.perhead_budget_fn = _perhead_budget_fn
        else:
            def _budget_fn(attn_pooled, _layer=i):
                return budget.num_to_select(_layer, attn_pooled, sparse_ratio, cdf_threshold, beta)

            block.attn.budget_fn = _budget_fn
