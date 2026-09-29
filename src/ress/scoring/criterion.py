"""Parser for the `block_criterion` restore-swap name grammar.

Grammar (suffixes may appear in ANY order -- they are stripped from the end in a loop):

    base    := ["damage_"] "restore_swap" ["_iter" N]
    suffix  := "_nowo" | "_wo" | "_bf16" | "_trackbest" | "_rf" NN | "_dc" NN
             | "_antiswap" | "_anti"

`_antiswap`/`_anti` are the drift-MAXIMIZATION negative controls: the same
pipeline run backwards, `_antiswap` flipping only the restore-swap rounds' objective and
`_anti` flipping the initial cut too. They exist to answer "is `J` the causal quantity, or
just a proxy for attention magnitude?" -- see `scoring.projected.restore_swap_rank_by`'s
`maximize`/`invert_cut` paragraph for why the swap-only arm is the informative one and the
full reversal is only the outer bracket. Not an ablation axis: no shipped recipe sets them.

`damage_` prefix and `_nowo` suffix mean the same thing (M = identity instead of
M = (W_O^h)^T W_O^h): the prefix is the legacy spelling, `_nowo`/`_wo` is the explicit
switch. This is the W_O ablation axis; `_wo` (the default) is "Ours".

Four design choices are NOT switchable, because they are settled on derivation rather
than measurement, and `restore_swap_rank_by` therefore has no code path for the
alternative:

- **kept-mass normalization.** The sparse kernel re-softmaxes over kept blocks, so the
  realized output error is exactly `delta = g / K` (`K` = kept probability mass), making
  `J = ||g||^2_M / K^2` the true squared perturbation. The unnormalized `||g||^2_M` is
  that times a per-query-row `K^2` -- a distortion, not an alternative objective.
- **compensate criterion = drift.** Restore and compensate are the same exact marginal
  applied in opposite directions; scoring one by the drift delta and the other by the
  raw score would be inconsistent.
- **the initial cut's `1/(1-p)` factor**. It is not a second convention
  sitting next to `1/K`: for a single-candidate drop set, `J = (s/(1-p))^2` exactly, so
  the cut and the rounds read ONE objective at two set sizes. Pinned by
  `tests/test_projected.py::test_renormalized_cut_is_the_single_drop_objective`.
  `RESS_RENORM=0` exists only to reproduce the earlier numbers.
- **sink-aware probabilities**. The kernel always keeps the special-token
  key blocks, so the model's distribution is the softmax over patch AND special blocks;
  scoring against the patch-conditional one drops 5% of the pooled mass on average (and
  over 90% on individual query rows) out of `P`, out of the reference `v̄`, and out of
  `K`. Nothing about the formulas changes -- only which quantities go in. See
  `scoring.projected.dense_reference`; `RESS_SINK_MASS=0` is the A/B escape hatch.
"""

import re
from dataclasses import dataclass

import torch

_SUFFIX_RE = re.compile(r"_(nowo|wo|bf16|trackbest|rf\d+|dc\d+|antiswap|anti)$")
_BASE_RE = re.compile(r"^(?P<damage>damage_)?restore_swap(?:_iter(?P<iters>\d+))?$")


@dataclass
class RestoreSwapSpec:
    """Everything `models.attention` needs to dispatch one restore-swap call.
    Field names match `scoring.projected.restore_swap_rank_by`'s parameters."""

    n_iters: int = 1
    use_wo: bool = True  # M = (W_O^h)^T W_O^h; False -> M = identity (the W_O ablation)
    track_best: bool = False
    big_dtype: torch.dtype | None = None
    # `_rf<NN>` -> NN/100. The default 0.5 is `restore_swap_rank_by`'s own, and under
    # geometric decay at n_iters=5 it spends a CUMULATIVE 0.484 x num_to_select -- i.e.
    # nearly half the kept blocks get swapped out across the rounds, which is already a
    # lot of movement. Only LARGER values were ever tried, and they measured worse; smaller
    # ones are unexplored.
    restore_frac: float = 0.5
    # `_antiswap` -> maximize; `_anti` -> maximize + invert_cut. Separate fields rather
    # than one enum because they are separate handles in `restore_swap_rank_by` and the
    # experiment reports them as two distinct arms.
    maximize: bool = False
    invert_cut: bool = False
    # `_dc<NN>` -> NN/100. Geometric-decay ratio: round i gets `decay*(1-decay)^i` of
    # `restore_frac`. The hardcoded 0.5 puts 51.7% of the whole spend into round 0, in one
    # un-re-evaluated batch -- the step the drift diagnostic measured as roughly doubling
    # ||g||^2_M. Lower = flatter schedule, smaller first spike, nearly the same total.
    decay: float = 0.5


def is_restore_swap(block_criterion: str) -> bool:
    """True iff `block_criterion` is a name `parse_restore_swap` can handle."""
    return _BASE_RE.match(_strip_suffixes(block_criterion)[0]) is not None


def _strip_suffixes(block_criterion: str):
    core = block_criterion
    found = []
    while True:
        m = _SUFFIX_RE.search(core)
        if m is None:
            return core, found
        found.append(m.group(1))
        core = core[: m.start()]


def parse_restore_swap(block_criterion: str) -> RestoreSwapSpec:
    core, suffixes = _strip_suffixes(block_criterion)
    base = _BASE_RE.match(core)
    if base is None:
        raise ValueError(f"not a restore-swap block_criterion: {block_criterion!r}")

    spec = RestoreSwapSpec(
        n_iters=int(base.group("iters")) if base.group("iters") else 1,
        use_wo=base.group("damage") is None,
    )
    for s in suffixes:
        if s == "nowo":
            spec.use_wo = False
        elif s == "wo":
            spec.use_wo = True
        elif s == "trackbest":
            spec.track_best = True
        elif s == "antiswap":
            spec.maximize = True
        elif s == "anti":
            spec.maximize = True
            spec.invert_cut = True
        elif s.startswith("rf"):
            spec.restore_frac = int(s[2:]) / 100.0
        elif s.startswith("dc"):
            spec.decay = int(s[2:]) / 100.0
        else:  # "bf16"
            spec.big_dtype = torch.bfloat16
    return spec
