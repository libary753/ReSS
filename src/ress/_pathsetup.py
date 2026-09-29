"""Puts the three backbone source trees on `sys.path`, and names the shipped assets.

The backbones (VGGT, Pi3, Depth-Anything-3) are third-party model code, not packages on
PyPI, so `scripts/setup_external.sh` clones each at a pinned commit under `external/` and
this module makes them importable. SpargeAttn is the exception: it is a compiled CUDA
extension and gets pip-installed into the environment, so it needs no path entry.

Importing this module has the side effect of extending `sys.path`. Every `ress` module
that reaches a backbone imports it first, so ordering inside those files is load-bearing
and isort must not reorder it.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL = os.path.join(PROJECT_ROOT, "external")

# VGGT. `models.vggt` wraps the architecture; nothing here modifies it beyond the
# one-line autocast patch in external/patches/vggt.patch.
VGGT_EXTERNAL = os.path.join(EXTERNAL, "vggt")

# Pi3. `pi3.models.layers.pos_embed` falls back to a pure-PyTorch RoPE2D when croco's
# compiled cuRoPE2D extension is absent, so this import needs nothing built.
PI3_EXTERNAL = os.path.join(EXTERNAL, "Pi3")

# Depth-Anything-3. Its package lives under `src/`, unlike the other two.
DA3_SRC = os.path.join(EXTERNAL, "depth-anything-3", "src")

for _p in (VGGT_EXTERNAL, PI3_EXTERNAL, DA3_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# HeSS's CO3Dv2 Fisher head-importance tables, shipped with the package so that using
# HeSS -- and therefore ReSS, whose head-wise budget reads the same tables -- needs no
# calibration run. `scripts/calibrate_fisher_scores.py` regenerates them.
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")

# Where `scripts/calibrate_fisher_scores.py` looks for the CO3Dv2 calibration set.
# Override with the CO3DV2_ROOT environment variable.
CO3DV2_DATA_DIR = os.environ.get("CO3DV2_ROOT", os.path.join(PROJECT_ROOT, "data", "co3dv2"))

VGGT_PRETRAINED = "facebook/VGGT-1B"
PI3_PRETRAINED = "yyfz233/Pi3"
DA3_PRETRAINED = "depth-anything/DA3-GIANT-1.1"
