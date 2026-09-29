#!/usr/bin/env bash
#
# Fetches the third-party code ReSS runs on: three backbones (imported from source, see
# src/ress/_pathsetup.py) and the SpargeAttn block-sparse CUDA kernel (compiled and
# installed). Each is pinned to the exact commit every number in the paper was measured
# with -- upstream moves, and two of these trees need local patches, so "clone the latest"
# does not reproduce the method.
#
# Run from the repository root:  bash scripts/setup_external.sh
#
# Pass --skip-build to fetch and patch the sources without compiling SpargeAttn, for a
# machine that has no GPU at setup time (a login node, say) or that already has the
# extension installed.
#
# Requirements: git, and for the SpargeAttn build, nvcc plus a visible GPU of compute
# capability 8.0/8.6/8.7/8.9/9.0 (its setup.py rejects anything else -- notably Blackwell
# sm_120, e.g. the RTX PRO 6000). Point CUDA_HOME at a CUDA whose major.minor matches the
# torch wheel in your environment; a mismatch typically builds and imports fine, then
# fails at the first kernel launch.
set -euo pipefail

SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
PATCHES="$EXT/patches"
mkdir -p "$EXT"

# name | url | commit | patch (empty for none)
REPOS=(
  "vggt|https://github.com/facebookresearch/vggt.git|44b3afbd1869d8bde4894dd8ea1e293112dd5eba|vggt.patch"
  "Pi3|https://github.com/yyfz/Pi3.git|42e5ec872350819b0dc9d52edd3fe5b992f818b7|"
  "depth-anything-3|https://github.com/ByteDance-Seed/depth-anything-3.git|3d835ec1a5802d64a8b8b15f817a1ab54809bfe4|"
  "SpargeAttn|https://github.com/thu-ml/SpargeAttn.git|7cb07d132aac910e14fb64887c9006b7c19863cc|spargeattn.patch"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit patch <<< "$entry"
  dest="$EXT/$name"
  if [ -d "$dest/.git" ]; then
    echo "[skip] $name already present at $dest"
    continue
  fi
  echo "[clone] $name -> $dest"
  git clone --quiet "$url" "$dest"
  git -C "$dest" checkout --quiet "$commit"
  if [ -n "$patch" ]; then
    echo "[patch] $name <- patches/$patch"
    git -C "$dest" apply "$PATCHES/$patch"
  fi
done

if [ "$SKIP_BUILD" -eq 1 ]; then
  echo
  echo "[done] sources fetched and patched; SpargeAttn build skipped (--skip-build)"
  exit 0
fi

echo
echo "[build] SpargeAttn (needs nvcc and an sm_80/86/87/89/90 GPU)"
echo "        If this fails, everything above is still fine -- rerun just this step."
# --no-build-isolation so the extension compiles against the torch already installed
# here rather than a fresh one pip would fetch into an isolated build env; --no-deps so
# it does not pull its own pinned torch over yours.
#
# `uv pip` when uv is present, because a `uv venv` has no `pip` in it at all and the bare
# `pip` this used to call resolved to whatever was outside the venv, or to nothing.
if command -v uv >/dev/null 2>&1; then
  uv pip install --no-build-isolation --no-deps "$EXT/SpargeAttn"
else
  python -m pip install --no-build-isolation --no-deps "$EXT/SpargeAttn"
fi

echo
echo "[check] imports"
python - <<'PY'
from spas_sage_attn._qattn import (
    qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold as kernel,
)
from spas_sage_attn.utils import block_map_lut_triton, fill_block_map_triton
print("  spas_sage_attn OK:", kernel.__name__ if hasattr(kernel, "__name__") else kernel)

import ress._pathsetup  # noqa: F401  (puts the backbone trees on sys.path)
import vggt.models.vggt, pi3.models.pi3, depth_anything_3.api  # noqa: F401
print("  vggt / pi3 / depth_anything_3 OK")
PY
echo
echo "[done]"
