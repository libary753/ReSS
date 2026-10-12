# ReSS & HeSS

Official implementation of two papers, released together — ReSS builds on HeSS and they
share a kernel, so `apply_hess` and `apply_ress` are two calls against the same model.

- **ReSS: Residual-Restoring Sparse Attention for 3D Vision Transformers**
  *Yongsung Kim, Jaehoon Lee, Minjun Park, Wooseok Song, Hun Hwangbo, Sungroh Yoon*
  — [arXiv:2609.35593](https://arxiv.org/abs/2609.35593)
- **HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT** (CVPR 2026)
  *Yongsung Kim, Wooseok Song, Jaihyun Lew, Hun Hwangbo, Jaehoon Lee, Sungroh Yoon*
  — [arXiv:2603.25336](https://arxiv.org/abs/2603.25336)

Block-sparse global attention for feed-forward 3D reconstruction (VGGT, π³,
Depth-Anything-3). HeSS decides **how many** key blocks each head keeps, from an offline
Fisher-trace calibration; ReSS decides **which** ones, by the residual drift their removal
would cause. Inference-time only, no retraining.

---

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
bash scripts/setup_external.sh      # fetches the backbones, builds SpargeAttn
```

Install the CUDA-matched PyTorch first — the index URL above is for CUDA 12.8; pick the
one for your driver. Python 3.10-3.12. [uv](https://docs.astral.sh/uv/) is what the
commands above use; plain `pip` in a `python -m venv` works the same, and
`setup_external.sh` takes either.

Two ways the build fails that are worth knowing in advance:

- **`CUDA_HOME` must match your torch build** in major.minor. A mismatch compiles and
  imports cleanly, then fails at the first kernel launch.
- **Compute capability must be 8.0, 8.6, 8.7, 8.9 or 9.0.** SpargeAttn's `setup.py`
  rejects anything else, Blackwell (sm_120) included.

`--skip-build` fetches and patches without compiling, for a machine with no GPU.

`setup_external.sh` pins each third-party repo to the commit these results were measured
with: vggt `44b3afb`, Pi3 `42e5ec8`, depth-anything-3 `3d835ec`, SpargeAttn `7cb07d1`. The
**SpargeAttn patch is required, not cosmetic**: it makes `attention_sink` a count of
leading tokens rather than a flag, and aligns the block-quantisation scale between
`quant_per_block` and `get_block_map_meansim_fuse_quant`. Without it the kernel runs and
selects different blocks. (vggt takes a one-line `torch.amp.autocast` fix; the other two
are unpatched.)

---

## Usage

```python
import torch
import ress

model = ress.load_backbone("vggt")           # or "pi3", "da3"
ress.apply_ress(model, sparse_ratio=0.1)
with torch.no_grad():
    predictions = model(images)
```

`sparse_ratio` is the fraction of key blocks dropped. Each `apply_*` fully resets the
previous mode, so one loaded model moves between them between forward passes — no reload,
no second copy of the weights.

| call | budget | block order | |
|---|---|---|---|
| `apply_dense(model)` | — | — | unmodified backbone |
| `apply_sparse_vggt(model, sparse_ratio, cdf_threshold)` | uniform across heads | attention probability | SparseVGGT, third-party baseline |
| `apply_hess(model, sparse_ratio, cdf_threshold)` | **calibrated per head** | attention probability | HeSS |
| `apply_ress(model, sparse_ratio)` | calibrated per head | **residual drift, restore-swap** | ReSS |

Each row changes exactly one thing from the row above, so reading down is the ablation.
`cdf_threshold` is SparseVGGT's own knob and is kept on the two modes that inherit its
block order; the stricter of it and `sparse_ratio` wins. `apply_ress` takes `n_iters`
(default 5, what the paper reports).

Nothing here requires ReSS — `apply_hess` reproduces the HeSS paper on its own; `ress` is
the package name, not a statement about which method you are running.

To wire ReSS into a backbone you construct yourself, load it through the matching module
(`from ress.models.vggt import load_vggt`) rather than `load_backbone`.

**Sparsity is a request, not a guarantee.** `sparse_ratio` sets the budget; the realized
drop fraction depends on content. Load with `aux_output=True` and read it back from
`model.aux_output_store`. The grid is block-sparse, so a couple of frames has little to
gain — the margin widens with view count.

---

## Calibration

The Fisher-trace head-importance tables ship with the package
(`src/ress/calibration/`, six files, ~35 KB), so **nothing above needs a calibration run**.
They are `(3, n_blocks, n_heads)` tensors — the leading axis is q/k/v and HeSS reads W_Q.
Each backbone has a point-cloud table and a camera table, mixed by HeSS Eq. (9) with a
per-backbone λ (VGGT 0.5, π³ 0.0, DA3 1.0; `ress/backbones.py` records where each
comes from).

To regenerate, you need CO3Dv2 and a local pytorch3d:

```bash
uv pip install -e ".[calibrate]"
# CPU-only build is enough -- pytorch3d is here for its CO3Dv2 loader.
FORCE_CUDA=0 uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d@v0.7.8"

CO3DV2_ROOT=/path/to/co3dv2 python -m scripts.calibrate_fisher_scores \
    --backbone vggt --loss pcd
```

The view draw is seeded by the caller (`--view-seed`), not by the loader, and different
seeds give measurably different tables — in roughly one layer in four the most important
head changes. Regenerate every table you intend to compare together, and keep the seed.

---

## Attribution and licence

The block-sparse scaffolding started from **SparseVGGT**
([paper](https://arxiv.org/abs/2509.07120) · [code](https://github.com/brianwang00001/sparse-vggt)),
which is also the uniform-budget baseline both papers measure against. How much survives
differs by file:

| file | relation to SparseVGGT |
|---|---|
| `models/tokens.py`, `models/hilbert.py` | taken essentially unchanged |
| `models/vggt.py` | their aggregator forward, our wiring around it |
| `models/attention.py` | their block-map/kernel-call skeleton; scoring, budget hooks and restore-swap are not |
| `models/pi3.py`, `models/qkv.py`, `budget/_proportional.py` | largely independent |
| `scoring/`, `selection/`, `budget/hess.py`, `runtime/` | new |

This repository is **CC BY-NC 4.0** (`LICENSE`), with two carve-outs:

- `src/ress/geometry/dense_align.py` derives from Depth-Anything-3 and keeps its
  Apache-2.0 notice.
- **SparseVGGT's repository carries no licence.** An unlicensed work is "all rights
  reserved" by default, so this repository cannot grant you rights to the files marked
  above as taken from it, and the licence above does not cure that. For clean terms on
  those parts, ask the SparseVGGT authors.

`external/` is fetched, not vendored, and each tree keeps its own licence: SpargeAttn and
Depth-Anything-3 Apache-2.0, π³ BSD-style non-commercial, VGGT under Meta's own licence,
which carries its terms to anyone you pass it on to.

## Citation

ReSS's head budget is HeSS's, so work building on ReSS rests on both.

```bibtex
@article{kim2026ress,
  title   = {ReSS: Residual-Restoring Sparse Attention for 3D Vision Transformers},
  author  = {Kim, Yongsung and Lee, Jaehoon and Park, Minjun and Song, Wooseok
             and Hwangbo, Hun and Yoon, Sungroh},
  journal = {arXiv preprint arXiv:2609.35593},
  year    = {2026}
}

@InProceedings{kim2026hess,
    author    = {Kim, Yongsung and Song, Wooseok and Lew, Jaihyun and Hwangbo, Hun and Lee, Jaehoon and Yoon, Sungroh},
    title     = {HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {36509-36517}
}
```
