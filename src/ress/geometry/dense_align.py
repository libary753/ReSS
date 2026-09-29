# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sim(3) alignment from dense per-pixel depth correspondences.

Calibration needs the predicted reconstruction placed in the ground-truth frame before a
loss can be taken against it. Fitting that similarity transform over the handful of camera
centres in a scene is fragile -- a few badly-estimated poses dominate the fit, and under
high sparsity there usually are a few. This fits it over per-pixel correspondences instead
(hundreds of thousands to hundreds of millions of points), masked by validity.

Two things about the implementation are load-bearing at scale:

- Unprojection is CHUNKED over frames. Batching all N frames into one [N, H, W, 3] tensor
  is fine at moderate resolution but not at DSLR resolution, where a single scene can need
  tens of GB.
- The fit is STREAMED (`_Sim3Accumulator`). A Umeyama/Kabsch similarity fit needs only the
  sufficient statistics n, sum(x), sum(y), sum(x y^T) and sum(|x|^2), all plain sums, so
  the correspondences never have to coexist in memory. Peak memory becomes independent of
  view count.

Accumulating in one pass means the covariance comes out as sum(x y^T)/n - mu_x mu_y^T
rather than from centred points, which cancels catastrophically when coordinates are large
relative to their spread. Each chunk is therefore shifted by a common origin before
accumulation, keeping the sums near zero; the float64 result then matches the centre-first
formula to ~1e-12.
"""

import numpy as np
import torch


def _unproject_torch(depths: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> torch.Tensor:
    """Back-project a batch of depth maps to world-space points, staying on GPU
    (torch tensor, not converted to numpy) -- `dense_correspondence_sim3` needs to
    mask/index and fit a Sim(3) transform on this output next, so keeping it as a
    CUDA tensor lets that follow-up work run on GPU too instead of round-tripping
    ~50M-point arrays through CPU numpy first (measured: the CPU/numpy
    masking + Umeyama fit was the dominant cost of `fuse3d`, ~11-12s of its ~16s,
    while this unprojection itself was already only ~2s on GPU).

    Args:
        depths: [N, H, W]
        intrinsics: [N, 3, 3]
        extrinsics: [N, 4, 4] or [N, 3, 4], world-to-camera

    Returns:
        [N, H, W, 3] world-space points, float64 CUDA tensor (upcast from a
        float32 compute, matching the precision the numpy callers
        always got -- see the old `.double()` cast this replaces).
    """
    n, h, w = depths.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    depths_t = torch.from_numpy(depths).to(device=device, dtype=torch.float32)
    intr_t = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32)
    ext_t = torch.from_numpy(extrinsics).to(device=device, dtype=torch.float32)
    if ext_t.shape[-2:] == (3, 4):
        pad = torch.zeros((n, 4, 4), device=device, dtype=torch.float32)
        pad[:, :3, :4] = ext_t
        pad[:, 3, 3] = 1.0
        ext_t = pad

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]

    inv_k = torch.inverse(intr_t)  # [N, 3, 3]
    rays = torch.einsum("nij,hwj->nhwi", inv_k, pix)  # [N, H, W, 3] (camera-space ray at depth=1)
    cam_pts = rays * depths_t[..., None]  # [N, H, W, 3]

    c2w = torch.inverse(ext_t)  # [N, 4, 4]
    rot = c2w[:, :3, :3]
    trans = c2w[:, :3, 3]
    world_pts = torch.einsum("nij,nhwj->nhwi", rot, cam_pts) + trans[:, None, None, :]
    return world_pts.double()




class _Sim3Accumulator:
    """Streaming sufficient statistics for a Umeyama/Kabsch similarity fit.

    `add()` may be called once per chunk of correspondences; nothing but a scalar,
    two 3-vectors and a 3x3 is retained between calls, so the caller never has to
    hold all the points at once (see this module's docstring). `solve()` then
    reproduces the numpy Umeyama/Kabsch fit step for step.

    All statistics are accumulated relative to `origin_from`/`origin_to` -- taken
    from the first chunk's means -- so the one-pass covariance identity stays
    numerically well-conditioned even for scene coordinates far from the origin.
    """

    def __init__(self):
        self.n = 0
        self._origin_from = None
        self._origin_to = None
        self.sum_x = None      # [3]   sum of (from - origin_from)
        self.sum_y = None      # [3]   sum of (to   - origin_to)
        self.sum_xy = None     # [3,3] sum of outer(from-origin, to-origin)
        self.sum_xx = None     # []    sum of |from - origin_from|^2

    def add(self, points_from: torch.Tensor, points_to: torch.Tensor) -> None:
        """Accumulate a chunk of [m, 3] correspondences (m may be 0)."""
        assert points_from.shape == points_to.shape and points_from.shape[-1] == 3
        if points_from.shape[0] == 0:
            return
        pf = points_from.double()
        pt = points_to.double()
        if self._origin_from is None:
            self._origin_from = pf.mean(dim=0)
            self._origin_to = pt.mean(dim=0)
            zeros = torch.zeros(3, device=pf.device, dtype=pf.dtype)
            self.sum_x = zeros.clone()
            self.sum_y = zeros.clone()
            self.sum_xy = torch.zeros(3, 3, device=pf.device, dtype=pf.dtype)
            self.sum_xx = torch.zeros((), device=pf.device, dtype=pf.dtype)

        x = pf - self._origin_from
        y = pt - self._origin_to
        self.n += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.sum_xy += x.T @ y
        self.sum_xx += (x * x).sum()

    def solve(self, allow_reflection: bool = False):
        """Returns (s: float, R: [3,3] float32 numpy, t: [3] float32 numpy) -- the
        same types the original numpy function returned."""
        if self.n < 3:
            raise ValueError("Need at least 3 non-collinear points.")

        mu_x_rel = self.sum_x / self.n
        mu_y_rel = self.sum_y / self.n

        # Identical to centring first: sum((x-mu_x)(y-mu_y)^T)/n and sum(|x-mu_x|^2)/n,
        # expanded so they can be built from per-chunk sums.
        h = self.sum_xy / self.n - torch.outer(mu_x_rel, mu_y_rel)
        sigma_x2 = self.sum_xx / self.n - (mu_x_rel * mu_x_rel).sum()

        u, s_vals, vt = torch.linalg.svd(h)
        v = vt.T

        r = v @ u.T

        if torch.linalg.det(r) < 0 and not allow_reflection:
            v = v.clone()
            v[:, -1] *= -1
            s_vals = s_vals.clone()
            s_vals[-1] *= -1
            r = v @ u.T

        s = s_vals.sum() / sigma_x2

        mu_x = self._origin_from + mu_x_rel
        mu_y = self._origin_to + mu_y_rel
        t = mu_y - s * (r @ mu_x)

        return float(s.item()), r.cpu().numpy().astype(np.float32), t.cpu().numpy().astype(np.float32)


def _find_similarity_transform_torch(points_from: torch.Tensor, points_to: torch.Tensor,
                                      allow_reflection: bool = False):
    """GPU/torch Umeyama/Kabsch similarity fit
    (Umeyama/Kabsch similarity fit) -- same algorithm, step for step, operating on
    CUDA tensors so the O(N) steps (centroid mean, centering, H=X^T@Y) run on GPU
    instead of single-threaded CPU numpy (measured: ~5s on CPU for
    N~52M points at DTU scale). Only the final scalar/3x3/3-vector result is moved
    to CPU. `points_from`/`points_to`: [N, 3] CUDA float64 tensors.

    Single-shot convenience wrapper over `_Sim3Accumulator` (which is what
    `dense_correspondence_sim3` drives chunk by chunk); kept for callers and tests
    that already hold both point sets in memory.

    Returns: (s: float, R: [3,3] float32 numpy, t: [3] float32 numpy) -- same
    types as the original numpy function.
    """
    assert points_from.shape == points_to.shape and points_from.shape[1] == 3
    if points_from.shape[0] < 3:
        raise ValueError("Need at least 3 non-collinear points.")

    acc = _Sim3Accumulator()
    acc.add(points_from, points_to)
    return acc.solve(allow_reflection=allow_reflection)


def _sim3_residual(points_from: torch.Tensor, points_to: torch.Tensor, s: float,
                   r: np.ndarray, t: np.ndarray) -> torch.Tensor:
    """Per-point distance ||s*R@from + t - to|| for a fitted Sim(3)."""
    r_t = torch.as_tensor(r, device=points_from.device, dtype=points_from.dtype)
    t_t = torch.as_tensor(t, device=points_from.device, dtype=points_from.dtype).reshape(1, 3)
    return (s * (points_from @ r_t.T) + t_t - points_to).norm(dim=-1)


def dense_correspondence_sim3(
    gt_depth: np.ndarray,
    gt_intrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
    pred_depth: np.ndarray,
    pred_intrinsics: np.ndarray,
    pred_extrinsics: np.ndarray,
    valid_mask: np.ndarray,
    chunk_size: int = 1,
    trim_rounds: int = 0,
    trim_quantile: float = 0.95,
):
    """Fit Sim(3) (s, R, t) mapping pred-frame points to GT-frame points from dense
    per-pixel depth correspondences (same pixel index across both -- matches HeSS's
    `Aligner.align_pcd`'s `align_type="depth"` path, not its pose-based `"cam_pos"`
    one).

    All depth/mask arrays share the same [N, H, W] pixel grid; intrinsics/extrinsics
    are each defined in their own (pred vs GT) frame at that same resolution.

    Args:
        chunk_size: Number of frames unprojected into a GPU batch at once (default
            1, see module docstring) -- only affects peak memory
            and speed, not the result. Raise it (e.g. for DTU-resolution datasets)
            to trade some of the safety margin back for fewer, larger kernel
            launches.
        trim_rounds: 0 (default) is the plain least-squares Umeyama fit every result in
            this project was measured with. >0 refits that many extra times, each time
            keeping only correspondences whose residual is below `trim_quantile` of the
            current residuals -- i.e. a trimmed (robust) Sim(3).

            Why it exists: the Umeyama scale is `sum(singular values) /
            variance(source)`, so a heavy tail in the PREDICTED cloud shrinks it even when
            almost every point is right. Measured on DTU scan1: DA3-GIANT's predicted depth
            matches GT to within 1% up to the 99th percentile, but its top 0.1% runs 2-4x
            too far, and that tail alone pulled the fitted scale 12% low (732 vs the 828
            implied by both the depth ratio and the camera baselines). Every point then
            came out uniformly 12% small, wrecking Chamfer while leaving pose -- aligned
            separately, and angular -- untouched. VGGT has no such tail (p99.9 = 1.016), so
            trimming barely moves it, which is what makes this safe to apply uniformly
            rather than as a per-backbone exception.

    Returns:
        (s: float, R: [3, 3] float32, t: [3] float32) such that
        `gt_point ~= s * R @ pred_point + t`.
    """
    n = gt_depth.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_flat_all = valid_mask.reshape(n, -1)

    def stream():
        """Yield (pred, gt) [m, 3] valid correspondences one chunk at a time.

        Re-unprojects on every pass rather than caching the concatenated clouds --
        that concatenation is exactly what made ETH3D-resolution scenes OOM (see the
        module docstring), and unprojection is a handful of GPU matmuls,
        cheap next to the eval that follows.
        """
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            gt_pts = _unproject_torch(gt_depth[start:end], gt_intrinsics[start:end],
                                       gt_extrinsics[start:end])
            pred_pts = _unproject_torch(pred_depth[start:end], pred_intrinsics[start:end],
                                         pred_extrinsics[start:end])
            mask_chunk = torch.from_numpy(mask_flat_all[start:end].reshape(-1)).to(device=device)
            yield pred_pts.reshape(-1, 3)[mask_chunk], gt_pts.reshape(-1, 3)[mask_chunk]

    acc = _Sim3Accumulator()
    for pred_pts, gt_pts in stream():
        acc.add(pred_pts, gt_pts)
    if acc.n < 3:
        raise ValueError(f"Not enough valid correspondences for Sim3 fit: {acc.n}")
    # Fixed across rounds: every round re-selects from the FULL correspondence set
    # using the latest fit (the trim is not cumulative), so the subsampling stride is
    # always derived from the full count.
    total = acc.n
    s, r, t = acc.solve(allow_reflection=False)

    for _ in range(trim_rounds):
        # `torch.quantile` refuses inputs beyond ~16M elements and a DTU scan has ~52M
        # correspondences, so the cutoff is read off a fixed-stride subsample instead --
        # deterministic (no RNG) and statistically indistinguishable at these counts.
        # `offset` walks the stride across chunk boundaries so the subsample is exactly
        # the one a global `cat(resid)[::stride]` would take.
        stride = max(1, total // 4_000_000)
        offset = 0
        probes = []
        for pred_pts, gt_pts in stream():
            resid = _sim3_residual(pred_pts, gt_pts, s, r, t)
            probes.append(resid[offset::stride].float())
            offset = (offset - resid.numel()) % stride
        cutoff = torch.quantile(torch.cat(probes), trim_quantile)

        acc = _Sim3Accumulator()
        for pred_pts, gt_pts in stream():
            keep = _sim3_residual(pred_pts, gt_pts, s, r, t) <= cutoff
            acc.add(pred_pts[keep], gt_pts[keep])
        if acc.n < 3:
            break
        s, r, t = acc.solve(allow_reflection=False)
    return s, r, t
