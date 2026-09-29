"""Recover a pinhole intrinsic matrix from a predicted camera-space point map.

Pi3 predicts a point map and no intrinsics, while the rest of this code is depth + K +
extrinsics throughout. Rather than give Pi3 its own point-map-native path -- which would
make its numbers incomparable with the other backbones' -- its K is fitted here, once, and
everything downstream stays identical across backbones.
"""

import torch


def _weighted_line_fit(a, u, w, eps=1e-12):
    """Weighted least-squares fit of `u ~= f * a + c` over the last dim. Returns (f, c)."""
    s_w = w.sum(-1)
    s_a = (w * a).sum(-1)
    s_aa = (w * a * a).sum(-1)
    s_u = (w * u).sum(-1)
    s_au = (w * a * u).sum(-1)
    denom = s_w * s_aa - s_a * s_a
    f = (s_w * s_au - s_a * s_u) / denom.clamp(min=eps)
    c = (s_u - f * s_a) / s_w.clamp(min=eps)
    return f, c


def fit_pinhole_intrinsics(cam_points, conf=None, n_iters=10, eps=1e-6, min_z=1e-3):
    """cam_points: (N, H, W, 3) camera-space points. conf: (N, H, W) non-negative weights
    (optional). Returns (N, 3, 3) intrinsics, one per view, on `cam_points`'s device.

    Pixels with non-finite coordinates or `Z <= min_z` get zero weight (a point at or
    behind the camera has no pinhole projection).
    """
    N, H, W, _ = cam_points.shape
    device = cam_points.device
    pts = cam_points.reshape(N, H * W, 3).to(torch.float32)

    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = u_grid.reshape(1, -1).expand(N, -1)
    v = v_grid.reshape(1, -1).expand(N, -1)

    z = pts[..., 2]
    valid = torch.isfinite(pts).all(-1) & (z > min_z)
    a = torch.where(valid, pts[..., 0] / z.clamp(min=min_z), torch.zeros_like(z))
    b = torch.where(valid, pts[..., 1] / z.clamp(min=min_z), torch.zeros_like(z))

    w0 = torch.ones_like(z) if conf is None else conf.reshape(N, H * W).to(torch.float32).clamp(min=0)
    w0 = torch.where(valid, w0, torch.zeros_like(w0))

    w = w0
    for _ in range(n_iters):
        fx, cx = _weighted_line_fit(a, u, w)
        fy, cy = _weighted_line_fit(b, v, w)
        # Shared IRLS weight for both axes: a pixel is an outlier as a whole (bad depth,
        # sky) rather than per-axis, and a shared weight keeps fx/fy fits consistent.
        r = ((fx[:, None] * a + cx[:, None] - u).abs() + (fy[:, None] * b + cy[:, None] - v).abs())
        w = w0 / r.clamp(min=eps)

    K = torch.zeros(N, 3, 3, device=device, dtype=torch.float32)
    K[:, 0, 0], K[:, 0, 2] = fx, cx
    K[:, 1, 1], K[:, 1, 2] = fy, cy
    K[:, 2, 2] = 1.0
    return K


