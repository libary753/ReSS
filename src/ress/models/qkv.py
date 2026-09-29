"""Shared Q/K/V projection + rope + patch/special-token split, factored out of
`sparse_vggt/models/attention.py::adaptive_sparse_attention_forward`'s opening lines
so both the runtime sparse path (`attention.py`) and the kernel-bypass exact path
(`reference_attention.py`) compute it identically instead of duplicating rope logic."""

from dataclasses import dataclass

import torch
from einops import rearrange

from ress.models.tokens import get_patch_tokens, get_special_tokens


@dataclass
class QKV:
    q_patch: torch.Tensor
    k_patch: torch.Tensor
    v_patch: torch.Tensor
    q_special: torch.Tensor | None
    k_special: torch.Tensor | None
    v_special: torch.Tensor | None
    shape: dict


def compute_patch_qkv(self, x, pos, num_special_tokens=5, num_heads=16, shape=None) -> QKV:
    """shape: optional {"N","H","W"} override for backbones whose global-attention
    positions cannot be read back as a patch grid. VGGT and Pi3 both hand their
    cross-frame blocks the real `+1`-shifted (y,x) grid, so `H`/`W` (and hence the
    per-frame token count) are recoverable from `pos` itself. DA3 does NOT: it passes
    `pos_nodiff` there (`vision_transformer._prepare_rope` -- an all-ones position for
    every patch token, deliberately removing spatial RoPE differences from cross-view
    attention), so inferring from `pos` yields H=W=1 and a nonsense token split
    (measured as a silently wrong sparsity before this override existed).
    `models.da3` therefore stashes the true grid on each attn module per forward."""
    B, NP, hidden_dim = x.shape
    S = num_special_tokens
    nh = num_heads
    hd = hidden_dim // nh

    if shape is not None:
        H, W, N = shape["H"], shape["W"], shape["N"]
    else:
        H = int(pos[0].max(0).values[0])
        W = int(pos[0].max(0).values[1])
        N = NP // (H * W + S)
    P = H * W + S
    assert N * P == NP, (
        f"token split mismatch: N={N} x P={P} (= H*W+S = {H}*{W}+{S}) != {NP} tokens"
    )

    qkv = self.qkv(x)
    qkv = rearrange(qkv, "B N (three nh hd)-> B N three nh hd", three=3, nh=nh, hd=hd)
    qkv = rearrange(qkv, "B N three nh hd -> three B nh N hd")
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    return QKV(
        q_patch=get_patch_tokens(q, N, P, S),
        k_patch=get_patch_tokens(k, N, P, S),
        v_patch=get_patch_tokens(v, N, P, S),
        q_special=get_special_tokens(q, N, P, S),
        k_special=get_special_tokens(k, N, P, S),
        v_special=get_special_tokens(v, N, P, S),
        shape={"B": B, "N": N, "P": P, "head_dim": hd, "num_heads": nh, "H": H, "W": W, "S": S},
    )


def combine_special_and_patch(x_special, x_patch, shape):
    """x_special: (B, nh, N*S, hd) or None; x_patch: (B, nh, N*(P-S), hd) (already in
    frame-major order). Returns (B, nh, N*P, hd)."""
    B, N, P, hd, nh, S = shape["B"], shape["N"], shape["P"], shape["head_dim"], shape["num_heads"], shape["S"]
    x_patch = rearrange(x_patch, "B nh (N HW) hd -> B nh N HW hd", N=N)
    if x_special is not None:
        x_special = x_special.view(B, nh, N, S, hd)
        x = torch.cat([x_special, x_patch], dim=-2)
    else:
        x = x_patch
    return x.view(B, nh, N * P, hd)


def project_out(self, x, B, N, P, nh, hd):
    x = x.transpose(1, 2).reshape(B, N * P, nh * hd)
    x = self.proj(x)
    return self.proj_drop(x)
