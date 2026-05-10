from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from nystrom_attention import NystromAttention as _NystromAttention
except Exception:  # pragma: no cover - optional dependency fallback
    _NystromAttention = None


class TransLayer(nn.Module):
    """Pre-norm self-attention block from TransMIL."""

    def __init__(self, dim: int = 512, heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.use_nystrom = _NystromAttention is not None
        if self.use_nystrom:
            self.attn = _NystromAttention(
                dim=dim,
                dim_head=dim // heads,
                heads=heads,
                num_landmarks=dim // 2,
                pinv_iterations=6,
                residual=True,
                dropout=0.1,
            )
        else:
            self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        if self.use_nystrom:
            return x + self.attn(normed)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        return x + attended


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator from TransMIL."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    """TransMIL aggregator (Shao et al., NeurIPS 2021)."""

    def __init__(self, d_in: int, n_classes: int = 4, d_hidden: int = 512, heads: int = 8) -> None:
        super().__init__()
        self._fc1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hidden))
        self.layer1 = TransLayer(dim=d_hidden, heads=heads)
        self.layer2 = TransLayer(dim=d_hidden, heads=heads)
        self.pos_layer = PPEG(dim=d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self._fc2 = nn.Linear(d_hidden, n_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() != 2:
            raise ValueError(f'TransMIL expects features [N, d_in], got {tuple(features.shape)}')
        h = self._fc1(features.float()).unsqueeze(0)
        n = h.shape[1]
        side = int(math.ceil(math.sqrt(max(n, 1))))
        pad = side * side - n
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)
        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1).to(h.dtype).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        h = self.norm(h)
        logits = self._fc2(h[:, 0]).squeeze(0)
        with torch.no_grad():
            cls = h[:, 0:1]
            patch_h = h[:, 1: n + 1]
            sim = F.cosine_similarity(cls.expand_as(patch_h), patch_h, dim=-1).squeeze(0)
            attn = F.softmax(sim, dim=0)
        return logits, attn
