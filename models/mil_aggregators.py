"""MIL aggregators for TDAE.

ABMIL is the CLAM-style ``Attn_Net_Gated`` head that MCAT and SurvPath both use
as their unimodal pathology baseline. TransMIL is ported from
https://github.com/szc19990412/TransMIL and requires ``nystrom-attention``.
CLAM-SB is ported from https://github.com/mahmoodlab/CLAM (single-branch
variant; the bag-level classifier and attention head are kept, instance-level
clustering is omitted because TDAE feeds pre-aggregated features).

All aggregators expose a uniform forward signature so ``models/tdae.py`` can
swap between them without changes::

    logits, attn = aggregator(features)  # features: [N, d_in]
                                          # logits:   [n_classes]
                                          # attn:     [N]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:  # optional dependency, only required for TransMIL
    from nystrom_attention import NystromAttention as _NystromAttention
except Exception:  # pragma: no cover - fallback handled at TransMIL build time
    _NystromAttention = None


# ---------------------------------------------------------------------------
# CLAM-style gated attention head (used by ABMIL and CLAM)
# ---------------------------------------------------------------------------


class Attn_Net_Gated(nn.Module):
    """Gated attention from CLAM (Lu et al., Nat. BME 2021).

    Direct port of ``Attn_Net_Gated`` in mahmoodlab/CLAM; output ``A`` has
    shape ``[N, n_classes]`` so the same head can serve single- and
    multi-branch variants.
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, n_classes: int = 1) -> None:
        super().__init__()
        layers_a: list[nn.Module] = [nn.Linear(L, D), nn.Tanh()]
        layers_b: list[nn.Module] = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout > 0:
            layers_a.append(nn.Dropout(dropout))
            layers_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*layers_a)
        self.attention_b = nn.Sequential(*layers_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # [N, n_classes]
        return A, x


# ---------------------------------------------------------------------------
# ABMIL — CLAM-style single-branch bag classifier with gated attention
# ---------------------------------------------------------------------------


class ABMIL(nn.Module):
    """Gated-attention ABMIL matching MCAT/SurvPath unimodal pathology baseline.

    Pipeline mirrors CLAM-SB without instance-level clustering:
    ``Linear(d_in, d_hidden) -> ReLU -> Dropout -> Attn_Net_Gated -> Linear``.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 512,
        d_attn: int = 256,
        n_classes: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_net = Attn_Net_Gated(L=d_hidden, D=d_attn, dropout=dropout, n_classes=1)
        self.classifier = nn.Linear(d_hidden, n_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() != 2:
            raise ValueError(f'ABMIL expects features [N, d_in], got {tuple(features.shape)}')
        h = self.fc(features.float())
        attn_logits, h = self.attention_net(h)        # attn_logits: [N, 1]
        attn = F.softmax(attn_logits, dim=0)          # softmax over patches
        bag = (attn * h).sum(dim=0)                   # [d_hidden]
        logits = self.classifier(bag)                 # [n_classes]
        return logits, attn.squeeze(-1)


# ---------------------------------------------------------------------------
# TransMIL — Shao et al., NeurIPS 2021 (port of szc19990412/TransMIL)
# ---------------------------------------------------------------------------


class TransLayer(nn.Module):
    """Pre-norm Nystrom self-attention block from TransMIL."""

    def __init__(self, dim: int = 512, heads: int = 8) -> None:
        super().__init__()
        if _NystromAttention is None:
            raise ImportError(
                "TransMIL requires the 'nystrom-attention' package. "
                "Install it with `pip install nystrom-attention`."
            )
        self.norm = nn.LayerNorm(dim)
        self.attn = _NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


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
    """TransMIL aggregator (Shao et al., NeurIPS 2021).

    Faithful port of szc19990412/TransMIL with two TransLayer blocks and a
    PPEG between them. CLS token is used for the final hazard logits.
    """

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
        h = self._fc1(features.float()).unsqueeze(0)  # [1, N, d_hidden]
        n = h.shape[1]
        side = int(math.ceil(math.sqrt(max(n, 1))))
        pad = side * side - n
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)
        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1).to(h.dtype).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)         # [1, N+1, d_hidden]
        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        h = self.norm(h)
        logits = self._fc2(h[:, 0]).squeeze(0)        # [n_classes]
        # Surrogate per-patch attention from CLS-token similarity (kept for
        # logging only; TransMIL itself doesn't expose per-patch weights).
        with torch.no_grad():
            cls = h[:, 0:1]
            patch_h = h[:, 1: n + 1]
            sim = F.cosine_similarity(cls.expand_as(patch_h), patch_h, dim=-1).squeeze(0)
            attn = F.softmax(sim, dim=0)
        return logits, attn


# ---------------------------------------------------------------------------
# CLAM-SB — Lu et al., Nat. BME 2021 (port of mahmoodlab/CLAM, single-branch)
# ---------------------------------------------------------------------------


class CLAM(nn.Module):
    """CLAM-SB bag classifier.

    Port of ``CLAM_SB`` from mahmoodlab/CLAM. The bag-level path is kept
    verbatim (gated attention -> single classifier). Instance-level clustering
    losses are intentionally omitted because TDAE supplies graph-propagated
    features; clustering against a hard pseudo-label inside the bag head would
    fight the upstream propagator.
    """

    SIZES = {
        'small': (512, 256),
        'big': (512, 384),
    }

    def __init__(
        self,
        d_in: int,
        n_classes: int = 4,
        size: str = 'small',
        dropout: float = 0.25,
        gate: bool = True,
    ) -> None:
        super().__init__()
        d_hidden, d_attn = self.SIZES[size]
        fc_layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc_layers)
        if gate:
            self.attention_net = Attn_Net_Gated(L=d_hidden, D=d_attn, dropout=dropout, n_classes=1)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(d_hidden, d_attn), nn.Tanh(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(d_attn, 1),
            )
        self.classifier = nn.Linear(d_hidden, n_classes)
        self._gated = gate

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() != 2:
            raise ValueError(f'CLAM expects features [N, d_in], got {tuple(features.shape)}')
        h = self.fc(features.float())
        if self._gated:
            attn_logits, h = self.attention_net(h)    # [N, 1]
        else:
            attn_logits = self.attention_net(h)       # [N, 1]
        attn = F.softmax(attn_logits, dim=0)
        bag = (attn * h).sum(dim=0)
        logits = self.classifier(bag)
        return logits, attn.squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_aggregator(name: str, d_in: int, n_classes: int, d_hidden: int = 512) -> nn.Module:
    name = name.lower()
    if name == 'abmil':
        return ABMIL(d_in=d_in, d_hidden=d_hidden, n_classes=n_classes)
    if name == 'transmil':
        return TransMIL(d_in=d_in, n_classes=n_classes, d_hidden=d_hidden)
    if name == 'clam' or name == 'clam_sb':
        return CLAM(d_in=d_in, n_classes=n_classes)
    raise ValueError(f'Unsupported aggregator: {name}')


__all__ = [
    'ABMIL', 'TransMIL', 'CLAM', 'Attn_Net_Gated', 'TransLayer', 'PPEG',
    'build_aggregator',
]
