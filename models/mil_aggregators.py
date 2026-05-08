from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Attention-based multiple instance learning aggregator."""

    def __init__(self, d_in: int, d_hidden: int = 256, n_classes: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = features.float()
        attn_logits = self.attention(features)
        attn = F.softmax(attn_logits, dim=0)
        bag = (attn * features).sum(dim=0)
        logits = self.classifier(bag)
        return logits, attn.squeeze(-1)


class TransMIL(ABMIL):
    """Compatibility wrapper; replace with full TransMIL for ablations when needed."""


class CLAM(ABMIL):
    """Compatibility wrapper; replace with full CLAM for ablations when needed."""


def build_aggregator(name: str, d_in: int, n_classes: int, d_hidden: int = 256) -> nn.Module:
    name = name.lower()
    if name == "abmil":
        return ABMIL(d_in=d_in, d_hidden=d_hidden, n_classes=n_classes)
    if name == "transmil":
        return TransMIL(d_in=d_in, d_hidden=d_hidden, n_classes=n_classes)
    if name == "clam":
        return CLAM(d_in=d_in, d_hidden=d_hidden, n_classes=n_classes)
    raise ValueError(f"Unsupported aggregator: {name}")

