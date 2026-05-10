from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attn_Net_Gated


class ABMIL(nn.Module):
    """Gated-attention ABMIL matching MCAT/SurvPath unimodal pathology baseline."""

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
        attn_logits, h = self.attention_net(h)
        attn = F.softmax(attn_logits, dim=0)
        bag = (attn * h).sum(dim=0)
        logits = self.classifier(bag)
        return logits, attn.squeeze(-1)
