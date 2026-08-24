"""Value network v0: a deliberately small MLP over the 774-float encoding.

Small on purpose: the chance-collapse net is called at every interior chance
node of the search, so serving latency budgets the size. The cluster buys us
label quality and sweep breadth, not parameter count.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import FEATURE_DIM


class ValueMLP(nn.Module):
    """774 -> hidden -> hidden -> 1 logit; sigmoid gives mover win probability."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(features))
