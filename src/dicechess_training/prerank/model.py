"""The student: nine raw features in, one unbounded score out.

Two things about this model are contract, not taste.

**It takes raw features.** The standardisation statistics live inside it as buffers and are
applied as a constant affine transform before the first layer, so they travel with the checkpoint
and with the exported graph. Whatever serves the model sends the engine's features exactly as the
engine produced them, and nothing has to reimplement a scaler at the seam. This is the rule ADR
0001 arrived at for the evaluator and it applies here for the same reason.

**Its output has no activation.** `contracts.prerank` says why at length: an ordering is invariant
to monotone transforms, so there is nothing to calibrate, and a sigmoid would compress precisely
the distinctions a ranker exists to make. The contract refuses a `calibration` block on an
artifact of this role, and a model that needed one would be a value model wearing the wrong name.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PreRankMLP(nn.Module):
    """A small MLP over one candidate's features, scoring it against its own group.

    Narrow on purpose. This runs on *every* legal turn before the search consults its deadline —
    up to 2,420 of them at one root in the first corpus — which is the same cost argument that
    chose a nine-column schema over `kcp-13`. Nine inputs do not need the evaluator's 64x64.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        feature_mean: np.ndarray | None = None,
        feature_scale: np.ndarray | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.ReLU())
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        if (feature_mean is None) != (feature_scale is None):
            raise ValueError("feature_mean and feature_scale must be given together")
        self.standardised = feature_mean is not None
        if self.standardised:
            mean = np.asarray(feature_mean, dtype=np.float32).reshape(-1)
            scale = np.asarray(feature_scale, dtype=np.float32).reshape(-1)
            if mean.shape != (input_dim,) or scale.shape != (input_dim,):
                raise ValueError(f"standardisation statistics must have shape ({input_dim},)")
            if not np.all(scale > 0):
                raise ValueError("every feature scale must be positive")
            self.register_buffer("feature_mean", torch.from_numpy(mean.copy()))
            self.register_buffer("feature_scale", torch.from_numpy(scale.copy()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.standardised:
            features = (features - self.feature_mean) / self.feature_scale
        return self.net(features)


def listwise_loss(
    scores: torch.Tensor,
    gains: torch.Tensor,
    group_index: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Cross-entropy between the target distribution and the model's, one term per group.

    `-sum_i P_i log softmax(s)_i`, where the softmax runs over each group separately. Candidates
    of many groups arrive concatenated with `group_index` saying which group each row belongs to,
    because the groups in one batch differ in length by three orders of magnitude and padding them
    to the longest would spend most of the batch on nothing.

    The maximum is subtracted per group before exponentiating, which is the ordinary guard against
    overflow and matters more than usual here: scores are unbounded by design.

    Minimised when the model's distribution equals the target's — so on a group the teacher could
    not separate, the model is pulled towards scoring its candidates equally rather than towards
    an arbitrary order. Returned as a mean over groups, so a root with 2,420 candidates counts as
    one decision and not as 2,420.
    """
    scores = scores.reshape(-1)
    largest = torch.full((groups,), float("-inf"), dtype=scores.dtype, device=scores.device)
    largest = largest.scatter_reduce(0, group_index, scores, reduce="amax", include_self=False)
    shifted = scores - largest[group_index]

    denominator = torch.zeros(groups, dtype=scores.dtype, device=scores.device)
    denominator = denominator.index_add(0, group_index, shifted.exp())
    log_probability = shifted - denominator.log()[group_index]

    per_group = torch.zeros(groups, dtype=scores.dtype, device=scores.device)
    per_group = per_group.index_add(0, group_index, -gains * log_probability)
    return per_group.mean()
