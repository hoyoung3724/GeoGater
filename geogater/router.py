"""GeoGater routing classifier and the distance-aware routing loss (Sec. 4.3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

ROUTE_ZS = 0  # zero-shot path
ROUTE_RET = 1  # retrieval path


class GeoGater(nn.Module):
    """Lightweight MLP router over the meta-feature vector F = [F_ZS, F_Ret].

    Outputs logits over {0: zero-shot path, 1: retrieval path}.
    With ``hidden_dim=1024`` and ``num_hidden_layers=2`` this is the
    three-layer MLP (~2.2M parameters) used in the paper.
    """

    def __init__(
        self,
        meta_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        meta_layers = [
            nn.Linear(meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_hidden_layers - 1):
            meta_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
        self.meta_net = nn.Sequential(*meta_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.head(self.meta_net(m))


class LogTargetCELoss(nn.Module):
    """Distance-aware routing loss L_dist (Eq. 10-11).

    Soft targets q_j = softmax(-log(1 + d_j) / tau) over the two paths, then
    cross-entropy against the router's predicted distribution.
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, logits, dist_targets):
        log_dists = torch.log1p(dist_targets)
        q = F.softmax(-log_dists / self.tau, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        return -(q * log_p).sum(dim=1)


class GeoLocLoss(nn.Module):
    """Routing objective L = lambda * L_dist + L_BCE (Sec. 4.3).

    ``mode`` lists the terms (``"ltce"`` = L_dist, ``"ce"`` = BCE on the oracle
    label) and ``loss_weights`` their weights; the paper uses
    ``mode=["ltce", "ce"], loss_weights=[5.0, 1.0]``.
    """

    def __init__(self, mode=("ltce", "ce"), tau: float = 1.0, loss_weights=None):
        super().__init__()
        self.modes = [mode] if isinstance(mode, str) else list(mode)
        unknown = set(self.modes) - {"ltce", "ce"}
        if unknown:
            raise ValueError(f"Unknown loss mode(s): {sorted(unknown)}")
        if loss_weights is not None and len(loss_weights) != len(self.modes):
            raise ValueError("loss_weights must match the number of loss modes.")
        self.loss_weights = loss_weights or [1.0] * len(self.modes)
        self.dist_loss = LogTargetCELoss(tau=tau)

    def forward(self, logits, dist_targets, class_targets):
        total = 0.0
        for mode, weight in zip(self.modes, self.loss_weights):
            if mode == "ltce":
                per_sample = self.dist_loss(logits, dist_targets)
            else:
                per_sample = F.cross_entropy(logits, class_targets, reduction="none")
            total = total + per_sample.mean() * weight
        return total
