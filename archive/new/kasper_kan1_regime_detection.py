"""
KASPER — KAN Layer 1: Regime Detection

PyTorch implementation of the forward pass described in Section 3.1 of
"KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable
Regimes" (TMLR, 02/2026).

Covers:
    - Hybrid spline activation f(x) = L(x) + C(x)          (Eq. 14)
    - Percentile-based knot initialization, robust to outliers (Eq. 15-16)
    - Feature embedding: (Linear -> BatchNorm -> GELU -> Dropout) x 2
    - Gumbel-Softmax differentiable regime classification   (Eq. 17)
    - Contrastive loss for intra-regime compactness         (Eq. 18)
    - Orthogonality regularization on regime weight vectors (Eq. 19)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineActivation(nn.Module):
    """
    Hybrid spline activation for a single scalar feature: f(x) = L(x) + C(x)

        L(x) = sum_m tanh(w_m) * [ReLU(x_norm - k_m) - ReLU(x_norm - k_{m+1})]
        C(x) = sum_m sigmoid(v_m) * x_norm^3

    Knots {k_m} sit on a uniform grid bounded by empirical percentiles of
    the feature (default 1st/99th), so the spline's coverage matches the
    feature's actual range instead of an arbitrary fixed interval.
    """

    def __init__(self, n_linear: int = 3, n_cubic: int = 2,
                 p_min: float = 0.01, p_max: float = 0.99):
        super().__init__()
        self.n_linear = n_linear
        self.n_cubic = n_cubic
        self.p_min = p_min
        self.p_max = p_max

        # Trainable per-segment weights: w_m for the linear part, v_m for the cubic part
        self.w = nn.Parameter(torch.randn(n_linear) * 0.1)
        self.v = nn.Parameter(torch.randn(n_cubic) * 0.1)

        # Knots + normalization bounds are data-derived, not learned directly
        self.register_buffer("knots", torch.linspace(-1.0, 1.0, n_linear + 1))
        self.register_buffer("x_min", torch.tensor(-1.0))
        self.register_buffer("x_max", torch.tensor(1.0))
        self._fitted = False

    @torch.no_grad()
    def fit_knots(self, x: torch.Tensor) -> None:
        """
        x_min = quantile(x, p_min), x_max = quantile(x, p_max)
        k_m   = x_min + (m - 1) * (x_max - x_min) / (G - 1)
        """
        flat = x.detach().reshape(-1)
        x_min = torch.quantile(flat, self.p_min)
        x_max = torch.quantile(flat, self.p_max)
        if (x_max - x_min).abs() < 1e-6:
            x_max = x_min + 1e-6
        self.x_min.copy_(x_min)
        self.x_max.copy_(x_max)
        self.knots.copy_(torch.linspace(x_min.item(), x_max.item(), self.n_linear + 1))
        self._fitted = True

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        span = (self.x_max - self.x_min).clamp_min(1e-6)
        return (x - self.x_min) / span * 2 - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            self.fit_knots(x)

        x_norm = self._normalize(x)

        # Linear (hinge) component L(x)
        lin_out = torch.zeros_like(x_norm)
        for m in range(self.n_linear):
            hinge = F.relu(x_norm - self.knots[m]) - F.relu(x_norm - self.knots[m + 1])
            lin_out = lin_out + torch.tanh(self.w[m]) * hinge

        # Cubic (sigmoid-gated) component C(x)
        x_cubed = x_norm ** 3
        cub_out = torch.zeros_like(x_norm)
        for m in range(self.n_cubic):
            cub_out = cub_out + torch.sigmoid(self.v[m]) * x_cubed

        return lin_out + cub_out


class FeatureSplineBlock(nn.Module):
    """Applies an independent SplineActivation to every input feature (column)."""

    def __init__(self, n_features: int, n_linear: int = 3, n_cubic: int = 2):
        super().__init__()
        self.splines = nn.ModuleList([
            SplineActivation(n_linear=n_linear, n_cubic=n_cubic)
            for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features), each column transformed independently
        outs = [self.splines[j](x[:, j]) for j in range(x.shape[1])]
        return torch.stack(outs, dim=1)


class RegimeDetectionLayer(nn.Module):
    """
    KAN Layer 1 (Section 3.1): spline activation -> embedding -> Gumbel-Softmax.

    Pipeline (matches Fig. 2):
        Phi_t --SplineActivation-->
              --stack features (Feature Embedding)-->
              --(Linear -> BatchNorm -> GELU -> Dropout) x 2-->  z_i
              --Linear--> regime logits
              --Gumbel-Softmax(tau)--> p_i  (soft regime probabilities)
    """

    def __init__(self, n_features: int, hidden_dim: int = 64,
                 n_regimes: int = 3, dropout: float = 0.1,
                 n_linear: int = 3, n_cubic: int = 2):
        super().__init__()
        self.n_regimes = n_regimes

        self.spline = FeatureSplineBlock(n_features, n_linear, n_cubic)

        def block(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.embed = nn.Sequential(
            block(n_features, hidden_dim),
            block(hidden_dim, hidden_dim),
        )
        self.to_logits = nn.Linear(hidden_dim, n_regimes)

        # Regime weight matrix W in R^{n_regimes x n_features}, used only
        # for the orthogonality regularizer (Eq. 19).
        self.regime_weights = nn.Parameter(torch.randn(n_regimes, n_features) * 0.1)

    def forward(self, phi_t: torch.Tensor, tau: float = 1.0, hard: bool = False):
        """
        Args:
            phi_t: (batch, n_features) input feature matrix.
            tau:   Gumbel-Softmax temperature (low tau -> near one-hot).
            hard:  if True, straight-through hard sampling.

        Returns:
            z:      (batch, hidden_dim)  pre-softmax embedding, used by the contrastive loss.
            p:      (batch, n_regimes)   soft regime probabilities, consumed by KAN 2.
            logits: (batch, n_regimes)   raw regime logits, for diagnostics.
        """
        spline_feats = self.spline(phi_t)                          # Eq. 14-16
        z = self.embed(spline_feats)                                # (L->BN->G->D) x 2
        logits = self.to_logits(z)                                  # f_r(Phi_t)
        p = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)     # Eq. 17
        return z, p, logits

    def orthogonality_loss(self) -> torch.Tensor:
        """L_orth = || W W^T - I_R ||_F^2   (Eq. 19)"""
        w = self.regime_weights
        gram = w @ w.T
        eye = torch.eye(self.n_regimes, device=w.device, dtype=w.dtype)
        return torch.norm(gram - eye, p="fro") ** 2


def contrastive_loss(z: torch.Tensor, regime_ids: torch.Tensor) -> torch.Tensor:
    """
    L_contrastive = E[ ||z_i - z_j||^2 * y_ij ]   (Eq. 18)

    y_ij = 1 if samples i and j are assigned to the same regime, else 0.
    Averaged over all off-diagonal pairs in the batch.
    """
    diff = z.unsqueeze(1) - z.unsqueeze(0)                # (B, B, hidden_dim)
    sq_dist = (diff ** 2).sum(-1)                          # (B, B)
    same_regime = (regime_ids.unsqueeze(1) == regime_ids.unsqueeze(0)).float()
    mask = 1.0 - torch.eye(z.shape[0], device=z.device)    # drop self-pairs
    return (sq_dist * same_regime * mask).sum() / mask.sum().clamp_min(1.0)


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, n_features, hidden_dim, n_regimes = 32, 8, 64, 3

    model = RegimeDetectionLayer(
        n_features=n_features, hidden_dim=hidden_dim, n_regimes=n_regimes
    )

    # Stand-in for a batch of standardized features (output of Sec. 4.1's
    # SelectKBest + StandardScaler step: HL_spread, OC_spread, ATR, etc.)
    phi_t = torch.randn(batch_size, n_features)

    z, p, logits = model(phi_t, tau=1.0)
    regime_ids = p.argmax(dim=-1)

    l_contrastive = contrastive_loss(z, regime_ids)
    l_orth = model.orthogonality_loss()

    print("embedding z:        ", tuple(z.shape))
    print("regime probs p:     ", tuple(p.shape), "rows sum to", round(p.sum(-1)[0].item(), 4))
    print("regime logits:      ", tuple(logits.shape))
    print("regime assignment:  ", regime_ids[:10].tolist())
    print("contrastive loss:   ", round(l_contrastive.item(), 4))
    print("orthogonality loss: ", round(l_orth.item(), 4))

    # Sanity check: gradients flow back through the spline knots' weights
    loss = l_contrastive + 0.01 * l_orth
    loss.backward()
    print("spline weight grad ok:", model.spline.splines[0].w.grad is not None)
