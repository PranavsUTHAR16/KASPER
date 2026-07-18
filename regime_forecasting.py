"""
KASPER — KAN Layer 2: Regime-Adaptive Forecasting

PyTorch implementation of the forward pass described in Section 3.2 of
"KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable
Regimes" (TMLR, 02/2026).

Covers:
    - Regime-specific B-spline basis functions phi_j^(r)(Phi_t) (Eq. 21)
    - Per-regime forecast y_hat^(r)_t = sum_j w_j^(r) * phi_j^(r)  (Eq. 20)
    - Sparsity enforcement via soft-thresholding of weights        (Eq. 22)
    - Aggregation across regimes using KAN 1's soft probabilities: 
          y_hat_t = sum_r p_t^(r) * y_hat_t^(r)

Note on fidelity: the paper specifies the B-spline basis form (Eq. 21) and
the sparsity rule (Eq. 22) precisely, but does not state the exact spline
order or grid resolution used for KAN 2. This implementation uses cubic
B-splines (k=3) with an 8-basis grid per (regime, feature) pair, initialized
from percentile bounds the same way as KAN 1's spline activation — a
reasonable default consistent with the paper's KAN-based design, not a
verbatim reproduction of an unstated hyperparameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extend_grid(knots: torch.Tensor, k: int) -> torch.Tensor:
    """Extend an interior knot vector by k linearly-spaced points on each
    side, as required to define a full B-spline basis of order k."""
    h = (knots[-1] - knots[0]) / (len(knots) - 1)
    left = knots[0] - h * torch.arange(k, 0, -1, device=knots.device, dtype=knots.dtype)
    right = knots[-1] + h * torch.arange(1, k + 1, device=knots.device, dtype=knots.dtype)
    return torch.cat([left, knots, right])


def bspline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Cox-de Boor recursion, vectorized over a batch of scalar inputs.

    Args:
        x:    (batch,) input values, already clamped inside [grid[0], grid[-1]]
        grid: (n_knots,) extended knot vector
        k:    spline order (0 = box, 3 = cubic)

    Returns:
        (batch, n_basis) with n_basis = len(grid) - k - 1
    """
    x_col = x.unsqueeze(-1)  # (batch, 1)

    if k == 0:
        left = grid[:-1].unsqueeze(0)
        right = grid[1:].unsqueeze(0)
        return ((x_col >= left) & (x_col < right)).float()

    b_km1 = bspline_basis(x, grid, k - 1)  # (batch, n_knots - k)

    left_num = x_col - grid[:-k - 1].unsqueeze(0)
    left_den = (grid[k:-1] - grid[:-k - 1]).unsqueeze(0)
    right_num = grid[k + 1:].unsqueeze(0) - x_col
    right_den = (grid[k + 1:] - grid[1:-k]).unsqueeze(0)

    left_den = torch.where(left_den.abs() < 1e-8, torch.ones_like(left_den), left_den)
    right_den = torch.where(right_den.abs() < 1e-8, torch.ones_like(right_den), right_den)

    term1 = (left_num / left_den) * b_km1[:, :-1]
    term2 = (right_num / right_den) * b_km1[:, 1:]
    return term1 + term2


class RegimeFeatureSpline(nn.Module):
    """
    phi_j^(r)(Phi_t) = sum_k beta_{j,k}^(r) * B_k(Phi_t; xi^(r))   (Eq. 21)

    One learnable B-spline for a single (regime, feature) pair. Knots are
    fit from the empirical percentile range of the feature the first time
    the module sees data, mirroring KAN 1's robust initialization.
    """

    def __init__(self, n_basis: int = 8, k: int = 3,
                 p_min: float = 0.01, p_max: float = 0.99):
        super().__init__()
        self.k = k
        self.p_min = p_min
        self.p_max = p_max
        n_interior = n_basis - k + 1
        assert n_interior >= 2, "n_basis must be >= 2k"

        self.beta = nn.Parameter(torch.randn(n_basis) * 0.1)  # spline coefficients beta_{j,k}^(r)
        self.register_buffer("interior", torch.linspace(-1.0, 1.0, n_interior))
        self.register_buffer("grid", extend_grid(self.interior, k))
        self._fitted = False

    @torch.no_grad()
    def fit_knots(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1)
        x_min = torch.quantile(flat, self.p_min)
        x_max = torch.quantile(flat, self.p_max)
        if (x_max - x_min).abs() < 1e-6:
            x_max = x_min + 1e-6
        interior = torch.linspace(x_min.item(), x_max.item(), len(self.interior))
        self.interior.copy_(interior)
        self.grid.copy_(extend_grid(interior, self.k))
        self._fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            self.fit_knots(x)
        x_c = x.clamp(min=self.grid[0].item() + 1e-6, max=self.grid[-1].item() - 1e-6)
        basis = bspline_basis(x_c, self.grid, self.k)  # (batch, n_basis)
        return basis @ self.beta  # (batch,)


class RegimeAdaptiveForecastingLayer(nn.Module):
    """
    KAN Layer 2 (Section 3.2): regime-specific spline forecasts, sparsity,
    and aggregation via the regime probabilities produced by KAN 1.
    """

    def __init__(self, n_features: int, n_regimes: int = 3,
                 n_basis: int = 8, spline_order: int = 3):
        super().__init__()
        self.n_features = n_features
        self.n_regimes = n_regimes

        self.splines = nn.ModuleList([
            nn.ModuleList([
                RegimeFeatureSpline(n_basis=n_basis, k=spline_order)
                for _ in range(n_features)
            ])
            for _ in range(n_regimes)
        ])

        # w_j^(r): trainable per-feature, per-regime forecast weights (Eq. 20)
        self.weights = nn.Parameter(torch.randn(n_regimes, n_features) * 0.1)
        # theta^(r): regime-specific sparsity threshold, kept non-negative via softplus.
        self.theta_raw = nn.Parameter(torch.full((n_regimes,), -6.0))

        # Fidelity Note (Fig. 2 Caption): KAN-2 Attention-based Aggregation & Final Refinement Head
        # "produces regime-adaptive forecasts by updating a sparsity mask over spline components
        # and computing attention-based aggregation before the final refinement."
        self.attn_proj = nn.Linear(n_regimes, n_regimes)
        self.refinement_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def sparsity_threshold(self) -> torch.Tensor:
        return F.softplus(self.theta_raw)  # (n_regimes,)

    def effective_weights(self) -> torch.Tensor:
        """w_j^(r) <- ReLU(|w_j^(r)| - theta^(r)), sign preserved   (Eq. 22)"""
        theta = self.sparsity_threshold().unsqueeze(1)  # (n_regimes, 1)
        w = self.weights
        return torch.sign(w) * F.relu(w.abs() - theta)

    def forward(self, phi_t: torch.Tensor, p: torch.Tensor, z: torch.Tensor = None):
        """
        Args:
            phi_t: (batch, n_features) input feature matrix (same Phi_t as KAN 1).
            p:     (batch, n_regimes)  soft regime probabilities from KAN 1.
            z:     (batch, hidden_dim) optional pre-softmax embedding from KAN 1 for attention.

        Returns:
            y_hat:                 (batch,)                    final aggregated & refined forecast
            forecast_per_regime:   (batch, n_regimes)           y_hat^(r)_t per regime
            phi_per_regime:        (batch, n_regimes, n_features)  phi_j^(r)(Phi_t) values
        """
        batch = phi_t.shape[0]
        w_eff = self.effective_weights()  # (n_regimes, n_features)

        phi_per_regime = phi_t.new_zeros(batch, self.n_regimes, self.n_features)
        for r in range(self.n_regimes):
            for j in range(self.n_features):
                phi_per_regime[:, r, j] = self.splines[r][j](phi_t[:, j])

        forecast_per_regime = (phi_per_regime * w_eff.unsqueeze(0)).sum(-1)  # Eq. 20, (batch, n_regimes)
        
        # 1. Attention-based Aggregation (Fig. 2 Caption)
        attn_logits = self.attn_proj(forecast_per_regime)
        attn_weights = F.softmax(attn_logits + torch.log(p.clamp_min(1e-8)), dim=-1)
        y_agg = (forecast_per_regime * attn_weights).sum(-1)  # (batch,)

        # 2. Final Refinement Head with residual connection (Fig. 2 Caption)
        refinement = self.refinement_head(y_agg.unsqueeze(1)).squeeze(1)
        y_hat = y_agg + refinement  # (batch,)

        return y_hat, forecast_per_regime, phi_per_regime

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on the raw weights: lambda_s * sum |w_j^(r)|, part of the composite loss."""
        return self.weights.abs().sum()

    def orthogonality_loss(self) -> torch.Tensor:
        """L_orth = || W_norm W_norm^T - I_R ||_F^2   (Eq. 19)

        W = Layer-2 forecast weights w_j^(r) (Eq. 19 target), not a separate parameter.
        Forcing regime forecast-weight rows to be orthogonal means each regime relies on a
        distinct combination of input features, directly producing the per-regime feature
        differentiation seen in Fig. 4 of the paper.
        """
        w = self.weights                                          # (n_regimes, n_features)
        w_norm = F.normalize(w, p=2, dim=1)                      # unit-normalize each row
        gram = w_norm @ w_norm.T                                  # (n_regimes, n_regimes)
        eye = torch.eye(self.n_regimes, device=w.device, dtype=w.dtype)
        return torch.norm(gram - eye, p="fro") ** 2

    @torch.no_grad()
    def regime_feature_importance(self) -> torch.Tensor:
        """
        Quick weight-magnitude proxy for feature importance per regime,
        normalized to sum to 1 within each regime.
        """
        w_eff = self.effective_weights().abs()
        return w_eff / w_eff.sum(dim=1, keepdim=True).clamp_min(1e-8)


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, n_features, n_regimes = 32, 8, 3

    kan2 = RegimeAdaptiveForecastingLayer(n_features=n_features, n_regimes=n_regimes)

    phi_t = torch.randn(batch_size, n_features)
    p = F.softmax(torch.randn(batch_size, n_regimes), dim=-1)

    y_hat, forecast_per_regime, phi_per_regime = kan2(phi_t, p)

    print("y_hat:                 ", tuple(y_hat.shape))
    print("forecast_per_regime:   ", tuple(forecast_per_regime.shape))
    print("phi_per_regime:        ", tuple(phi_per_regime.shape))
    print("sparsity thresholds:   ", kan2.sparsity_threshold().tolist())
    print("sparsity loss (L1):    ", round(kan2.sparsity_loss().item(), 4))
    print("feature importance r0: ", [round(v, 3) for v in kan2.regime_feature_importance()[0].tolist()])

    # Sanity check: gradients reach the spline coefficients and weights
    target = torch.randn(batch_size)
    loss = F.huber_loss(y_hat, target) + 1e-3 * kan2.sparsity_loss()
    loss.backward()
    print("spline beta grad ok:   ", kan2.splines[0][0].beta.grad is not None)
    print("weight grad ok:        ", kan2.weights.grad is not None)
