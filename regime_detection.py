"""
KASPER — KAN Layer 1: Regime Detection

PyTorch implementation of the forward pass described in Section 3.1 of
"KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable
Regimes" (TMLR, 02/2026).

Covers:
    - Hybrid spline activation f(x) = L(x) + C(x)          (Eq. 14)
    - Percentile-based knot initialization, robust to outliers (Eq. 15-16)
    - Per-feature embedding: SplineActivation → Linear(1, d) per feature
    - Stack embeddings: concatenate to (batch, n_features * feature_embed_dim)
    - MLP block: (Linear → BatchNorm → GELU → Dropout) x 2 → Linear
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


class FeatureEmbeddingBlock(nn.Module):
    """
    Per-feature embedding pipeline: SplineActivation → Linear(1, feature_embed_dim).

    Each scalar feature is first transformed by its own SplineActivation (which
    handles quantile normalization and the hybrid linear-cubic basis), and then
    projected into a ``feature_embed_dim``-dimensional vector by an independent
    linear layer. The resulting per-feature vectors are concatenated into a flat
    tensor of shape ``(batch, n_features * feature_embed_dim)``.
    """

    def __init__(self, n_features: int, feature_embed_dim: int = 4,
                 n_linear: int = 3, n_cubic: int = 2):
        super().__init__()
        self.n_features = n_features
        self.feature_embed_dim = feature_embed_dim
        self.splines = nn.ModuleList([
            SplineActivation(n_linear=n_linear, n_cubic=n_cubic)
            for _ in range(n_features)
        ])
        # Independent linear projection per feature: scalar → d-dim vector
        self.projectors = nn.ModuleList([
            nn.Linear(1, feature_embed_dim)
            for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features) raw input features.

        Returns:
            (batch, n_features * feature_embed_dim) concatenated per-feature embeddings.
        """
        embedded = []
        for j in range(self.n_features):
            # (batch,) -> spline -> (batch,) -> unsqueeze -> (batch, 1) -> Linear -> (batch, d)
            s = self.splines[j](x[:, j])           # (batch,)
            e = self.projectors[j](s.unsqueeze(1)) # (batch, feature_embed_dim)
            embedded.append(e)
        # Concatenate all per-feature embeddings: (batch, n_features * feature_embed_dim)
        return torch.cat(embedded, dim=1)


class RegimeDetectionLayer(nn.Module):
    """
    KAN Layer 1 (Section 3.1): per-feature embedding -> MLP -> Gumbel-Softmax.

    Pipeline:
        phi_t
          │
          ▼
        SplineActivation (per feature, with quantile normalization)
          │
          ▼
        Feature Embedding (nn.Linear(1, feature_embed_dim) per feature)
          │
          ▼
        Stack / Concatenate  →  (batch, n_features * feature_embed_dim)
          │
          ▼
        MLP Block: (Linear → BatchNorm → GELU → Dropout) × 2 → Linear
          │
          ▼  z  (batch, hidden_dim)
          │
          ▼
        Gumbel-Softmax(τ)  →  p  (batch, n_regimes)
    """

    def __init__(self, n_features: int, hidden_dim: int = 64,
                 n_regimes: int = 3, dropout: float = 0.1,
                 n_linear: int = 3, n_cubic: int = 2,
                 feature_embed_dim: int = 4):
        super().__init__()
        self.n_regimes = n_regimes
        self.feature_embed_dim = feature_embed_dim

        # Per-feature: SplineActivation + independent Linear(1 → feature_embed_dim)
        self.spline = FeatureEmbeddingBlock(
            n_features=n_features,
            feature_embed_dim=feature_embed_dim,
            n_linear=n_linear,
            n_cubic=n_cubic,
        )

        # MLP input is the concatenated per-feature embeddings
        mlp_input_dim = n_features * feature_embed_dim

        def block(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.embed = nn.Sequential(
            block(mlp_input_dim, hidden_dim),
            block(hidden_dim, hidden_dim),
        )
        self.to_logits = nn.Linear(hidden_dim, n_regimes)

    def forward(self, phi_t: torch.Tensor, tau: float = 1.0, hard: bool = False, deterministic: bool = False):
        """
        Args:
            phi_t:         (batch, n_features) input feature matrix.
            tau:           Gumbel-Softmax / Softmax temperature (low tau -> near one-hot).
            hard:          if True, straight-through hard sampling.
            deterministic: if True, use deterministic F.softmax(logits / tau) without Gumbel noise (inference mode).

        Returns:
            z:      (batch, hidden_dim)  embedding used by the contrastive loss.
            p:      (batch, n_regimes)   soft/deterministic regime probabilities, consumed by KAN 2.
            logits: (batch, n_regimes)   raw regime logits, for diagnostics.
        """
        # Step 1: Spline + per-feature projection → concatenated embedding
        # (batch, n_features) → (batch, n_features * feature_embed_dim)
        embedded_feats = self.spline(phi_t)                         # Eq. 14-16 + per-feature Linear

        # Step 2: Global MLP across all stacked feature embeddings
        z = self.embed(embedded_feats)                              # (batch, hidden_dim)

        # Step 3: Route to regime logits → Gumbel-Softmax or Deterministic Softmax
        logits = self.to_logits(z)                                  # f_r(Phi_t)
        if deterministic:
            p = F.softmax(logits / tau, dim=-1)
        else:
            # Fidelity Note (Eq. 17): Paper writes exp(f_r(Phi_t + g_r)/tau) with noise inside spline input.
            # Standard F.gumbel_softmax is used here on logits for numerical stability and gradient reliability during training.
            p = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)    # Eq. 17
        return z, p, logits

    # NOTE: orthogonality_loss (Eq. 19) is a method of RegimeAdaptiveForecastingLayer.
    # Per the paper, W ∈ R^{R×F} where w^(r)_j is the forecast weight for feature j in regime r
    # — that is self.weights in Layer 2, not the routing logits here.


def contrastive_loss(z: torch.Tensor, target: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Contrastive loss with both attractive and repulsive terms (Eq. 18).

    Attractive: penalize large distance for same-regime pairs → compacts clusters.
    Repulsive:  penalize distance < margin for diff-regime pairs → pushes clusters apart.

    Args:
        z:      (batch, hidden_dim) pre-softmax embeddings from KAN 1.
        target: (batch, n_regimes) soft regime probabilities p OR (batch,) hard regime IDs.
        margin: minimum distance required between different-regime pairs.
    """
    diff = z.unsqueeze(1) - z.unsqueeze(0)          # (B, B, D)
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)   # (B, B) Euclidean distance

    if target.dim() == 2:
        # Soft pairwise regime similarity y_ij = p_i @ p_j^T in [0, 1] (differentiable wrt p)
        same_regime = target @ target.T             # (B, B)
        diff_regime = 1.0 - same_regime
    else:
        same_regime = (target.unsqueeze(1) == target.unsqueeze(0)).float()
        diff_regime = 1.0 - same_regime

    mask = 1.0 - torch.eye(z.shape[0], device=z.device)  # exclude self-pairs

    attractive = ((dist ** 2) * same_regime * mask).sum() / (same_regime * mask).sum().clamp_min(1.0)
    repulsive  = (F.relu(margin - dist) * diff_regime * mask).sum() / (diff_regime * mask).sum().clamp_min(1.0)

    return attractive + repulsive


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, n_features, hidden_dim, n_regimes = 32, 14, 64, 3
    feature_embed_dim = 4

    model = RegimeDetectionLayer(
        n_features=n_features, hidden_dim=hidden_dim, n_regimes=n_regimes,
        feature_embed_dim=feature_embed_dim
    )

    phi_t = torch.randn(batch_size, n_features)

    z, p, logits = model(phi_t, tau=1.0)
    regime_ids = p.argmax(dim=-1)

    l_contrastive = contrastive_loss(z, regime_ids)

    print("stacked embed shape :  ", tuple(model.spline(phi_t).shape))  # (32, 14*4=56)
    print("embedding z:           ", tuple(z.shape))                     # (32, 64)
    print("regime probs p:        ", tuple(p.shape), "rows sum to", round(p.sum(-1)[0].item(), 4))
    print("regime logits:         ", tuple(logits.shape))
    print("regime assignment:     ", regime_ids[:10].tolist())
    print("contrastive loss:      ", round(l_contrastive.item(), 4))

    # Sanity check: gradients flow back through the spline knots' weights AND the projectors
    l_contrastive.backward()
    print("spline  w grad ok:", model.spline.splines[0].w.grad is not None)
    print("project w grad ok:", model.spline.projectors[0].weight.grad is not None)
