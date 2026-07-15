import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def soft_threshold_ste(x, threshold):
    return SoftThresholdSTE.apply(x, threshold)


class SplineActivation(nn.Module):
    """
    KAN Spline activation layer (Eq 14, 15, 16).

    BUG 5 FIX: Paper Table 1 specifies "Hybrid linear(3) and cubic(2) splines".
    Previous code used n_linear=9 (grid_size-1) and n_cubic=5.  Now uses the
    paper's exact counts: N_linear=3 trainable w coefficients and N_cubic=2
    trainable v coefficients.

    The 3 linear terms span the full input range via evenly-spaced knot pairs
    selected from the grid.  The 2 cubic terms weight x_norm^3.
    """

    def __init__(self, num_inputs=80, num_outputs=64, grid_size=10,
                 n_linear=3, n_cubic=2, sparsity_threshold=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.grid_size = grid_size
        self.n_linear = n_linear       # Paper: 3
        self.n_cubic = n_cubic         # Paper: 2
        self.sparsity_threshold = sparsity_threshold

        # Scale init by 1/sqrt(num_inputs) so that summing over all inputs
        # keeps each output z_j ~ O(1) instead of O(num_inputs).
        # Without this, contrastive loss ||z_i - z_j||^2 explodes.
        _scale = 0.02 / (num_inputs ** 0.5)
        # Eq 15: 3 trainable linear coefficients  (was grid_size-1 = 9)
        self.w = nn.Parameter(torch.randn(num_inputs, num_outputs, n_linear) * _scale)
        # Eq 16: 2 trainable cubic coefficients    (was 5)
        self.v = nn.Parameter(torch.randn(num_inputs, num_outputs, n_cubic) * _scale)

        # Uniform knot grid on [0, 1]
        knots = torch.linspace(0.0, 1.0, grid_size)
        self.register_buffer("knots", knots)

        # Per-feature quantile bounds (set via fit_knots)
        self.register_buffer("x_min", torch.zeros(num_inputs))
        self.register_buffer("x_max", torch.ones(num_inputs))

    # ------------------------------------------------------------------
    def fit_knots(self, X_train):
        """
        Computes x_min and x_max using percentiles (p=0.01, 0.99).
        Handles both 3D (B, T, F) and 2D (B, D) inputs.
        """
        if X_train.dim() == 3:
            X_flat = X_train.view(X_train.size(0), -1)
        else:
            X_flat = X_train

        q_min = torch.tensor([0.01], device=X_train.device)
        q_max = torch.tensor([0.99], device=X_train.device)

        for j in range(self.num_inputs):
            feat = X_flat[:, j]
            self.x_min[j] = torch.quantile(feat, q_min)
            self.x_max[j] = torch.quantile(feat, q_max)
            if self.x_max[j] == self.x_min[j]:
                self.x_max[j] += 1e-5

    # ------------------------------------------------------------------
    def _soft_threshold(self, param, threshold=None):
        """
        Eq 22:  w_j^(r) ~ ReLU(|w_j^(r)| - theta^(r))
        Soft thresholding using Straight-Through Estimator (STE).
        """
        t = threshold if threshold is not None else self.sparsity_threshold
        return soft_threshold_ste(param, t)

    # ------------------------------------------------------------------
    def forward(self, phi_t, threshold=None):
        """
        Args:
            phi_t: (B, T, F) state matrix or (B, D) flattened
            threshold: optional override for the sparsity threshold (Eq 22).
                BUG 10 FIX: init std (~0.02/sqrt(num_inputs)) is far smaller
                than the fixed sparsity_threshold (0.01), so at t=0 nearly
                every |w|,|v| < threshold -> ReLU(|w|-threshold) sits in its
                zero-gradient region for ALL weights simultaneously. Nothing
                can ever grow out, so z stays ~0 forever. Callers now anneal
                this threshold up from 0 during training (see main.py) so
                weights can move before sparsity starts pruning them.
        Returns:
            z: (B, num_outputs) embeddings
        """
        if phi_t.dim() == 3:
            phi_t = phi_t.view(phi_t.size(0), -1)

        # Quantile-normalise to [0, 1]
        x_norm = (phi_t - self.x_min.unsqueeze(0)) / \
                 (self.x_max.unsqueeze(0) - self.x_min.unsqueeze(0) + 1e-8)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)       # prevent x^3 explosion
        x_norm_uns = x_norm.unsqueeze(-1)              # (B, num_inputs, 1)

        # ---- Eq 15: Linear component L(x) ----
        # Select n_linear+1 evenly-spaced knot positions from the full grid
        # so the 3 hat-functions span the entire [0, 1] range.
        knot_idx = torch.linspace(0, self.grid_size - 1, self.n_linear + 1,
                                  device=phi_t.device).long()
        sel_knots = self.knots[knot_idx]              # (n_linear + 1,)

        relu_diffs = []
        for m in range(self.n_linear):
            diff = (torch.relu(x_norm - sel_knots[m])
                    - torch.relu(x_norm - sel_knots[m + 1]))
            relu_diffs.append(diff)
        # (B, num_inputs, n_linear)
        relu_stack = torch.stack(relu_diffs, dim=-1)

        # Soft-threshold then tanh (Eq 22 + Eq 15 tanh wrapper)
        w_sparse = self._soft_threshold(self.w, threshold)
        tanh_w = torch.tanh(w_sparse)                 # (num_inputs, num_outputs, n_linear)

        # L(x) = sum_m tanh(w_m) * hat_m(x)
        L_x = torch.sum(relu_stack.unsqueeze(2) * tanh_w.unsqueeze(0), dim=-1)
        # -> (B, num_inputs, num_outputs)

        # ---- Eq 16: Cubic component C(x) ----
        v_sparse = self._soft_threshold(self.v, threshold)
        v_sigmoid_sum = torch.sum(torch.sigmoid(v_sparse), dim=-1)   # (num_inputs, num_outputs)
        C_x = (x_norm_uns ** 3) * v_sigmoid_sum.unsqueeze(0)
        # -> (B, num_inputs, num_outputs)

        # f(x) = L(x) + C(x), summed over input features, then
        # divide by sqrt(num_inputs) to keep z ~ O(1) for contrastive loss.
        z = torch.sum(L_x + C_x, dim=1) / (self.num_inputs ** 0.5)
        return z


# ======================================================================
class RegimeDetectionLayer(nn.Module):
    """
    KAN Layer 1: Regime Detection Layer (Section 3.1)
    """

    def __init__(self, num_inputs=80, hidden_dim=64, num_regimes=3,
                 grid_size=10, n_linear=3, n_cubic=2, dropout_rate=0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes

        self.spline = SplineActivation(num_inputs, hidden_dim,
                                       grid_size, n_linear, n_cubic)

        # Fig 2: (L -> BN -> GELU -> D) x 2
        self.classifier_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.regime_projection = nn.Linear(hidden_dim, num_regimes)
        # BUG 13 FIX (part 2): default nn.Linear init gives no separation
        # guarantee between the num_regimes rows of regime_projection.weight
        # -- two rows can start near-parallel, and with only lambda_o=0.01
        # pulling them apart, one regime can end up permanently "runner-up"
        # to another and never win the hard argmax. Orthogonal init gives
        # the 3 regime directions maximal separation from step one, so the
        # orthogonality loss only has to maintain separation, not create it.
        nn.init.orthogonal_(self.regime_projection.weight)

    def fit_knots(self, X_train):
        self.spline.fit_knots(X_train)

    def forward(self, phi_t, tau=1.0, hard=False, threshold=None):
        if self.training:
            # Eq 17: Generate independent Gumbel noise for each of the 3 regimes
            U = torch.rand(self.num_regimes, phi_t.size(0), phi_t.size(1), phi_t.size(2), device=phi_t.device)
            g = -torch.log(-torch.log(U + 1e-20) + 1e-20) # (R, B, 10, 8)
            
            logits_list = []
            z_list = []
            
            for r in range(self.num_regimes):
                # Add scaled noise (0.1) to input features for this regime
                phi_noisy_r = phi_t + 0.1 * g[r]
                
                # Forward pass through Spline + Classifier + Projection for this regime
                z_r = self.spline(phi_noisy_r, threshold=threshold)
                z_refined_r = self.classifier_block(z_r)
                logits_r = self.regime_projection(z_refined_r) # (B, R)
                
                # Collect the specific logit corresponding to this regime r
                logits_list.append(logits_r[:, r])
                z_list.append(z_refined_r)
                
            # Stack to (B, R)
            logits = torch.stack(logits_list, dim=1)
            
            # Apply standard softmax scaled by temperature tau
            probabilities = F.softmax(logits / tau, dim=-1)
            
            if hard:
                # Straight-Through Estimator one-hot output
                idx = torch.argmax(probabilities, dim=-1)
                probs_hard = F.one_hot(idx, num_classes=self.num_regimes).float()
                probabilities = probs_hard - probabilities.detach() + probabilities
                
            # Return average z_refined across the noisy paths for Layer 2
            z = torch.stack(z_list, dim=0).mean(dim=0)
            
        else:
            # Deterministic forward pass at evaluation (no noise)
            z_spline = self.spline(phi_t, threshold=threshold)
            z = self.classifier_block(z_spline)
            logits = self.regime_projection(z)
            
            if hard:
                idx = torch.argmax(logits, dim=-1)
                probabilities = F.one_hot(idx, num_classes=self.num_regimes).float()
            else:
                probabilities = F.softmax(logits / tau, dim=-1)
                
        return logits, probabilities, z




# ======================================================================
class RegimeAdaptiveForecastingLayer(nn.Module):
    """
    KAN Layer 2: Regime-Adaptive Forecasting (Section 3.2)

    BUG 2 FIX: Complete rewrite to match paper Eq 20-21.
      Eq 20:  y_hat_t^(r) = sum_j  w_j^(r) * phi_j^(r)(Phi_t)
      Eq 21:  phi_j^(r)(Phi_t) = sum_k  beta_j,k^(r) * B_k(Phi_t; xi^(r))

    Each regime has:
      - Per-feature B-spline coefficients beta (num_inputs x n_basis)
      - Forecast weights w (num_inputs,)
    The B-spline basis B_k is a standard hat-function on the knot grid.

    BUG 9 FIX: Added attention-based aggregation (Fig 2) that learns
    per-sample regime importance from the embedding z.
    """

    def __init__(self, num_inputs=64, num_regimes=3, n_basis=5,
                 sparsity_threshold=0.05):
        super().__init__()
        self.num_regimes = num_regimes
        self.num_inputs = num_inputs
        self.n_basis = n_basis
        self.sparsity_threshold = sparsity_threshold

        # Eq 21: regime-specific B-spline coefficients beta^(r)
        self.beta = nn.ParameterList([
            nn.Parameter(torch.randn(num_inputs, n_basis) * 0.1)
            for _ in range(num_regimes)
        ])

        # Eq 20: regime-specific forecast weights w^(r)
        self.forecast_weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_inputs) * 0.125)
            for _ in range(num_regimes)
        ])

        # BUG 9 FIX: Attention mechanism for regime aggregation (Fig 2)
        self.attention_net = nn.Linear(num_inputs, num_regimes)

        # B-spline knots: n_basis intervals -> n_basis+1 knot positions
        knots = torch.linspace(0.0, 1.0, n_basis + 1)
        self.register_buffer("knots", knots)

        self.register_buffer("x_min", torch.zeros(num_inputs))
        self.register_buffer("x_max", torch.ones(num_inputs))

    # ------------------------------------------------------------------
    def fit_knots(self, z_train):
        """
        Fits B-spline knots based on Layer 1 embeddings z.
        BUG 1 FIX: This must receive the 64-dim embeddings, NOT raw input.
        """
        if z_train.dim() == 3:
            z_train = z_train.view(z_train.size(0), -1)

        with torch.no_grad():
            q_min = torch.tensor([0.01], device=z_train.device)
            q_max = torch.tensor([0.99], device=z_train.device)
            for j in range(self.num_inputs):
                feat = z_train[:, j]
                self.x_min[j] = torch.quantile(feat, q_min)
                self.x_max[j] = torch.quantile(feat, q_max)
                if self.x_max[j] == self.x_min[j]:
                    self.x_max[j] += 1e-5

    # ------------------------------------------------------------------
    def _soft_threshold(self, param, threshold=None):
        """Eq 22:  w ~ ReLU(|w| - theta) using Straight-Through Estimator (STE)"""
        t = threshold if threshold is not None else self.sparsity_threshold
        return soft_threshold_ste(param, t)

    # ------------------------------------------------------------------
    def forward(self, z, threshold=None):
        """
        Args:
            z: (B, num_inputs)  regime embeddings from Layer 1
            threshold: optional override for sparsity threshold (Eq 22).
                BUG 10 FIX: same dead-zone issue as SplineActivation -- init
                std (0.02) is well below the default threshold (0.05), so
                beta/forecast_weights start almost entirely inside the
                zero-gradient region and never move. Anneal from 0 during
                training (see main.py).
        Returns:
            y_preds:      (B, num_regimes)  regime-specific forecasts
            attn_weights: (B, num_regimes)  attention-based aggregation weights
        """
        # Normalise to [0, 1]
        x_norm = (z - self.x_min.unsqueeze(0)) / \
                 (self.x_max.unsqueeze(0) - self.x_min.unsqueeze(0) + 1e-8)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)

        # B-spline basis: B_k(x) = ReLU(x - k_k) - ReLU(x - k_{k+1})
        # shape: (B, num_inputs, n_basis)
        basis = (torch.relu(x_norm.unsqueeze(-1) - self.knots[:-1])
                 - torch.relu(x_norm.unsqueeze(-1) - self.knots[1:]))

        y_preds = []
        for r in range(self.num_regimes):
            # Soft-threshold (Eq 22)
            beta_sparse = self._soft_threshold(self.beta[r], threshold)       # (num_inputs, n_basis)
            w_sparse   = self._soft_threshold(self.forecast_weights[r], threshold)  # (num_inputs,)

            # Eq 21: phi_j^(r)(z) = sum_k beta_j,k * B_k(z_j)
            phi = torch.sum(basis * beta_sparse.unsqueeze(0), dim=-1)  # (B, num_inputs)

            # Eq 20: y_hat^(r) = sum_j w_j^(r) * phi_j^(r)(z)
            pred_r = torch.sum(phi * w_sparse.unsqueeze(0), dim=-1)   # (B,)
            y_preds.append(pred_r)

        y_preds = torch.stack(y_preds, dim=1)          # (B, R)

        # BUG 9 FIX: Attention-based aggregation (Fig 2)
        attn_logits = self.attention_net(z)             # (B, R)
        attn_weights = F.softmax(attn_logits, dim=-1)   # (B, R)

        return y_preds, attn_weights

    def get_orthogonality_loss(self):
        """Eq 19: L_orth = ||W W^T - I||_F^2 on forecast weights"""
        # Stack forecast weights of all regimes: shape (R, F)
        W = torch.stack(list(self.forecast_weights), dim=0)
        W_Wt = torch.mm(W, W.t())
        I = torch.eye(self.num_regimes, device=W.device)
        return torch.norm(W_Wt - I, p='fro') ** 2