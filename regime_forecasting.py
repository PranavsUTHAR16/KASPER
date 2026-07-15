import torch
import torch.nn as nn

class RegimeAdaptiveForecastingLayer(nn.Module):
    """
    KAN Layer 2: Regime-Adaptive Forecasting Layer from the KASPER framework.
    Generates regime-specific price forecasts using local basis spline approximations
    and aggregates them based on regime probabilities.
    
    Equations:
    - Eq 21: phi_j^(r)(Phi_t) = sum_{k=1}^K beta_j,k^(r) * B_k(Phi_t; xi^(r))
    - Eq 22: w_sparse = sign(w) * ReLU(|w| - theta)
    - Eq 20: y_hat_t^(r) = sum_{j=1}^F w_sparse_j^(r) * phi_j^(r)(Phi_t)
    - Aggregation: y_hat = sum_{r=1}^R y_hat_t^(r) * P_t^(r)
    """

    def __init__(self, num_features=80, num_regimes=3, num_knots=5, sparsity_threshold=1e-3):
        """
        Args:
            num_features (int): Number of flat input features F.
            num_regimes (int): Number of regimes R (default: 3).
            num_knots (int): Number of knots K in the spline grid (default: 5).
            sparsity_threshold (float): Threshold theta for weight pruning (default: 1e-3).
        """
        super().__init__()
        self.num_features = num_features
        self.num_regimes = num_regimes
        self.num_knots = num_knots
        self.sparsity_threshold = sparsity_threshold

        # Eq 21: Trainable spline coefficients beta^(r)
        # Shape: [num_regimes, num_features, num_knots]
        self.beta = nn.Parameter(torch.randn(num_regimes, num_features, num_knots) * 0.1)

        # Eq 22: Trainable forecast weights w^(r)
        # Shape: [num_regimes, num_features]
        self.w = nn.Parameter(torch.randn(num_regimes, num_features) * 0.125)

        # Non-trainable buffers for input normalization range [x_min, x_max]
        # Registered per feature: shape [num_features]
        self.register_buffer("x_min", torch.zeros(num_features))
        self.register_buffer("x_max", torch.ones(num_features))

    def fit_knots(self, X_train):
        """
        Fits empirical quantile bounds of the features on the training set.
        Assumes 2D input of shape (B, num_features).
        """
        if X_train.size(1) != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.num_features}, got {X_train.size(1)}"
            )

        # Compute empirical quantiles (p_min = 0.01, p_max = 0.99)
        x_min = torch.quantile(X_train, 0.01, dim=0)
        x_max = torch.quantile(X_train, 0.99, dim=0)

        # Handle zero-variance/constant features
        eps = 1e-5
        x_max = torch.where(x_max == x_min, x_max + eps, x_max)

        # Update buffers
        self.x_min.copy_(x_min)
        self.x_max.copy_(x_max)

    def forward(self, phi_t, probs):
        """
        Args:
            phi_t (torch.Tensor): Input features tensor, shape (B, num_features).
            probs (torch.Tensor): Regime probabilities from Layer 1, shape (B, num_regimes).
            
        Returns:
            torch.Tensor: Aggregated prediction of shape (B, 1).
        """
        # 1. Normalize features to [0, 1] range to ensure B-spline activation stability
        x_min = self.x_min.unsqueeze(0)
        x_max = self.x_max.unsqueeze(0)
        x_norm = (phi_t - x_min) / (x_max - x_min + 1e-8)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)  # Shape: (B, num_features)

        # 2. Localized basis spline approximation (Gaussian RBF basis functions)
        # B_k(x) = exp(-0.5 * ((x - c_k) / sigma)^2)
        # Uniformly space centers on [0, 1]
        knots = torch.linspace(0.0, 1.0, self.num_knots, device=phi_t.device)  # Shape: (num_knots,)
        sigma = 1.0 / (self.num_knots - 1)  # Bandwidth factor

        # Compute basis activation for each feature and knot: shape (B, num_features, num_knots)
        basis = torch.exp(-0.5 * ((x_norm.unsqueeze(-1) - knots.unsqueeze(0).unsqueeze(0)) / sigma) ** 2)

        # 3. Apply trainable spline coefficients (beta) and sum over the K dimension (Eq 21)
        # Using einsum: phi_{b, r, f} = sum_k basis_{b, f, k} * beta_{r, f, k}
        # Result shape: [B, num_regimes, num_features]
        phi = torch.einsum('bfk,rfk->brf', basis, self.beta)

        # 4. Apply Sparsity Enforcement via soft-thresholding (Eq 22)
        # w_sparse = sign(w) * ReLU(|w| - theta)
        w_sparse = torch.sign(self.w) * torch.relu(torch.abs(self.w) - self.sparsity_threshold)

        # 5. Generate regime-specific forecasts (Eq 20)
        # Using einsum: y_hat_{b, r} = sum_f phi_{b, r, f} * w_sparse_{r, f}
        # Result shape: [B, num_regimes]
        y_hat = torch.einsum('brf,rf->br', phi, w_sparse)

        # 6. Weighted Sum Aggregation using regime probabilities
        # final_forecast_{b, 1} = sum_r y_hat_{b, r} * probs_{b, r}
        final_forecast = torch.sum(y_hat * probs, dim=-1, keepdim=True)  # Shape: (B, 1)

        return final_forecast


if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Testing RegimeAdaptiveForecastingLayer")
    print("--------------------------------------------------")

    # Parameters
    batch_size = 32
    num_features = 8
    num_regimes = 3
    num_knots = 5
    sparsity_threshold = 1e-3

    # Generate dummy input tensors
    # Raw features (Batch of 32, 8 features)
    phi_t = torch.randn(batch_size, num_features)
    # Regime probabilities (Batch of 32, 3 regimes) summing to 1.0
    probs = torch.randn(batch_size, num_regimes).softmax(dim=-1)

    print(f"Inputs:")
    print(f" - phi_t (features) shape: {phi_t.shape}")
    print(f" - probs (probabilities) shape: {probs.shape}")

    # Instantiate layer
    layer = RegimeAdaptiveForecastingLayer(
        num_features=num_features,
        num_regimes=num_regimes,
        num_knots=num_knots,
        sparsity_threshold=sparsity_threshold
    )

    print("\nFitting knots on dummy data...")
    layer.fit_knots(phi_t)

    print("\nRunning forward pass...")
    # Verify outputs are trainable
    output = layer(phi_t, probs)
    print(f" - Output shape: {output.shape} (Expected: [{batch_size}, 1])")

    # Verification checks
    assert output.shape == (batch_size, 1), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Found NaN in the forecasting layer outputs"
    
    # Verify gradient flow
    loss = output.sum()
    loss.backward()
    print(" - Gradient wrt beta: ", layer.beta.grad is not None)
    print(" - Gradient wrt w:    ", layer.w.grad is not None)

    print("\nSuccess! RegimeAdaptiveForecastingLayer functions correctly, is fully vectorized, and maintains gradient flow.")
