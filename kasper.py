import torch
import torch.nn as nn
from regime_detection import RegimeDetectionLayer
from regime_forecasting import RegimeAdaptiveForecastingLayer

class KASPER(nn.Module):
    """
    KASPER (Kolmogorov-Arnold Networks for Stock Prediction and Explainable Regimes)
    Master Model wrapper.
    
    Wires Layer 1 (RegimeDetectionLayer) and Layer 2 (RegimeAdaptiveForecastingLayer) together.
    """

    def __init__(self, num_inputs=33, hidden_dim=64, num_regimes=3,
                 grid_size=10, n_linear=3, n_cubic=2, dropout_rate=0.2,
                 num_knots=5, sparsity_threshold=1e-3, feature_embed_dim=4):
        """
        Args:
            num_inputs (int): Flat input features size.
            hidden_dim (int): Dimensionality of the latent representation.
            num_regimes (int): Total number of regimes (default: 3).
            grid_size (int): Grid size (not used in new implementation).
            n_linear (int): Number of linear basis splines in Layer 1.
            n_cubic (int): Number of cubic basis splines in Layer 1.
            dropout_rate (float): Dropout probability in Layer 1.
            num_knots (int): Number of basis functions in Layer 2 (n_basis).
            sparsity_threshold (float): Not used directly in new implementation constructor.
            feature_embed_dim (int): Per-feature embedding dimension in Layer 1 (default: 4).
        """
        super().__init__()
        
        # Layer 1: Regime Detection Layer
        self.layer1 = RegimeDetectionLayer(
            n_features=num_inputs,
            hidden_dim=hidden_dim,
            n_regimes=num_regimes,
            dropout=dropout_rate,
            n_linear=n_linear,
            n_cubic=n_cubic,
            feature_embed_dim=feature_embed_dim,
        )
        
        # Layer 2: Regime-Adaptive Forecasting Layer
        # num_knots parameter maps to n_basis in the new implementation
        self.layer2 = RegimeAdaptiveForecastingLayer(
            n_features=num_inputs,
            n_regimes=num_regimes,
            n_basis=num_knots,
            spline_order=3
        )

    def fit_knots(self, X_train):
        """
        Fits empirical quantile bounds for both Layer 1 and Layer 2 spline components.
        
        Args:
            X_train (torch.Tensor): Training data sequence, shape (B, num_inputs).
        """
        # Fit Layer 1 splines
        for j, spline in enumerate(self.layer1.spline.splines):
            spline.fit_knots(X_train[:, j])
        # Fit Layer 2 splines
        for r in range(self.layer2.n_regimes):
            for j, spline in enumerate(self.layer2.splines[r]):
                spline.fit_knots(X_train[:, j])

    def forward(self, phi_t, tau=1.0, hard=False):
        """
        Args:
            phi_t (torch.Tensor): Input features tensor, shape (B, num_inputs).
            tau (float): Temperature for Gumbel-Softmax routing.
            hard (bool): If True, routes using hard one-hot assignments.
            
        Returns:
            final_forecast (torch.Tensor): Output forecast values, shape (B, 1) or (B,).
            probs (torch.Tensor): Soft/hard regime probabilities, shape (B, num_regimes).
            embeddings (torch.Tensor): Stable latent representation, shape (B, hidden_dim).
        """
        # 1. Evaluate Layer 1 to get probabilities and latent representations
        embeddings, probs, logits = self.layer1(phi_t, tau=tau, hard=hard)

        # 2. Evaluate Layer 2 to get final predictions guided by the probabilities.
        #    Standard joint gradient flow — the paper does not use stop-gradient.
        #    Layer 2 returns: (y_hat, forecast_per_regime, phi_per_regime)
        final_forecast, _, _ = self.layer2(phi_t, probs)

        return final_forecast, probs, embeddings


if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Testing KASPER Master Wrapper Model")
    print("--------------------------------------------------")

    # Generate dummy input tensors (Batch of 32, 33 features)
    batch_size = 32
    num_inputs = 33

    phi_t = torch.randn(batch_size, num_inputs)
    print(f"Input phi_t shape: {phi_t.shape}")

    # Instantiate model
    model = KASPER(num_inputs=num_inputs, hidden_dim=64, num_regimes=3)
    
    print("\nFitting knots sequentially across both layers...")
    model.fit_knots(phi_t)
    
    print("\nRunning forward pass...")
    final_forecast, probs, embeddings = model(phi_t, tau=1.0)
    print(f" - Output final_forecast shape: {final_forecast.shape} (Expected: [{batch_size}])")
    print(f" - Output probs shape:          {probs.shape} (Expected: [{batch_size}, 3])")
    print(f" - Output embeddings shape:     {embeddings.shape} (Expected: [{batch_size}, 64])")

    # Verification checks
    assert final_forecast.shape == (batch_size,), f"Incorrect prediction shape: {final_forecast.shape}"
    assert probs.shape == (batch_size, 3), f"Incorrect probability shape: {probs.shape}"
    assert embeddings.shape == (batch_size, 64), f"Incorrect embeddings shape: {embeddings.shape}"

    print("\nSuccess! KASPER master wrapper module initialized and ran correctly.")
