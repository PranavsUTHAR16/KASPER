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

    def __init__(self, num_inputs=80, hidden_dim=64, num_regimes=3,
                 grid_size=10, n_linear=3, n_cubic=2, dropout_rate=0.2,
                 num_knots=5, sparsity_threshold=1e-3):
        """
        Args:
            num_inputs (int): Flat input features size.
            hidden_dim (int): Dimensionality of the latent representation.
            num_regimes (int): Total number of regimes (default: 3).
            grid_size (int): Number of knots in Layer 1 spline grids.
            n_linear (int): Number of linear basis splines in Layer 1.
            n_cubic (int): Number of cubic basis splines in Layer 1.
            dropout_rate (float): Dropout probability in Layer 1.
            num_knots (int): Number of knots in Layer 2 spline grid (default: 5).
            sparsity_threshold (float): Sparsity threshold for Layer 2 weights.
        """
        super().__init__()
        
        # Layer 1: Regime Detection Layer
        self.layer1 = RegimeDetectionLayer(
            num_inputs=num_inputs,
            hidden_dim=hidden_dim,
            num_regimes=num_regimes,
            grid_size=grid_size,
            n_linear=n_linear,
            n_cubic=n_cubic,
            dropout_rate=dropout_rate
        )
        
        # Layer 2: Regime-Adaptive Forecasting Layer
        # Processes the same input features, guided by regime routing probabilities
        self.layer2 = RegimeAdaptiveForecastingLayer(
            num_features=num_inputs,
            num_regimes=num_regimes,
            num_knots=num_knots,
            sparsity_threshold=sparsity_threshold
        )

    def fit_knots(self, X_train):
        """
        Fits empirical quantile bounds for both Layer 1 and Layer 2 spline components.
        
        Args:
            X_train (torch.Tensor): Training data sequence, shape (B, T, F) or (B, num_inputs).
        """
        self.layer1.fit_knots(X_train)
        self.layer2.fit_knots(X_train)

    def forward(self, phi_t, tau=1.0, hard=False):
        """
        Args:
            phi_t (torch.Tensor): Input features tensor, shape (B, T, F) or (B, num_inputs).
            tau (float): Temperature for Gumbel-Softmax routing.
            hard (bool): If True, routes using hard one-hot assignments.
            
        Returns:
            final_forecast (torch.Tensor): Output forecast values, shape (B, 1).
            probs (torch.Tensor): Soft/hard regime probabilities, shape (B, num_regimes).
            embeddings (torch.Tensor): Stable latent representation, shape (B, hidden_dim).
        """
        # 1. Evaluate Layer 1 to get probabilities and latent representations
        probs, embeddings = self.layer1(phi_t, tau=tau, hard=hard)
        
        # 2. Evaluate Layer 2 to get final predictions guided by the probabilities
        final_forecast = self.layer2(phi_t, probs)
        
        return final_forecast, probs, embeddings


if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Testing KASPER Master Wrapper Model")
    print("--------------------------------------------------")

    # Generate dummy input tensors (Batch of 32, 8 features, 10 time steps -> 80 flat inputs)
    batch_size = 32
    time_steps = 10
    features = 8
    num_inputs = time_steps * features

    phi_t = torch.randn(batch_size, time_steps, features)
    print(f"Input phi_t shape: {phi_t.shape}")

    # Instantiate model
    model = KASPER(num_inputs=num_inputs, hidden_dim=64, num_regimes=3)
    
    print("\nFitting knots sequentially across both layers...")
    model.fit_knots(phi_t)
    
    print("\nRunning forward pass...")
    final_forecast, probs, embeddings = model(phi_t, tau=1.0)
    print(f" - Output final_forecast shape: {final_forecast.shape} (Expected: [{batch_size}, 1])")
    print(f" - Output probs shape:          {probs.shape} (Expected: [{batch_size}, 3])")
    print(f" - Output embeddings shape:     {embeddings.shape} (Expected: [{batch_size}, 64])")

    # Verification checks
    assert final_forecast.shape == (batch_size, 1), f"Incorrect prediction shape: {final_forecast.shape}"
    assert probs.shape == (batch_size, 3), f"Incorrect probability shape: {probs.shape}"
    assert embeddings.shape == (batch_size, 64), f"Incorrect embeddings shape: {embeddings.shape}"

    print("\nSuccess! KASPER master wrapper module initialized and ran correctly.")
