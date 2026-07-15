import torch
import torch.nn as nn
import torch.nn.functional as F
from spline_activation import SplineActivation

class RegimeDetectionLayer(nn.Module):
    """
    RegimeDetectionLayer of the KASPER framework.
    Encapsulates the spline feature transformation, MLP classifier, 
    and differentiable regime classification via Gumbel-Softmax.
    """

    def __init__(self, num_inputs=80, hidden_dim=64, num_regimes=3,
                 grid_size=10, n_linear=3, n_cubic=2, dropout_rate=0.2):
        """
        Args:
            num_inputs (int): Flat input features size (e.g. 10 time steps * 8 features = 80).
            hidden_dim (int): Dimensionality of the latent representations/embeddings.
            num_regimes (int): Total number of regimes (default: 3).
            grid_size (int): Number of knots in spline grids.
            n_linear (int): Number of linear basis splines.
            n_cubic (int): Number of cubic basis splines.
            dropout_rate (float): Dropout probability in MLP block (default: 0.2).
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes

        # 1. Instantiate the SplineActivation module
        self.spline = SplineActivation(
            num_inputs=num_inputs,
            num_outputs=hidden_dim,
            grid_size=grid_size,
            n_linear=n_linear,
            n_cubic=n_cubic
        )

        # 2. Implement classifier block: (L -> BN -> G -> D) x 2
        self.classifier_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

        # 3. Logits projection layer
        self.regime_proj = nn.Linear(hidden_dim, num_regimes)
        
        # Initialize regime projection weights to ensure good separation
        nn.init.orthogonal_(self.regime_proj.weight)

    def fit_knots(self, X_train):
        """
        Fits the quantile bounds of the SplineActivation module on training data.
        """
        self.spline.fit_knots(X_train)

    def forward(self, phi_t, tau=1.0, hard=False):
        """
        Args:
            phi_t (torch.Tensor): Input tensor, shape (B, T, F) or (B, num_inputs).
            tau (float): Temperature parameter for Gumbel-Softmax (default: 1.0).
            hard (bool): If True, returns hard one-hot probabilities (default: False).
            
        Returns:
            probs (torch.Tensor): Regime probabilities of shape (B, num_regimes).
            embeddings (torch.Tensor): Latent embeddings after MLP block, shape (B, hidden_dim).
        """
        # 1. Run spline activation to generate stable representations
        spline_out = self.spline(phi_t, aggregate=True)  # Shape: (B, hidden_dim)

        # 2. Run MLP block to get latent embeddings Z_t
        # BatchNorm1d requires batch size > 1. If batch size is 1, BN is bypassed or in eval mode.
        embeddings = self.classifier_block(spline_out)  # Shape: (B, hidden_dim)

        # 3. Logits projection
        logits = self.regime_proj(embeddings)  # Shape: (B, num_regimes)

        # 4. Equation 17 (Differentiable Regime Classification via Gumbel-Softmax)
        if self.training:
            # Differentiable sampling via Gumbel-Softmax
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            # Deterministic routing during evaluation to maintain consistency and stability
            probs = F.softmax(logits / tau, dim=-1)
            if hard:
                idx = torch.argmax(probs, dim=-1)
                probs = F.one_hot(idx, num_classes=self.num_regimes).float()

        return probs, embeddings


if __name__ == "__main__":
    print("--------------------------------------------------")
    print("RegimeDetectionLayer Shape Verification Test")
    print("--------------------------------------------------")
    device = torch.device("cpu")
    layer = RegimeDetectionLayer(
        num_inputs=80,
        hidden_dim=64,
        num_regimes=3,
        grid_size=10,
        n_linear=3,
        n_cubic=2,
        dropout_rate=0.2
    ).to(device)

    # Generate dummy input of shape [32, 80]
    dummy_input = torch.randn(32, 80)
    
    # Fit knots to prevent uninitialized RegisterBuffer error
    layer.fit_knots(dummy_input)

    # Run forward pass in training mode
    layer.train()
    probs, embeddings = layer(dummy_input, tau=1.0)

    print(f" - Input shape:      {dummy_input.shape}")
    print(f" - Probs shape:      {probs.shape} (Expected: [32, 3])")
    print(f" - Embeddings shape: {embeddings.shape} (Expected: [32, 64])")

    assert probs.shape == (32, 3), f"Incorrect probs shape: {probs.shape}"
    assert embeddings.shape == (32, 64), f"Incorrect embeddings shape: {embeddings.shape}"
    print("Verification completed successfully!")
    print("--------------------------------------------------")
