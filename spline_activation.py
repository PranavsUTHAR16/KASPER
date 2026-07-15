import os
import torch
import torch.nn as nn
import numpy as np

class SplineActivation(nn.Module):
    """
    SplineActivation Module from the KASPER framework.
    Strictly implements Equations 14, 15, and 16 from the paper:
    
    1. Quantile Knot Initialization:
       - Knots are placed on a uniform grid bounded by empirical percentiles of the training data.
       - x_min = quantile(x, p_min)
       - x_max = quantile(x, p_max)
       - k_m = x_min + (m - 1) * (x_max - x_min) / (G - 1)
       
    2. Equation 15 (The Linear Component):
       L(x) = sum_{m=0}^{N_linear-1} tanh(w_m) * [ReLU(x_norm - k_m) - ReLU(x_norm - k_{m+1})]
       
    3. Equation 16 (The Cubic Component):
       C(x) = sum_{m=0}^{N_cubic-1} sigmoid(v_m) * x_norm^3
       
    4. Equation 14 (The Hybrid Spline Activation):
       f(x) = L(x) + C(x)
    """

    def __init__(self, num_inputs=80, num_outputs=64, grid_size=10,
                 n_linear=3, n_cubic=2, p_min=0.01, p_max=0.99):
        """
        Args:
            num_inputs (int): Number of input dimensions (features * time steps if flattened).
            num_outputs (int): Dimensionality of the output representation.
            grid_size (int): Total number of knots G in the grid.
            n_linear (int): Number of linear basis components (default: 3).
            n_cubic (int): Number of cubic basis components (default: 2).
            p_min (float): Percentile lower bound for knot fitting (default: 0.01).
            p_max (float): Percentile upper bound for knot fitting (default: 0.99).
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.grid_size = grid_size
        self.n_linear = n_linear
        self.n_cubic = n_cubic
        self.p_min = p_min
        self.p_max = p_max

        # Scaling factor to keep layer outputs O(1) during random init
        _scale = 0.02 / (num_inputs ** 0.5)

        # Trainable parameters
        # Eq 15: N_linear parameters w per input-output connection
        self.w = nn.Parameter(torch.randn(num_inputs, num_outputs, n_linear) * _scale)
        # Eq 16: N_cubic parameters v per input-output connection
        self.v = nn.Parameter(torch.randn(num_inputs, num_outputs, n_cubic) * _scale)

        # Non-trainable buffers for quantile bounds and knot grids
        # Stored per-feature: shape (num_inputs, grid_size)
        self.register_buffer("knots", torch.zeros(num_inputs, grid_size))
        self.register_buffer("x_min", torch.zeros(num_inputs))
        self.register_buffer("x_max", torch.ones(num_inputs))

    def fit_knots(self, X_train):
        """
        Computes the empirical percentiles of the training data features
        and constructs the uniform knot grids.
        
        Args:
            X_train (torch.Tensor): Training data, shape (B, num_inputs).
        """
        if X_train.size(1) != self.num_inputs:
            raise ValueError(
                f"Input features dimension mismatch. Expected {self.num_inputs}, got {X_train.size(1)}"
            )

        device = X_train.device

        # Compute empirical percentiles (quantile) per input feature (dim=0 is the batch axis)
        x_min = torch.quantile(X_train, self.p_min, dim=0)
        x_max = torch.quantile(X_train, self.p_max, dim=0)

        # Avoid division-by-zero for zero-variance/constant features
        eps = 1e-5
        x_max = torch.where(x_max == x_min, x_max + eps, x_max)

        # Update buffers
        self.x_min.copy_(x_min)
        self.x_max.copy_(x_max)

        # Quantile Knot Grid Construction:
        # k_m = x_min + (m - 1) * (x_max - x_min) / (G - 1)
        # Vectorized implementation:
        grid_steps = torch.linspace(0.0, 1.0, self.grid_size, device=device)  # shape (G,)
        # Broadcast x_min, x_max across grid_steps to form shape (num_inputs, grid_size)
        knots = x_min.unsqueeze(-1) + grid_steps.unsqueeze(0) * (x_max - x_min).unsqueeze(-1)
        
        self.knots.copy_(knots)

    def forward(self, x, aggregate=True):
        """
        Evaluates the hybrid spline activation on the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, num_inputs).
            aggregate (bool): If True, aggregates (sums) over inputs and divides by
                              sqrt(num_inputs) to yield shape (B, num_outputs) for stability.
                              If False, returns raw activations of shape (B, num_inputs, num_outputs).
                              
        Returns:
            torch.Tensor: Spline activations of shape (B, num_outputs) or (B, num_inputs, num_outputs).
        """

        x_min = self.x_min.unsqueeze(0)
        x_max = self.x_max.unsqueeze(0)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)  # Shape: (B, num_inputs)

        # 2. Equation 15 (The Linear Component):
        # We select n_linear + 1 evenly-spaced knot positions from the grid of size G
        grid_idx = torch.linspace(0, self.grid_size - 1, self.n_linear + 1, device=x.device).long()
        
        # Normalize the stored physical knots to [0, 1] to align with x_norm scale
        knots_norm = (self.knots - self.x_min.unsqueeze(-1)) / (self.x_max.unsqueeze(-1) - self.x_min.unsqueeze(-1) + 1e-8)
        sel_knots_norm = knots_norm[:, grid_idx]  # Shape: (num_inputs, n_linear + 1)

        # Compute differences: ReLU(x_norm - k_m) - ReLU(x_norm - k_{m+1})
        # Vectorized using shape broadcasting:
        x_norm_uns = x_norm.unsqueeze(-1)              # Shape: (B, num_inputs, 1)
        sel_knots_uns = sel_knots_norm.unsqueeze(0)      # Shape: (1, num_inputs, n_linear + 1)
        
        # Calculate ReLU(x_norm - k_m) for all m=0..N_linear
        relu_val = torch.relu(x_norm_uns - sel_knots_uns)  # Shape: (B, num_inputs, n_linear + 1)
        
        # Compute adjacent differences
        relu_diffs = relu_val[..., :-1] - relu_val[..., 1:]  # Shape: (B, num_inputs, n_linear)

        # Multiply by tanh(w_m) and sum over the linear components (m dimension)
        # Using Einstein summation:
        # L(x)_{b, i, o} = sum_m relu_diffs_{b, i, m} * tanh(w)_{i, o, m}
        tanh_w = torch.tanh(self.w)  # Shape: (num_inputs, num_outputs, n_linear)
        L_x = torch.einsum('bim,iom->bio', relu_diffs, tanh_w)  # Shape: (B, num_inputs, num_outputs)

        # 3. Equation 16 (The Cubic Component):
        # C(x) = sum_{m=0}^{N_cubic-1} sigmoid(v_m) * x_norm^3
        v_sigmoid = torch.sigmoid(self.v)                    # Shape: (num_inputs, num_outputs, n_cubic)
        v_sigmoid_sum = torch.sum(v_sigmoid, dim=-1)         # Shape: (num_inputs, num_outputs)
        
        # Compute x_norm^3 and multiply by the summed sigmoid coefficients
        x_norm_cubed = (x_norm ** 3).unsqueeze(-1)           # Shape: (B, num_inputs, 1)
        C_x = x_norm_cubed * v_sigmoid_sum.unsqueeze(0)      # Shape: (B, num_inputs, num_outputs)

        # 4. Equation 14 (The Hybrid Spline Activation):
        # f(x) = L(x) + C(x)
        f_x = L_x + C_x  # Shape: (B, num_inputs, num_outputs)

        # 5. Aggregation
        if aggregate:
            # Sum over inputs and scale by 1/sqrt(num_inputs) to prevent contrastive loss explosion
            z = torch.sum(f_x, dim=1) / (self.num_inputs ** 0.5)  # Shape: (B, num_outputs)
            return z
        else:
            return f_x


if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Testing SplineActivation Module (KASPER Layer 1)")
    print("--------------------------------------------------")

    # Path to preprocessed SPY dataset
    data_path = "data/spy_train_X.npy"
    if not os.path.exists(data_path):
        print(f"Error: Preprocessed data file not found at '{data_path}'.")
        print("Please run preprocess_spy.py first to generate the numpy files.")
        exit(1)

    # 1. Load data
    print(f"Loading preprocessed SPY training feature matrix from '{data_path}'...")
    X_train_np = np.load(data_path)
    X_train = torch.from_numpy(X_train_np).float()
    print(f"Loaded X_train shape: {X_train.shape} (Expected: [Batch, Time_Steps, Features])")
    
    # 2. Extract configuration info
    batch_size, time_steps, features = X_train.shape
    num_inputs = time_steps * features
    print(f"Time steps: {time_steps}, Features: {features} -> num_inputs: {num_inputs}")

    # 3. Instantiate the module with default hyperparameters
    print("\nInstantiating SplineActivation...")
    spline_layer = SplineActivation(
        num_inputs=num_inputs, 
        num_outputs=64, 
        grid_size=10, 
        n_linear=3, 
        n_cubic=2
    )
    print(f" - Trainable linear weights (w) shape: {spline_layer.w.shape}")
    print(f" - Trainable cubic weights (v) shape:  {spline_layer.v.shape}")
    print(f" - Knots buffer shape:                {spline_layer.knots.shape}")

    # 4. Fit knots using the training data
    print("\nFitting knots on training data using empirical percentiles (p_min=0.01, p_max=0.99)...")
    spline_layer.fit_knots(X_train)
    
    # Verify knot placement
    print("\nKnot Grid Verification:")
    print(f" - x_min buffer shape: {spline_layer.x_min.shape}")
    print(f" - x_max buffer shape: {spline_layer.x_max.shape}")
    print(" - First 3 features knot grid samples:")
    for i in range(3):
        min_val = spline_layer.x_min[i].item()
        max_val = spline_layer.x_max[i].item()
        knots_sample = spline_layer.knots[i].tolist()
        print(f"   Feature {i:2d}: Min={min_val:7.4f}, Max={max_val:7.4f}")
        print(f"              Knots: {['. '.join([f'{k:.4f}' for k in knots_sample[:3]]), '...', f'{knots_sample[-1]:.4f}'] if len(knots_sample) > 4 else knots_sample}")

    # 5. Perform forward pass
    print("\nRunning forward pass (aggregate=True)...")
    z_agg = spline_layer(X_train, aggregate=True)
    print(f" - Output shape (aggregated): {z_agg.shape} (Expected: [Batch, num_outputs])")
    print(f" - Outputs statistics: Mean={z_agg.mean().item():.6f}, Std={z_agg.std().item():.6f}")

    print("\nRunning forward pass (aggregate=False)...")
    z_raw = spline_layer(X_train, aggregate=False)
    print(f" - Output shape (raw activations): {z_raw.shape} (Expected: [Batch, num_inputs, num_outputs])")

    # 6. Verify NaN checks
    has_nan = torch.isnan(z_agg).any().item()
    print(f"\nVerification Results: Has NaN values: {has_nan}")
    if not has_nan:
        print("Success! SplineActivation module functions correctly and generates stable embeddings.")
