import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from kasper import KASPER

def load_feature_names():
    selected_path = "data/selected_features.txt"
    if os.path.exists(selected_path):
        with open(selected_path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
            if names:
                return names
    return [
        "OC_Spread", "Log_Return_1d", "Log_Return_High_1d", "Log_Return_Low_1d",
        "Log_Return_Open_1d", "ATR_21d", "Velocity", "Acceleration"
    ]

FEATURE_NAMES = load_feature_names()


def get_regime_forecasts(model, phi_t):
    """
    Evaluates the un-aggregated regime-specific forecasts from Layer 2.
    
    Args:
        model (nn.Module): The KASPER master model.
        phi_t (torch.Tensor): Features tensor, shape (B, num_features).
        
    Returns:
        torch.Tensor: Un-aggregated predictions of shape (B, num_regimes).
    """
    dummy_p = torch.ones(phi_t.shape[0], model.layer2.n_regimes, device=phi_t.device) / model.layer2.n_regimes
    _, forecast_per_regime, _ = model.layer2(phi_t, dummy_p)
    return forecast_per_regime


def estimate_shapley(model: nn.Module, x: torch.Tensor, num_permutations: int = 50, y_std: float = 1.0) -> torch.Tensor:
    """
    Computes permutation-based Monte Carlo Shapley values for all inputs of a single sample (Eq. 24, 25).
    Evaluates the progressive feature subsets in parallel for speed.
    
    Args:
        model (nn.Module): The KASPER master model.
        x (torch.Tensor): Input sample of shape (num_features,).
        num_permutations (int): Number of random permutations to sample.
        
    Returns:
        torch.Tensor: Shapley values of shape (num_regimes, num_features).
    """
    D = x.shape[0]  # num_features
    
    # Initialize Shapley values buffer
    shapley_values = torch.zeros(model.layer2.n_regimes, D, device=x.device)
    
    for m in range(num_permutations):
        # Sample a random permutation of feature indices
        perm = torch.randperm(D, device=x.device)
        
        # Build batch of progressive subsets of shape (D + 1, D)
        batch_inputs = torch.zeros(D + 1, D, device=x.device)
        for k in range(1, D + 1):
            batch_inputs[k, :] = batch_inputs[k - 1, :].clone()
            batch_inputs[k, perm[k - 1]] = x[perm[k - 1]]
            
        # Evaluate model forecasts for all regimes: shape (D + 1, num_regimes)
        y_hat = get_regime_forecasts(model, batch_inputs)
        
        # Compute marginal contributions: (y_hat[1:] - y_hat[:-1]) * y_std
        # Note: y_mean cancels out in difference ((y1*y_std + y_mean) - (y0*y_std + y_mean) = (y1-y0)*y_std).
        diffs = (y_hat[1:] - y_hat[:-1]) * y_std
        
        # Accumulate diffs into the corresponding permutation indices
        shapley_values.scatter_add_(1, perm.unsqueeze(0).expand(model.layer2.n_regimes, -1), diffs.t())
        
    # Average across all permutations
    shapley_values /= num_permutations
    return shapley_values


def apply_temporal_weighting(shapley_tensor: torch.Tensor, gamma: float = 0.95) -> torch.Tensor:
    """
    Applies Eq. 26 temporal weighting scheme to sequence of Shapley values:
    w_t = gamma^(T-t) / sum_{k=1}^T gamma^(T-k)

    Args:
        shapley_tensor: Tensor of shape (T, num_regimes, num_features).
        gamma: Exponential decay factor.

    Returns:
        Temporally weighted Shapley values tensor of shape (num_regimes, num_features).
    """
    T = shapley_tensor.shape[0]
    t_indices = torch.arange(1, T + 1, dtype=torch.float32, device=shapley_tensor.device)
    raw_weights = gamma ** (T - t_indices)
    weights = raw_weights / raw_weights.sum()  # shape (T,)

    # Weighted sum across time dimension T
    weighted_shapley = (shapley_tensor * weights.view(T, 1, 1)).sum(dim=0)
    return weighted_shapley


def main():
    print("--------------------------------------------------")
    print("KASPER Shapley Value Interpretability & Rule Extraction")
    print("--------------------------------------------------")

    weights_path = "best_kasper.pth"
    test_x_path = "data/spy_test_X.npy"
    train_x_path = "data/spy_train_X.npy"

    if not os.path.exists(weights_path):
        print(f"Error: Model weights file '{weights_path}' not found.")
        print("Please train the model first by running train.py.")
        return

    if not os.path.exists(test_x_path):
        print(f"Error: Required NumPy test dataset not found at '{test_x_path}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading test features...")
    X_test = np.load(test_x_path)
    print(f" - X_test shape: {X_test.shape} ([Batch, Features])")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    batch_size, num_inputs = X_test.shape
    num_regimes = 3

    model = KASPER(
        num_inputs=num_inputs,
        hidden_dim=64,
        num_regimes=num_regimes,
        grid_size=10,
        n_linear=3,
        n_cubic=2,
        dropout_rate=0.2,
        num_knots=8,
        sparsity_threshold=1e-3
    ).to(device)

    if os.path.exists(train_x_path):
        print("Fitting quantile knots using training features...")
        model.fit_knots(torch.tensor(np.load(train_x_path), dtype=torch.float32).to(device))
    else:
        model.fit_knots(X_test_tensor.to(device))

    print(f"Loading weights and normalization parameters from '{weights_path}'...")
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        y_std = float(checkpoint.get("y_std", 0.013237))
    else:
        model.load_state_dict(checkpoint)
        y_std = 0.013237

    model.eval()

    print("\nClassifying test set regimes using Layer 1 (deterministic, tau=0.3)...")
    with torch.no_grad():
        _, probs, _ = model.layer1(X_test_tensor.to(device), tau=0.3, deterministic=True)
    assigned_regimes = torch.argmax(probs, dim=-1).cpu()

    for r in range(num_regimes):
        count = torch.sum(assigned_regimes == r).item()
        print(f" - Samples assigned to Regime {r}: {count:3d} ({count/batch_size*100.1:.1f}%)")

    num_samples_to_process = min(50, batch_size)
    print(f"\nComputing Monte Carlo Shapley values for {num_samples_to_process} test samples...")
    print(f"Permutations per sample: 50 | Total features: {num_inputs}")
    all_shapley = []
    
    for i in range(num_samples_to_process):
        x_sample = X_test_tensor[i].to(device)
        shapley = estimate_shapley(model, x_sample, num_permutations=50, y_std=y_std)
        all_shapley.append(shapley.cpu())

    shapley_tensor = torch.stack(all_shapley, dim=0)

    # Apply Eq. 26 Temporal Weighting Scheme (gamma = 0.95)
    print("\nApplying Eq. 26 Temporal Weighting Scheme (gamma = 0.95)...")
    weighted_shapley_matrix = apply_temporal_weighting(shapley_tensor, gamma=0.95)

    print("\n" + "=" * 60)
    print("REGIME-SPECIFIC ATTRIBUTIONS & RULE EXTRACTION (Eq. 27, 28)")
    print("=" * 60)

    feature_names = FEATURE_NAMES[:num_inputs]

    for r in range(num_regimes):
        print(f"\n>>> REGIME {r} Rules:")
        regime_shapley = weighted_shapley_matrix[r, :]  # shape: (num_inputs,)

        sorted_idx = torch.argsort(torch.abs(regime_shapley), descending=True)

        print("-" * 50)
        print(f"   {'FEATURE NAME':25s} | {'SHAPLEY ATTRIBUTION':18s}")
        print("-" * 50)
        for rank in range(min(3, num_inputs)):
            feat_idx = sorted_idx[rank].item()
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
            feat_val = regime_shapley[feat_idx].item()
            print(f"   {feat_name:25s} | {feat_val:+.4e}")
        print("-" * 50)

        top_feats = [feature_names[sorted_idx[k].item()] for k in range(min(3, num_inputs))]
        print(f"   Symbolic Rule (Eq. 28): Regime {r} : {' + '.join(top_feats)} -> Y_{r}")
        print(f"                   1. {top_feats[0]} (major driver)")
        print(f"                   2. {top_feats[1]}")
        print(f"                   3. {top_feats[2]}")

    print("\n" + "=" * 60)
    print("Rule extraction completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
