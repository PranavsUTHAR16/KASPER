import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from kasper import KASPER

# All 33 candidate features from preprocessed SPY dataset
FEATURE_NAMES = [
    "HL_Spread", "OC_Spread", "Log_Return_1d", "Log_Return_7d", 
    "Log_Return_High_1d", "Log_Return_Low_1d", "Log_Return_Open_1d", 
    "Log_Return_Volume_1d", "Rolling_Volatility_21d", "Volatility_Ratio_21d", 
    "ATR_21d", "Velocity", "Acceleration", "Delta_Volume", "Volume_State_Ratio",
    "Volatility_Regime_63d", "Momentum_Regime_63d", "Acceleration_Regime_63d",
    "DOW_1", "DOW_2", "DOW_3", "DOW_4",
    "Month_2", "Month_3", "Month_4", "Month_5", "Month_6",
    "Month_7", "Month_8", "Month_9", "Month_10", "Month_11", "Month_12"
]

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


def estimate_shapley(model, x, num_permutations=50):
    """
    Computes permutation-based Monte Carlo Shapley values for all inputs of a single sample.
    Evaluates the progressive feature subsets in parallel for speed.
    
    Args:
        model (nn.Module): The KASPER master model.
        x (torch.Tensor): Input sample of shape (num_features,).
        num_permutations (int): Number of random permutations to sample.
        
    Returns:
        torch.Tensor: Shapley values of shape (num_regimes, num_features).
    """
    D = x.shape[0]  # 33 features
    
    # Initialize Shapley values buffer
    shapley_values = torch.zeros(model.layer2.n_regimes, D, device=x.device)
    
    for m in range(num_permutations):
        # Sample a random permutation of feature indices
        perm = torch.randperm(D, device=x.device)
        
        # Build batch of progressive subsets of shape (D + 1, D)
        # batch_inputs[k] represents the coalition with the first k features in the permutation active
        batch_inputs = torch.zeros(D + 1, D, device=x.device)
        for k in range(1, D + 1):
            batch_inputs[k, :] = batch_inputs[k - 1, :].clone()
            batch_inputs[k, perm[k - 1]] = x[perm[k - 1]]
            
        # Evaluate model forecasts for all regimes: shape (D + 1, num_regimes)
        y_hat = get_regime_forecasts(model, batch_inputs)
        
        # Compute marginal contributions (difference between step k and k-1)
        # diffs shape: (D, num_regimes)
        diffs = y_hat[1:] - y_hat[:-1]
        
        # Accumulate diffs into the corresponding permutation indices:
        shapley_values.scatter_add_(1, perm.unsqueeze(0).expand(model.layer2.n_regimes, -1), diffs.t())
        
    # Average across all permutations
    shapley_values /= num_permutations
    return shapley_values


def main():
    print("--------------------------------------------------")
    print("KASPER Shapley Value Interpretability & Rule Extraction")
    print("--------------------------------------------------")

    # Paths
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

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load test set
    print("Loading test features...")
    X_test = np.load(test_x_path)
    print(f" - X_test shape: {X_test.shape} ([Batch, Features])")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 3. Instantiate and Load KASPER Model
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
        num_knots=8, # maps to n_basis in new layer
        sparsity_threshold=1e-3
    ).to(device)

    # Fit knots using training data to maintain identical scaling bounds
    if os.path.exists(train_x_path):
        print("Fitting quantile knots using training features...")
        model.fit_knots(torch.tensor(np.load(train_x_path), dtype=torch.float32).to(device))
    else:
        model.fit_knots(X_test_tensor.to(device))

    print(f"Loading weights from '{weights_path}'...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # Disable sparsity thresholding during explanation by filling theta_raw with a large negative value
    model.layer2.theta_raw.data.fill_(-100.0)
    model.eval()

    # 4. Get assigned regimes from Layer 1 classification
    print("\nClassifying test set regimes using Layer 1...")
    with torch.no_grad():
        _, probs, _ = model.layer1(X_test_tensor.to(device), tau=1.0)
    assigned_regimes = torch.argmax(probs, dim=-1).cpu()  # Shape: (B,)
    
    # Print sample distribution across regimes
    for r in range(num_regimes):
        count = torch.sum(assigned_regimes == r).item()
        print(f" - Samples assigned to Regime {r}: {count:3d} ({count/batch_size*100.1:.1f}%)")

    # 5. Compute Shapley Values for a subset of test samples for performance
    num_samples_to_process = min(30, batch_size)
    print(f"\nComputing Monte Carlo Shapley values for {num_samples_to_process} test samples...")
    print(f"Permutations per sample: 50 | Total features: {num_inputs}")
    all_shapley = []
    
    for i in range(num_samples_to_process):
        if (i + 1) % 10 == 0 or i == 0:
            print(f" - Processing sample {i+1:3d}/{num_samples_to_process:3d}...")
        x_sample = X_test_tensor[i].to(device)
        # Get Shapley values shape: (num_regimes, num_inputs)
        shapley = estimate_shapley(model, x_sample, num_permutations=50)
        all_shapley.append(shapley.cpu())

    # Stack to tensor: shape (num_samples, num_regimes, num_inputs)
    shapley_tensor = torch.stack(all_shapley, dim=0)

    # 7. Aggregate attributions by assigned regime
    print("\n" + "=" * 60)
    print("REGIME-SPECIFIC ATTRIBUTIONS & RULE EXTRACTION")
    print("=" * 60)

    assigned_regimes_subset = assigned_regimes[:num_samples_to_process]

    for r in range(num_regimes):
        indices = (assigned_regimes_subset == r).nonzero(as_tuple=True)[0]
        
        print(f"\n>>> REGIME {r} Rules:")
        if len(indices) == 0:
            print("   No test samples assigned to this regime. Skipping rule extraction.")
            continue
            
        # Select Shapley values for samples assigned to regime r: shape (count, num_inputs)
        regime_attributions = shapley_tensor[indices, r, :]
        # Average across all assigned samples to get global feature importance vector
        mean_attribution = torch.mean(regime_attributions, dim=0)  # Shape: (num_inputs,)

        # Sort features by absolute contribution to extract top 3 rules
        sorted_idx = torch.argsort(torch.abs(mean_attribution), descending=True)

        print(f"   Sample size: {len(indices)}")
        print("   Top 3 Most Influential Features:")
        print("-" * 50)
        print(f"   {'FEATURE NAME':25s} | {'ATTRIBUTION VALUE':18s}")
        print("-" * 50)
        for rank in range(3):
            feat_idx = sorted_idx[rank].item()
            feat_name = FEATURE_NAMES[feat_idx]
            feat_val = mean_attribution[feat_idx].item()
            print(f"   {feat_name:25s} | {feat_val:+.4e}")
        print("-" * 50)

        # Output readable rules
        top_feats = [FEATURE_NAMES[sorted_idx[k].item()] for k in range(3)]
        print(f"   Rule Statement: Regime {r} pricing dynamics are primarily driven by:")
        print(f"                   1. {top_feats[0]} (major driver)")
        print(f"                   2. {top_feats[1]}")
        print(f"                   3. {top_feats[2]}")

    print("\n" + "=" * 60)
    print("Rule extraction completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
