import torch
import numpy as np
from kasper import KASPER

def inspect_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = np.load("data/spy_train_X.npy")
    X_val = np.load("data/spy_val_X.npy")

    model = KASPER(num_inputs=8, hidden_dim=64, num_regimes=3, num_knots=8).to(device)
    model.fit_knots(torch.tensor(X_train, dtype=torch.float32).to(device))
    model.load_state_dict(torch.load("best_kasper.pth", map_location=device))
    model.eval()

    eff_w = model.layer2.effective_weights().detach().cpu().numpy()
    theta = model.layer2.sparsity_threshold().detach().cpu().numpy()

    print("=" * 70)
    print("BEST CHECKPOINT LAYER 2 WEIGHTS AUDIT")
    print("=" * 70)
    print("Effective Thresholds θ per regime:", np.round(theta, 6))
    print("\nEffective Weight Matrix W (n_regimes=3, n_features=8):")
    print(np.round(eff_w, 6))

    non_zero_indices = np.where(eff_w != 0)
    print(f"\nActive (Non-Zero) Weight Positions (count = {len(non_zero_indices[0])}):")
    for r, j in zip(non_zero_indices[0], non_zero_indices[1]):
        print(f"  - Regime {r}, Feature {j}: weight = {eff_w[r, j]:+.6f}")

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_hat_norm, probs, _ = model(X_val_t, tau=0.3, deterministic=True)

    y_h = y_hat_norm.cpu().numpy()
    p = probs.cpu().numpy()

    print(f"\nValidation Predictions (Normalized):")
    print(f"  Mean:   {y_h.mean():+.6f}")
    print(f"  Std:    {y_h.std():.6e}")
    print(f"  Min:    {y_h.min():+.6f}")
    print(f"  Max:    {y_h.max():+.6f}")

    print(f"\nValidation Regime Probabilities Mean: {p.mean(axis=0)}")

if __name__ == "__main__":
    inspect_checkpoint()
