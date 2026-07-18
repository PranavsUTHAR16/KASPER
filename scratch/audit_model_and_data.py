import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from kasper import KASPER

def audit_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "best_kasper.pth"
    train_x_path = "data/spy_train_X.npy"
    train_y_path = "data/spy_train_y.npy"
    val_x_path = "data/spy_val_X.npy"
    val_y_path = "data/spy_val_y.npy"
    test_x_path = "data/spy_test_X.npy"
    test_y_path = "data/spy_test_y.npy"

    print("=" * 80)
    print("STEP 3 AUDIT: GROUND-TRUTH ACTUAL RETURNS DISTRIBUTION & THRESHOLD SANITY CHECK")
    print("=" * 80)

    y_train_raw = np.load(train_y_path)
    y_val_raw = np.load(val_y_path)
    y_test_raw = np.load(test_y_path)

    # y_data in npy files is ALREADY raw unstandardized daily returns (e.g. +0.001 = +0.1%)
    actual_returns_train = y_train_raw
    actual_returns_val = y_val_raw
    actual_returns_test = y_test_raw

    def print_return_stats(label, ret_array):
        s = pd.Series(ret_array)
        std_val = ret_array.std()
        upper = 0.25 * std_val
        lower = -0.25 * std_val
        bullish = (ret_array > upper).sum()
        bearish = (ret_array < lower).sum()
        neutral = len(ret_array) - bullish - bearish

        print(f"\n--- {label} Ground-Truth Actual Returns Summary (N = {len(ret_array)}) ---")
        print(f"  Mean:   {s.mean():+.6f} | Std: {std_val:.6f}")
        print(f"  Min:    {s.min():+.6f} | 25%: {s.quantile(0.25):+.6f} | 50%: {s.median():+.6f} | 75%: {s.quantile(0.75):+.6f} | Max: {s.max():+.6f}")
        print(f"  Thresholds (+/- 0.25 * Std): Upper = {upper:+.6f}, Lower = {lower:+.6f}")
        print(f"  Directional Counts: Bullish = {bullish} ({bullish/len(ret_array)*100:.1f}%), Neutral = {neutral} ({neutral/len(ret_array)*100:.1f}%), Bearish = {bearish} ({bearish/len(ret_array)*100:.1f}%)")

    print_return_stats("TRAIN SET", actual_returns_train)
    print_return_stats("VAL SET", actual_returns_val)
    print_return_stats("TEST SET", actual_returns_test)

    # Check model if weights exist
    if not os.path.exists(weights_path):
        print(f"\nError: {weights_path} not found.")
        return

    X_test = np.load(test_x_path)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = KASPER(num_inputs=X_test.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
    model.fit_knots(torch.tensor(np.load(train_x_path), dtype=torch.float32).to(device))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_hat_scaled_t, probs_t, _ = model(X_test_tensor, tau=0.3, deterministic=True)

    y_hat_scaled = y_hat_scaled_t.cpu().numpy()
    y_hat_unscaled = y_hat_scaled
    actual_returns = actual_returns_test

    print("\n" + "=" * 80)
    print("STEP 1 AUDIT: MODEL PREDICTIONS COLLAPSE DIAGNOSTICS (TEST SET)")
    print("=" * 80)

    mean_pred = y_hat_unscaled.mean()
    std_pred = y_hat_unscaled.std()
    cov_pred = std_pred / (abs(mean_pred) + 1e-8)
    positive_ratio = (y_hat_unscaled > 0).mean() * 100.0

    print(f"  y_hat_unscaled Mean:                     {mean_pred:+.6e}")
    print(f"  y_hat_unscaled Std:                      {std_pred:.6e}")
    print(f"  Coefficient of Variation (std / |mean|): {cov_pred:.6f}")
    print(f"  Positive Predictions Ratio ((y_hat>0)):  {positive_ratio:.2f}%")

    print("\n  Feature-by-Feature Correlation against y_hat_unscaled:")
    feature_names_path = "data/selected_features.txt"
    if os.path.exists(feature_names_path):
        with open(feature_names_path) as f:
            feat_names = [line.strip() for line in f if line.strip()]
    else:
        feat_names = [f"Feature_{j}" for j in range(X_test.shape[1])]

    for j in range(X_test.shape[1]):
        feat_vals = X_test[:, j]
        corr = np.corrcoef(feat_vals, y_hat_unscaled)[0, 1] if std_pred > 1e-12 else 0.0
        print(f"    - {feat_names[j]:22s} : Pearson r = {corr:+.6f}")

    print("\n" + "=" * 80)
    print("STEP 2 AUDIT: NAIVE BASELINES COMPARISON (TEST SET)")
    print("=" * 80)

    def compute_metrics(positions, y_hat_vals, y_true):
        strategy_returns = positions * y_true
        direction_acc = (np.sign(positions) == np.sign(y_true)).mean() * 100.0
        cum_returns = (np.prod(1.0 + strategy_returns) - 1.0) * 100.0
        std_ret = np.std(strategy_returns)
        sharpe = (np.mean(strategy_returns) / (std_ret + 1e-8)) * np.sqrt(252.0) if std_ret > 0 else 0.0
        r2 = r2_score(y_true, y_hat_vals) if np.std(y_hat_vals) > 0 else 0.0
        mse = mean_squared_error(y_true, y_hat_vals)
        return direction_acc, cum_returns, sharpe, r2, mse

    # Baselines
    b_long_pos = np.ones_like(actual_returns)
    b_short_pos = -np.ones_like(actual_returns)
    b_zero_pred = np.zeros_like(actual_returns)
    b_train_mean_pred = np.full_like(actual_returns, actual_returns_train.mean())

    k_pos = np.sign(y_hat_unscaled)
    # Handle exact 0 in predictions by defaulting to long
    k_pos[k_pos == 0] = 1.0

    models_dict = {
        "Always-Long Baseline": (b_long_pos, b_train_mean_pred),
        "Always-Short Baseline": (b_short_pos, -b_train_mean_pred),
        "Zero-Predictor Baseline": (b_long_pos, b_zero_pred),
        "Train-Mean Baseline": (b_long_pos, b_train_mean_pred),
        "KASPER Model": (k_pos, y_hat_unscaled)
    }

    print(f"{'Model / Baseline':26s} | {'Dir Acc (%)':12s} | {'Cum Return (%)':15s} | {'Sharpe':10s} | {'R^2':10s} | {'MSE':10s}")
    print("-" * 95)
    for name, (pos, y_h) in models_dict.items():
        acc, cum_ret, sh, r2, mse = compute_metrics(pos, y_h, actual_returns)
        print(f"{name:26s} | {acc:11.2f}% | {cum_ret:14.2f}% | {sh:10.4f} | {r2:10.4f} | {mse:10.6f}")
    print("-" * 95)

if __name__ == "__main__":
    audit_model_and_data()
