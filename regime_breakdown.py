import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from kasper import KASPER

def run_regime_breakdown(dataset_type="test"):
    """
    Computes Fig. 3 / Fig. 8-style regime x market direction x confidence breakdown.

    - Market Direction Bins (Ground truth target return):
        Bullish: actual_return > +0.25 * std(actual_returns)
        Bearish: actual_return < -0.25 * std(actual_returns)
        Neutral: otherwise
    - Confidence Bins (Deterministic softmax max_prob with tau=0.3):
        High Confidence: max_prob > 0.6
        Low Confidence:  max_prob <= 0.6
    """
    print("=" * 75)
    print(f"KASPER REGIME & CONFIDENCE BREAKDOWN ({dataset_type.upper()} SET)")
    print("=" * 75)

    weights_path = "best_kasper.pth"
    train_x_path = "data/spy_train_X.npy"
    train_y_path = "data/spy_train_y.npy"

    if dataset_type == "train":
        x_path = "data/spy_train_X.npy"
        y_path = "data/spy_train_y.npy"
    else:
        x_path = "data/spy_test_X.npy"
        y_path = "data/spy_test_y.npy"

    if not os.path.exists(weights_path) or not os.path.exists(x_path):
        print("Error: Model weights or data file not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load evaluation dataset (y_data is raw percentage returns)
    X_data = np.load(x_path)
    y_data = np.load(y_path)
    actual_returns = y_data

    N = len(actual_returns)
    print(f"Loaded {dataset_type} dataset: N = {N} samples.")

    # 3. Model Inference (Deterministic mode with tau=0.3)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    model = KASPER(num_inputs=X_data.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)

    if os.path.exists(train_x_path):
        model.fit_knots(torch.tensor(np.load(train_x_path), dtype=torch.float32).to(device))
    else:
        print(f"Loading weights from '{weights_path}'...")
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        _, probs_tensor, _ = model(X_tensor.to(device), tau=0.3, deterministic=True)
    
    probs = probs_tensor.cpu().numpy()  # shape (N, 3)
    regimes = np.argmax(probs, axis=1)  # shape (N,)
    max_probs = probs.max(axis=1)       # shape (N,)

    print(f"Deterministic Softmax (tau=0.3) Confidence Summary:")
    print(f" - Mean Max Prob:   {np.mean(max_probs):.4f}")
    print(f" - Median Max Prob: {np.median(max_probs):.4f}")
    print(f" - Min / Max Prob:  {np.min(max_probs):.4f} / {np.max(max_probs):.4f}")

    # 4. Define Market Direction Bins (Ground truth actual_returns)
    ret_std = np.std(actual_returns)
    upper_thresh = 0.25 * ret_std
    lower_thresh = -0.25 * ret_std

    directions = []
    for r_val in actual_returns:
        if r_val > upper_thresh:
            directions.append("Bullish")
        elif r_val < lower_thresh:
            directions.append("Bearish")
        else:
            directions.append("Neutral")
    directions = np.array(directions)

    # 5. Define Confidence Bins (max_prob threshold = 0.6)
    conf_labels = np.where(max_probs > 0.6, "High_Conf (>0.6)", "Low_Conf (<=0.6)")

    # 6. Build 18-cell Cross-Tabulation Table (Regime x Direction x Confidence)
    regime_list = [0, 1, 2]
    direction_list = ["Bearish", "Neutral", "Bullish"]
    conf_list = ["Low_Conf (<=0.6)", "High_Conf (>0.6)"]

    records = []
    high_conf_total = 0
    low_conf_total = 0

    for r in regime_list:
        for d in direction_list:
            for c in conf_list:
                mask = (regimes == r) & (directions == d) & (conf_labels == c)
                count = np.sum(mask)
                pct = (count / N) * 100.0
                records.append({
                    "Regime": f"Regime {r}",
                    "Market Direction": d,
                    "Confidence Level": c,
                    "Count": count,
                    "Share (%)": f"{pct:.2f}%"
                })
                if c.startswith("High"):
                    high_conf_total += count
                else:
                    low_conf_total += count

    df_table = pd.DataFrame(records)

    print("\n" + "─" * 75)
    print("18-CELL CROSS-TABULATION TABLE (Fig. 3 Format)")
    print("─" * 75)
    print(df_table.to_string(index=False))
    print("─" * 75)

    # 7. Summary Diagnostic (Comparing against Fig. 3 vs Fig. 8 Collapse Mode)
    low_conf_pct = (low_conf_total / N) * 100.0
    high_conf_pct = (high_conf_total / N) * 100.0
    neutral_pct = (np.sum(directions == "Neutral") / N) * 100.0

    print("\n" + "=" * 75)
    print("REGIME SEPARATION DIAGNOSTIC SUMMARY")
    print("=" * 75)
    print(f"Total High-Confidence Samples (>0.6) : {high_conf_total} ({high_conf_pct:.2f}%)")
    print(f"Total Low-Confidence Samples (<=0.6) : {low_conf_total} ({low_conf_pct:.2f}%)")
    print(f"Active Regimes Represented           : {len(np.unique(regimes))}/3")

    print("\nDIAGNOSTIC PATTERN MATCH:")
    if low_conf_pct > 80.0:
        print("  [ALERT] MATCHES FIG. 8 ABLATION FAILURE / COLLAPSE MODE")
        print(f"   - {low_conf_pct:.1f}% of samples sit in the Low-Confidence tier (max_prob <= 0.6).")
        print("   - Softmax probabilities remain near uniform (~0.33 per regime), indicating that")
        print("     Layer 1 (RegimeDetectionLayer) is NOT separating latent space clusters cleanly.")
        print("   - Evaluation at low tau=0.3 reveals regime logits are nearly equal, meaning previous")
        print("     regime assignments were driven primarily by random Gumbel noise sampling.")
    else:
        print("  [SUCCESS] MATCHES FIG. 3 WELL-SEPARATED REGIME DISTRIBUTION")
        print(f"   - {high_conf_pct:.1f}% of samples show confident regime routing (>0.6).")
    print("=" * 75)


if __name__ == "__main__":
    print("\nRunning Regime Breakdown on TEST set...")
    run_regime_breakdown(dataset_type="test")
    print("\nRunning Regime Breakdown on TRAIN set...")
    run_regime_breakdown(dataset_type="train")
