import numpy as np
import torch
from scipy.stats import binomtest
from kasper import KASPER

def compute_sharpe(returns):
    std = np.std(returns)
    if std <= 1e-12:
        return 0.0
    return (np.mean(returns) / std) * np.sqrt(252.0)

def block_bootstrap_sharpe(returns, block_length=15, num_bootstraps=10000, seed=42):
    np.random.seed(seed)
    n = len(returns)
    num_blocks = int(np.ceil(n / block_length))
    boot_sharpes = []

    for _ in range(num_bootstraps):
        # Sample block start indices uniformly
        start_indices = np.random.randint(0, n - block_length + 1, size=num_blocks)
        sampled_blocks = [returns[idx : idx + block_length] for idx in start_indices]
        sampled_returns = np.concatenate(sampled_blocks)[:n]
        boot_sharpes.append(compute_sharpe(sampled_returns))

    boot_sharpes = np.array(boot_sharpes)
    ci_90 = np.percentile(boot_sharpes, [5.0, 95.0])
    ci_95 = np.percentile(boot_sharpes, [2.5, 97.5])
    return boot_sharpes, ci_90, ci_95

def run_significance_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "best_kasper.pth"
    train_x_path = "data/spy_train_X.npy"
    train_y_path = "data/spy_train_y.npy"
    val_x_path = "data/spy_val_X.npy"
    val_y_path = "data/spy_val_y.npy"
    test_x_path = "data/spy_test_X.npy"
    test_y_path = "data/spy_test_y.npy"

    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path)
    y_val = np.load(val_y_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)

    model = KASPER(num_inputs=X_train.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
    model.fit_knots(torch.tensor(X_train, dtype=torch.float32).to(device))

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        y_mean = float(checkpoint.get("y_mean", y_train.mean()))
        y_std = float(checkpoint.get("y_std", y_train.std()))
    else:
        model.load_state_dict(checkpoint)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
    model.eval()

    def evaluate_split(label, X_data, y_actual):
        X_t = torch.tensor(X_data, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_hat_norm, probs, _ = model(X_t, tau=0.3, deterministic=True)

        y_h_norm = y_hat_norm.cpu().numpy()
        y_hat_unscaled = y_h_norm * y_std + y_mean
        positions = np.where(y_hat_unscaled > 0.0, 1.0, -1.0)
        strat_returns = positions * y_actual

        n_samples = len(y_actual)
        correct_hits = np.sum(np.sign(positions) == np.sign(y_actual))
        dir_acc = (correct_hits / n_samples) * 100.0

        # 1. Exact Binomial Test
        b_res = binomtest(correct_hits, n_samples, p=0.5, alternative='two-sided')
        p_val = b_res.pvalue

        # 2. Block-Bootstrap Sharpe CIs
        point_sharpe = compute_sharpe(strat_returns)
        boot_sharpes, ci_90, ci_95 = block_bootstrap_sharpe(strat_returns, block_length=15, num_bootstraps=10000)

        zero_in_90 = (ci_90[0] <= 0.0 <= ci_90[1])
        zero_in_95 = (ci_95[0] <= 0.0 <= ci_95[1])

        print("=" * 80)
        print(f"STATISTICAL SIGNIFICANCE REPORT: {label.upper()} (N = {n_samples})")
        print("=" * 80)
        print(f"1. Directional Accuracy (Hit Rate):")
        print(f"   - Hits: {correct_hits} / {n_samples} ({dir_acc:.2f}%)")
        print(f"   - Null Hypothesis H0: Chance hit rate p = 50.0%")
        print(f"   - Binomial Test p-value: {p_val:.4f}")
        print(f"   - Statistically Significant at α=0.05? {'YES (p < 0.05)' if p_val < 0.05 else 'NO (p >= 0.05 — Cannot reject H0)'}")

        print(f"\n2. Annualized Sharpe Ratio & Block Bootstrap CIs (10,000 resamples, block_len=15):")
        print(f"   - Point Estimate Sharpe: {point_sharpe:+.4f}")
        print(f"   - 90% Block Bootstrap CI: [{ci_90[0]:+.4f}, {ci_90[1]:+.4f}] (Zero in 90% CI? {zero_in_90})")
        print(f"   - 95% Block Bootstrap CI: [{ci_95[0]:+.4f}, {ci_95[1]:+.4f}] (Zero in 95% CI? {zero_in_95})")

        return {
            "label": label,
            "n": n_samples,
            "hits": correct_hits,
            "acc": dir_acc,
            "p_val": p_val,
            "sharpe": point_sharpe,
            "ci_90": ci_90,
            "ci_95": ci_95
        }

    res_val = evaluate_split("Validation Set", X_val, y_val)
    print()
    res_test = evaluate_split("Test Set", X_test, y_test)

if __name__ == "__main__":
    run_significance_tests()
