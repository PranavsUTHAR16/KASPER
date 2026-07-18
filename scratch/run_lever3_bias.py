import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from kasper import KASPER
from regime_forecasting import RegimeAdaptiveForecastingLayer
from regime_detection import contrastive_loss
from losses import unverified_regime_balance_penalty

def compute_sharpe(returns):
    std = np.std(returns)
    if std <= 1e-12:
        return 0.0
    return (np.mean(returns) / std) * np.sqrt(252.0)

# Lever 3 Model Extension: Adding learnable per-regime bias b^(r) to Eq. 20
class KASPERWithBias(KASPER):
    def __init__(self, num_inputs=8, hidden_dim=64, num_regimes=3, grid_size=10, n_linear=3, n_cubic=2, dropout_rate=0.2, num_knots=8, sparsity_threshold=1e-3):
        super().__init__(num_inputs, hidden_dim, num_regimes, grid_size, n_linear, n_cubic, dropout_rate, num_knots, sparsity_threshold)
        # Learnable per-regime bias parameter b^(r)
        self.regime_bias = nn.Parameter(torch.zeros(num_regimes))

    def forward(self, x, tau=1.0, hard=False, deterministic=False):
        batch_size = x.shape[0]
        # Layer 1: Regime detection returns (embeddings, probs, logits)
        embeddings, probs, logits = self.layer1(x, tau=tau, hard=hard, deterministic=deterministic)

        # Layer 2: Regime-adaptive forecasts with per-regime bias b^(r)
        eff_w = self.layer2.effective_weights()
        n_regimes = self.layer2.n_regimes
        n_features = self.layer2.n_features
        forecast_per_regime = torch.zeros(batch_size, n_regimes, device=x.device)

        for r in range(n_regimes):
            phi_r = torch.zeros(batch_size, n_features, device=x.device)
            for j in range(n_features):
                phi_r[:, j] = self.layer2.splines[r][j](x[:, j])

            w_r = eff_w[r]
            # Eq. 20 with explicit per-regime bias term b^(r)
            forecast_per_regime[:, r] = self.regime_bias[r] + (phi_r * w_r.unsqueeze(0)).sum(dim=-1)

        # Weighted expectation across regimes
        y_hat = (forecast_per_regime * probs).sum(dim=-1)
        return y_hat, probs, embeddings

def run_lever3_bias_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = np.load("data/spy_train_X.npy")
    y_train = np.load("data/spy_train_y.npy")
    X_val = np.load("data/spy_val_X.npy")
    y_val = np.load("data/spy_val_y.npy")
    X_test = np.load("data/spy_test_X.npy")
    y_test = np.load("data/spy_test_y.npy")

    X_full = np.concatenate([X_train, X_val, X_test], axis=0)
    y_full = np.concatenate([y_train, y_val, y_test], axis=0)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_train_norm = (y_train - y_mean) / y_std

    print("=" * 100)
    print("LEVER 3 EXPERIMENT: EXPLICIT PER-REGIME BIAS TERM b^(r) WITH REGULARIZATION")
    print("=" * 100)

    for lambda_bias_var in [0.01, 0.05]:
        torch.manual_seed(42)
        model = KASPERWithBias(num_inputs=8, hidden_dim=64, num_regimes=3, num_knots=8).to(device)
        model.fit_knots(torch.tensor(X_train, dtype=torch.float32).to(device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train_norm, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(1, 12):
            model.train()
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                y_hat, p, z = model(x_b, tau=0.5)

                loss_huber = F.huber_loss(y_hat, y_b)
                loss_sparsity = 0.0005 * torch.sum(torch.abs(model.layer2.effective_weights()))
                # Penalty on variance of bias terms to prevent arbitrary divergence
                loss_bias_var = lambda_bias_var * torch.var(model.regime_bias)
                loss_c = 0.05 * contrastive_loss(z, p)
                loss_b = 0.05 * unverified_regime_balance_penalty(p)
                loss_orth = 0.01 * model.layer2.orthogonality_loss()
                loss = loss_huber + loss_sparsity + loss_bias_var + loss_c + loss_b + loss_orth

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on Validation Set
        model.eval()
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_hat_val_norm, p_val, _ = model(X_val_t, tau=0.3, deterministic=True)

        y_h_val_norm = y_hat_val_norm.cpu().numpy()
        y_h_val_unscaled = y_h_val_norm * y_std + y_mean
        reg_val = p_val.cpu().numpy().argmax(axis=1)

        learned_biases = model.regime_bias.detach().cpu().numpy()
        learned_biases_unscaled = learned_biases * y_std + y_mean

        val_means = [y_h_val_unscaled[reg_val == r].mean() if (reg_val == r).sum() > 0 else 0.0 for r in range(3)]
        val_stds = [y_h_val_unscaled[reg_val == r].std() if (reg_val == r).sum() > 0 else 0.0 for r in range(3)]

        # Ground Rule #2 Audit: Bias contribution vs feature variation ratio
        bias_contributions = learned_biases_unscaled
        feature_std_contributions = val_stds

        # Rolling walk-forward backtest (8 windows)
        wf_sharpes = []
        wf_dir_accs = []
        wf_bias_order_stable = []

        for w_i in range(8):
            train_end = 520 + w_i * 120
            eval_end = min(train_end + 180, len(X_full))

            X_tr_w = X_full[:train_end]
            y_tr_w = y_full[:train_end]
            X_ev_w = X_full[train_end:eval_end]
            y_ev_w = y_full[train_end:eval_end]

            y_tr_w_mean = float(y_tr_w.mean())
            y_tr_w_std = float(y_tr_w.std())
            y_tr_w_norm = (y_tr_w - y_tr_w_mean) / y_tr_w_std

            torch.manual_seed(42 + w_i)
            m_w = KASPERWithBias(num_inputs=8, hidden_dim=64, num_regimes=3, num_knots=8).to(device)
            m_w.fit_knots(torch.tensor(X_tr_w, dtype=torch.float32).to(device))

            opt_w = torch.optim.AdamW(m_w.parameters(), lr=0.001, weight_decay=1e-5)
            ds_w = torch.utils.data.TensorDataset(
                torch.tensor(X_tr_w, dtype=torch.float32),
                torch.tensor(y_tr_w_norm, dtype=torch.float32)
            )
            ld_w = torch.utils.data.DataLoader(ds_w, batch_size=32, shuffle=True)

            for ep in range(1, 10):
                m_w.train()
                for xb, yb in ld_w:
                    xb, yb = xb.to(device), yb.to(device)
                    yh, pw, zw = m_w(xb, tau=0.5)
                    lh = F.huber_loss(yh, yb)
                    ls = 0.0005 * torch.sum(torch.abs(m_w.layer2.effective_weights()))
                    lb_v = lambda_bias_var * torch.var(m_w.regime_bias)
                    lc = 0.05 * contrastive_loss(zw, pw)
                    lb = 0.05 * unverified_regime_balance_penalty(pw)
                    lo = 0.01 * m_w.layer2.orthogonality_loss()
                    loss_w = lh + ls + lb_v + lc + lb + lo
                    opt_w.zero_grad()
                    loss_w.backward()
                    opt_w.step()

            m_w.eval()
            X_ev_t = torch.tensor(X_ev_w, dtype=torch.float32).to(device)
            with torch.no_grad():
                y_hat_w_norm, p_w, _ = m_w(X_ev_t, tau=0.3, deterministic=True)

            y_h_w_norm = y_hat_w_norm.cpu().numpy()
            y_h_w_unscaled = y_h_w_norm * y_tr_w_std + y_tr_w_mean
            pos_w = np.where(y_h_w_unscaled > 0.0, 1.0, -1.0)
            ret_w = pos_w * y_ev_w

            wf_sharpes.append(compute_sharpe(ret_w))
            wf_dir_accs.append((np.sum(np.sign(pos_w) == np.sign(y_ev_w)) / len(y_ev_w)) * 100.0)

            # Check if bias signs flip across windows
            b_w = m_w.regime_bias.detach().cpu().numpy()
            wf_bias_order_stable.append(b_w)

        wf_sh = np.array(wf_sharpes)
        wf_acc = np.array(wf_dir_accs)
        wf_biases = np.array(wf_bias_order_stable)

        print(f"\n>>> Bias Penalty Lambda λ_b = {lambda_bias_var:.2f}:")
        print(f"  - Learned Regime Biases b^(r) (unscaled): {[round(b, 6) for b in learned_biases_unscaled]}")
        print(f"  - Val Feature-Cond Variation std(y|r):    {[round(s, 6) for s in val_stds]}")
        print(f"  - Cheating Audit (Bias / Feature Std):   {[round(abs(b)/(s+1e-8), 2) for b, s in zip(learned_biases_unscaled, val_stds)]}")
        print(f"  - Walk-Forward Mean Sharpe:              {wf_sh.mean():+.4f} (Std: {wf_sh.std():.4f}, Positive: {(wf_sh>0).sum()}/8)")
        print(f"  - Walk-Forward Mean Dir Acc:             {wf_acc.mean():.2f}% (Min: {wf_acc.min():.2f}%, Max: {wf_acc.max():.2f}%)")
        print(f"  - Walk-Forward Bias Std across windows:  {wf_biases.std(axis=0)}")

if __name__ == "__main__":
    run_lever3_bias_experiment()
