import numpy as np
import torch
import torch.nn.functional as F
from kasper import KASPER
from regime_detection import contrastive_loss
from losses import unverified_regime_balance_penalty

def compute_sharpe(returns):
    std = np.std(returns)
    if std <= 1e-12:
        return 0.0
    return (np.mean(returns) / std) * np.sqrt(252.0)

def evaluate_regime_metrics(model, X_eval, y_eval_actual, y_train_mean, y_train_std, device):
    model.eval()
    X_t = torch.tensor(X_eval, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_hat_norm, probs, _ = model(X_t, tau=0.3, deterministic=True)

    y_h_norm = y_hat_norm.cpu().numpy()
    p_np = probs.cpu().numpy()
    regimes = p_np.argmax(axis=1)
    y_h_unscaled = y_h_norm * y_train_std + y_train_mean

    # Effective Layer 2 Weights Audit
    eff_w = model.layer2.effective_weights().detach().cpu().numpy()
    n_regimes, n_features = eff_w.shape
    active_weights_count = int(np.sum(eff_w != 0))

    # Weight variation across regimes (std of w across regimes for each feature)
    weight_var_across_regimes = np.std(eff_w, axis=0).mean()

    # Per-regime prediction statistics
    regime_means = []
    regime_stds = []
    regime_counts = []
    regime_cond_ratios = []

    for r in range(n_regimes):
        mask = (regimes == r)
        count = int(mask.sum())
        regime_counts.append(count)
        if count > 0:
            r_mean = float(y_h_unscaled[mask].mean())
            r_std = float(y_h_unscaled[mask].std())
            cond_ratio = r_std / (abs(r_mean) + 1e-8)
        else:
            r_mean = 0.0
            r_std = 0.0
            cond_ratio = 0.0

        regime_means.append(r_mean)
        regime_stds.append(r_std)
        regime_cond_ratios.append(cond_ratio)

    regime_mean_spread = max(regime_means) - min(regime_means)

    return {
        "active_weights": active_weights_count,
        "weight_regime_var": weight_var_across_regimes,
        "regime_counts": regime_counts,
        "regime_means": regime_means,
        "regime_stds": regime_stds,
        "regime_mean_spread": regime_mean_spread,
        "regime_cond_ratios": regime_cond_ratios,
        "eff_w": eff_w
    }

def run_lever1_sparsity_sweep():
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

    lambdas = [0.001, 0.0005, 0.0002, 0.0001, 0.0]

    print("=" * 100)
    print("LEVER 1 EXPERIMENT: SPARSITY RELAXATION SWEEP ON VALIDATION & ROLLING WALK-FORWARD")
    print("=" * 100)

    for l_sp in lambdas:
        torch.manual_seed(42)
        model = KASPER(num_inputs=X_train.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
        model.fit_knots(torch.tensor(X_train, dtype=torch.float32).to(device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train_norm, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(1, 30):
            model.train()
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                y_hat, p, z = model(x_b, tau=0.5)

                loss_huber = F.huber_loss(y_hat, y_b)
                # Compute sparsity penalty
                w_l2 = torch.sum(torch.abs(model.layer2.effective_weights()))
                loss_sparsity = l_sp * w_l2
                loss_c = 0.05 * contrastive_loss(z, p)
                loss_b = 0.05 * unverified_regime_balance_penalty(p)
                loss_orth = 0.01 * model.layer2.orthogonality_loss()
                loss = loss_huber + loss_sparsity + loss_c + loss_b + loss_orth

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on Validation Set
        m_val = evaluate_regime_metrics(model, X_val, y_val, y_mean, y_std, device)

        # Evaluate Rolling Walk-Forward Backtest across 8 windows
        num_windows = 8
        wf_sharpes = []
        wf_dir_accs = []
        wf_regime_spreads = []

        for w_i in range(num_windows):
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
            m_w = KASPER(num_inputs=X_tr_w.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
            m_w.fit_knots(torch.tensor(X_tr_w, dtype=torch.float32).to(device))
            opt_w = torch.optim.AdamW(m_w.parameters(), lr=0.001, weight_decay=1e-5)
            ds_w = torch.utils.data.TensorDataset(
                torch.tensor(X_tr_w, dtype=torch.float32),
                torch.tensor(y_tr_w_norm, dtype=torch.float32)
            )
            ld_w = torch.utils.data.DataLoader(ds_w, batch_size=32, shuffle=True)

            for ep in range(1, 25):
                m_w.train()
                for xb, yb in ld_w:
                    xb, yb = xb.to(device), yb.to(device)
                    yh, pw, zw = m_w(xb, tau=0.5)
                    lh = F.huber_loss(yh, yb)
                    ls = l_sp * torch.sum(torch.abs(m_w.layer2.effective_weights()))
                    lc = 0.05 * contrastive_loss(zw, pw)
                    lb = 0.05 * unverified_regime_balance_penalty(pw)
                    lo = 0.01 * m_w.layer2.orthogonality_loss()
                    loss_w = lh + ls + lc + lb + lo
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

            # Measure regime mean spread in this window
            p_w_np = p_w.cpu().numpy().argmax(axis=1)
            w_means = [y_h_w_unscaled[p_w_np == r].mean() if (p_w_np == r).sum() > 0 else 0.0 for r in range(3)]
            wf_regime_spreads.append(max(w_means) - min(w_means))

        wf_sh = np.array(wf_sharpes)
        wf_acc = np.array(wf_dir_accs)
        wf_sp = np.array(wf_regime_spreads)

        print(f"\n>>> Sparsity Lambda λ_s = {l_sp:.4f}:")
        print(f"  - Active Weights:               {m_val['active_weights']} / 24")
        print(f"  - Inter-Regime Weight Variation: {m_val['weight_regime_var']:.6f}")
        print(f"  - Val Regime Counts:            {m_val['regime_counts']}")
        print(f"  - Val Regime Means (unscaled):  {[round(m, 6) for m in m_val['regime_means']]}")
        print(f"  - Val Regime Mean Spread:       {m_val['regime_mean_spread']:.6e}")
        print(f"  - Val Feature-Cond Ratio std/|m|: {[round(r, 4) for r in m_val['regime_cond_ratios']]}")
        print(f"  - Walk-Forward Mean Sharpe:     {wf_sh.mean():+.4f} (Std: {wf_sh.std():.4f}, Positive: {(wf_sh>0).sum()}/8)")
        print(f"  - Walk-Forward Mean Dir Acc:    {wf_acc.mean():.2f}% (Min: {wf_acc.min():.2f}%, Max: {wf_acc.max():.2f}%)")
        print(f"  - Walk-Forward Mean Regime Spread: {wf_sp.mean():.6e}")

if __name__ == "__main__":
    run_lever1_sparsity_sweep()
