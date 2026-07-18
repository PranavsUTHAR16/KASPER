import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from kasper import KASPER
from regime_detection import contrastive_loss
from losses import unverified_regime_balance_penalty

def compute_sharpe(returns):
    std = np.std(returns)
    if std <= 1e-12:
        return 0.0
    return (np.mean(returns) / std) * np.sqrt(252.0)

def build_rich_features():
    df = pd.read_csv("data/spy_full_features.csv")

    # Compute additional leak-free technical indicators
    # 1. Multi-horizon realized vol (5-day)
    df["Vol_5d"] = df["Log_Return_1d"].rolling(5, min_periods=1).std().fillna(0.0)

    # 2. RSI (14-day)
    delta = df["Log_Return_1d"]
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df["RSI_14d"] = 100.0 - (100.0 / (1.0 + rs))

    # 3. MACD
    ema12 = df["Log_Return_1d"].ewm(span=12, adjust=False).mean()
    ema26 = df["Log_Return_1d"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # 4. Day-of-week seasonality (cos / sin)
    df["Date"] = pd.to_datetime(df["Date"])
    dow = df["Date"].dt.dayofweek
    df["DOW_Sin"] = np.sin(2 * np.pi * dow / 5.0)
    df["DOW_Cos"] = np.cos(2 * np.pi * dow / 5.0)

    feature_cols = [c for c in df.columns if c not in ["Date", "Target_Return_Next_Day"]]
    X_full = df[feature_cols].values
    y_full = df["Target_Return_Next_Day"].values

    return X_full, y_full, feature_cols

def run_lever2_feature_expansion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_full, y_full, feature_cols = build_rich_features()

    N_total = len(X_full)
    N_train = 1040
    N_val = 223

    X_train_raw = X_full[:N_train]
    y_train_raw = y_full[:N_train]

    X_val_raw = X_full[N_train : N_train + N_val]
    y_val_raw = y_full[N_train : N_train + N_val]

    print("=" * 100)
    print(f"LEVER 2 EXPERIMENT: RICHER FEATURE EXPANSION (Total Candidate Features: {len(feature_cols)})")
    print("=" * 100)

    for k_feats in [12, 16]:
        # Fit SelectKBest & StandardScaler strictly on train set (Zero lookahead)
        selector = SelectKBest(score_func=f_regression, k=k_feats)
        selector.fit(X_train_raw, y_train_raw)

        scaler = StandardScaler()
        X_train_k = scaler.fit_transform(selector.transform(X_train_raw))
        X_val_k = scaler.transform(selector.transform(X_val_raw))

        y_mean = float(y_train_raw.mean())
        y_std = float(y_train_raw.std())
        y_train_norm = (y_train_raw - y_mean) / y_std

        for l_sp in [0.0005, 0.0001]:
            torch.manual_seed(42)
            model = KASPER(num_inputs=k_feats, hidden_dim=64, num_regimes=3, num_knots=8).to(device)
            model.fit_knots(torch.tensor(X_train_k, dtype=torch.float32).to(device))

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train_k, dtype=torch.float32),
                torch.tensor(y_train_norm, dtype=torch.float32)
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

            for epoch in range(1, 15):
                model.train()
                for x_b, y_b in loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    y_hat, p, z = model(x_b, tau=0.5)

                    loss_huber = F.huber_loss(y_hat, y_b)
                    loss_sparsity = l_sp * torch.sum(torch.abs(model.layer2.effective_weights()))
                    loss_c = 0.05 * contrastive_loss(z, p)
                    loss_b = 0.05 * unverified_regime_balance_penalty(p)
                    loss_orth = 0.01 * model.layer2.orthogonality_loss()
                    loss = loss_huber + loss_sparsity + loss_c + loss_b + loss_orth

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Evaluate Validation Set
            model.eval()
            X_val_t = torch.tensor(X_val_k, dtype=torch.float32).to(device)
            with torch.no_grad():
                y_hat_val_norm, p_val, _ = model(X_val_t, tau=0.3, deterministic=True)

            y_h_val_norm = y_hat_val_norm.cpu().numpy()
            y_h_val_unscaled = y_h_val_norm * y_std + y_mean
            reg_val = p_val.cpu().numpy().argmax(axis=1)

            val_means = [y_h_val_unscaled[reg_val == r].mean() if (reg_val == r).sum() > 0 else 0.0 for r in range(3)]
            val_stds = [y_h_val_unscaled[reg_val == r].std() if (reg_val == r).sum() > 0 else 0.0 for r in range(3)]
            val_spread = max(val_means) - min(val_means)

            # Rolling walk-forward backtest (8 windows)
            wf_sharpes = []
            wf_dir_accs = []
            wf_spreads = []

            for w_i in range(8):
                train_end = 520 + w_i * 120
                eval_end = min(train_end + 180, N_total)

                X_tr_w_raw = X_full[:train_end]
                y_tr_w_raw = y_full[:train_end]
                X_ev_w_raw = X_full[train_end:eval_end]
                y_ev_w_raw = y_full[train_end:eval_end]

                sel_w = SelectKBest(score_func=f_regression, k=k_feats)
                sel_w.fit(X_tr_w_raw, y_tr_w_raw)

                sc_w = StandardScaler()
                X_tr_w_k = sc_w.fit_transform(sel_w.transform(X_tr_w_raw))
                X_ev_w_k = sc_w.transform(sel_w.transform(X_ev_w_raw))

                y_tr_w_mean = float(y_tr_w_raw.mean())
                y_tr_w_std = float(y_tr_w_raw.std())
                y_tr_w_norm = (y_tr_w_raw - y_tr_w_mean) / y_tr_w_std

                torch.manual_seed(42 + w_i)
                m_w = KASPER(num_inputs=k_feats, hidden_dim=64, num_regimes=3, num_knots=8).to(device)
                m_w.fit_knots(torch.tensor(X_tr_w_k, dtype=torch.float32).to(device))

                opt_w = torch.optim.AdamW(m_w.parameters(), lr=0.001, weight_decay=1e-5)
                ds_w = torch.utils.data.TensorDataset(
                    torch.tensor(X_tr_w_k, dtype=torch.float32),
                    torch.tensor(y_tr_w_norm, dtype=torch.float32)
                )
                ld_w = torch.utils.data.DataLoader(ds_w, batch_size=32, shuffle=True)

                for ep in range(1, 15):
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
                X_ev_t = torch.tensor(X_ev_w_k, dtype=torch.float32).to(device)
                with torch.no_grad():
                    y_hat_w_norm, p_w, _ = m_w(X_ev_t, tau=0.3, deterministic=True)

                y_h_w_norm = y_hat_w_norm.cpu().numpy()
                y_h_w_unscaled = y_h_w_norm * y_tr_w_std + y_tr_w_mean
                pos_w = np.where(y_h_w_unscaled > 0.0, 1.0, -1.0)
                ret_w = pos_w * y_ev_w_raw

                wf_sharpes.append(compute_sharpe(ret_w))
                wf_dir_accs.append((np.sum(np.sign(pos_w) == np.sign(y_ev_w_raw)) / len(y_ev_w_raw)) * 100.0)

                p_w_np = p_w.cpu().numpy().argmax(axis=1)
                w_means = [y_h_w_unscaled[p_w_np == r].mean() if (p_w_np == r).sum() > 0 else 0.0 for r in range(3)]
                wf_spreads.append(max(w_means) - min(w_means))

            wf_sh = np.array(wf_sharpes)
            wf_acc = np.array(wf_dir_accs)
            wf_sp = np.array(wf_spreads)

            print(f"\n>>> Features K = {k_feats:2d} | Sparsity λ_s = {l_sp:.4f}:")
            print(f"  - Val Regime Means (unscaled):  {[round(m, 6) for m in val_means]}")
            print(f"  - Val Regime Mean Spread:       {val_spread:.6e}")
            print(f"  - Walk-Forward Mean Sharpe:     {wf_sh.mean():+.4f} (Std: {wf_sh.std():.4f}, Positive: {(wf_sh>0).sum()}/8)")
            print(f"  - Walk-Forward Mean Dir Acc:    {wf_acc.mean():.2f}% (Min: {wf_acc.min():.2f}%, Max: {wf_acc.max():.2f}%)")
            print(f"  - Walk-Forward Mean Regime Spread: {wf_sp.mean():.6e}")

if __name__ == "__main__":
    run_lever2_feature_expansion()
