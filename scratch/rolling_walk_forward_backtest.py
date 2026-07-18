import numpy as np
import torch
from kasper import KASPER
from regime_detection import contrastive_loss
from losses import unverified_regime_balance_penalty
import torch.nn.functional as F

def compute_sharpe(returns):
    std = np.std(returns)
    if std <= 1e-12:
        return 0.0
    return (np.mean(returns) / std) * np.sqrt(252.0)

def run_rolling_walk_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full dataset
    X_train = np.load("data/spy_train_X.npy")
    y_train = np.load("data/spy_train_y.npy")
    X_val = np.load("data/spy_val_X.npy")
    y_val = np.load("data/spy_val_y.npy")
    X_test = np.load("data/spy_test_X.npy")
    y_test = np.load("data/spy_test_y.npy")

    # Combine chronologically into single master series
    X_full = np.concatenate([X_train, X_val, X_test], axis=0)
    y_full = np.concatenate([y_train, y_val, y_test], axis=0)

    N_total = len(X_full)
    eval_window_size = 180
    num_windows = (N_total - 520) // 120  # ~6 non-overlapping/rolling windows

    print("=" * 90)
    print("STEP 2: ROLLING WALK-FORWARD BACKTEST ACROSS FULL 2018-2023 DATASET")
    print(f"Total Available Samples: N = {N_total} (~5 years of daily SPY data)")
    print("=" * 90)

    results = []

    # Rolling window evaluation loop
    for i in range(num_windows):
        train_end = 520 + i * 120
        eval_end = min(train_end + eval_window_size, N_total)

        X_tr = X_full[:train_end]
        y_tr = y_full[:train_end]
        X_ev = X_full[train_end:eval_end]
        y_ev = y_full[train_end:eval_end]

        y_tr_mean = float(y_tr.mean())
        y_tr_std = float(y_tr.std())
        y_tr_norm = (y_tr - y_tr_mean) / y_tr_std

        # Train model from scratch on training window
        torch.manual_seed(42 + i)
        model = KASPER(num_inputs=X_tr.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
        model.fit_knots(torch.tensor(X_tr, dtype=torch.float32).to(device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr_norm, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(1, 25):  # Train to optimal early-stopping window
            model.train()
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                y_hat, p, z = model(x_b, tau=0.5)

                loss_huber = F.huber_loss(y_hat, y_b)
                loss_c = 0.05 * contrastive_loss(z, p)
                loss_b = 0.05 * unverified_regime_balance_penalty(p)
                loss_orth = 0.01 * model.layer2.orthogonality_loss()
                loss = loss_huber + loss_c + loss_b + loss_orth

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Out-of-sample evaluation on evaluation window
        model.eval()
        X_ev_t = torch.tensor(X_ev, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_hat_ev_norm, probs_ev, _ = model(X_ev_t, tau=0.3, deterministic=True)

        y_h_ev_norm = y_hat_ev_norm.cpu().numpy()
        # Documented Threshold Rule (matching evaluate.py):
        # Unscale model predictions: y_h_ev_unscaled = y_h_ev_norm * y_tr_std + y_tr_mean
        # Trading position: Long if y_h_ev_unscaled > 0.0, else Short (absolute return direction)
        y_h_ev_unscaled = y_h_ev_norm * y_tr_std + y_tr_mean
        positions = np.where(y_h_ev_unscaled > 0.0, 1.0, -1.0)
        strat_returns = positions * y_ev
        market_returns = y_ev

        n_ev = len(y_ev)
        hits = np.sum(np.sign(positions) == np.sign(y_ev))
        acc = (hits / n_ev) * 100.0
        sh_strat = compute_sharpe(strat_returns)
        sh_mkt = compute_sharpe(market_returns)
        cum_strat = (np.prod(1.0 + strat_returns) - 1.0) * 100.0
        cum_mkt = (np.prod(1.0 + market_returns) - 1.0) * 100.0

        w_name = f"Window {i+1} (Samples {train_end}-{eval_end})"
        results.append({
            "window": w_name,
            "train_n": train_end,
            "eval_n": n_ev,
            "acc": acc,
            "sharpe_strat": sh_strat,
            "sharpe_mkt": sh_mkt,
            "cum_strat": cum_strat,
            "cum_mkt": cum_mkt
        })

    print(f"{'Rolling Window':32s} | {'Eval N':7s} | {'Dir Acc (%)':12s} | {'Strat Sharpe':13s} | {'Market Sharpe':14s} | {'Strat Cum (%)':14s}")
    print("-" * 100)
    sharpes = []
    accs = []
    for r in results:
        sharpes.append(r["sharpe_strat"])
        accs.append(r["acc"])
        print(f"{r['window']:32s} | {r['eval_n']:7d} | {r['acc']:11.2f}% | {r['sharpe_strat']:+12.4f} | {r['sharpe_mkt']:+13.4f} | {r['cum_strat']:+13.2f}%")
    print("-" * 100)

    sharpes = np.array(sharpes)
    accs = np.array(accs)
    pos_windows = np.sum(sharpes > 0)

    print("\n" + "=" * 90)
    print("ROLLING WALK-FORWARD OUT-OF-SAMPLE DISTRIBUTION SUMMARY")
    print("=" * 90)
    print(f"Total Evaluated Windows:             {len(results)}")
    print(f"Positive Sharpe Windows:             {pos_windows} / {len(results)} ({pos_windows/len(results)*100:.1f}%)")
    print(f"Mean Out-of-Sample Sharpe:           {sharpes.mean():+.4f}")
    print(f"Std of Out-of-Sample Sharpe:          {sharpes.std():.4f}")
    print(f"Min / Max Out-of-Sample Sharpe:      {sharpes.min():+.4f} / {sharpes.max():+.4f}")
    print(f"Mean Out-of-Sample Directional Acc:  {accs.mean():.2f}%")
    print(f"Min / Max Out-of-Sample Dir Acc:     {accs.min():.2f}% / {accs.max():.2f}%")

if __name__ == "__main__":
    run_rolling_walk_forward()
