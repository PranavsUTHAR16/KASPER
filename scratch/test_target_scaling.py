import torch
import torch.nn.functional as F
import numpy as np
from kasper import KASPER
from losses import l1_sparsity, unverified_regime_balance_penalty
from regime_detection import contrastive_loss

def test_target_scaling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = np.load("data/spy_train_X.npy")
    y_train = np.load("data/spy_train_y.npy")
    X_val = np.load("data/spy_val_X.npy")
    y_val = np.load("data/spy_val_y.npy")

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    model = KASPER(num_inputs=X_train.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
    model.fit_knots(torch.tensor(X_train, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"--- Training with Normalized Target (y_mean={y_mean:.6f}, y_std={y_std:.6f}) ---")

    for epoch in range(1, 101):
        model.train()
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            y_hat, p, z = model(x_b, tau=0.5)

            # Standard Huber loss on normalized target
            loss_huber = F.huber_loss(y_hat, y_b)
            loss_sparsity = 0.001 * l1_sparsity(model.layer1, model.layer2)
            loss_c = 0.10 * contrastive_loss(z, p)
            loss_b = 0.10 * unverified_regime_balance_penalty(p)
            loss_orth = 0.01 * model.layer2.orthogonality_loss()

            loss = loss_huber + loss_sparsity + loss_c + loss_b + loss_orth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_h_norm, p_val, _ = model(X_val_t, tau=0.3, deterministic=True)
            y_h_unnorm = y_h_norm.cpu().numpy() * y_std + y_mean

        pred_std = y_h_unnorm.std()
        eff_w = model.layer2.effective_weights()
        non_zero_w = (eff_w != 0).sum().item()
        corrs = [np.corrcoef(X_val[:, j], y_h_unnorm)[0, 1] for j in range(X_val.shape[1])]
        max_corr = max(abs(c) for c in corrs)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | NonZero Weights: {non_zero_w}/24 | Pred Std: {pred_std:.6e} | Max Corr: {max_corr:.4f}")

if __name__ == "__main__":
    test_target_scaling()
