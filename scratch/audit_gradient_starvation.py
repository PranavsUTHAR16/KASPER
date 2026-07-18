import torch
import numpy as np
import os
from kasper import KASPER
import torch.nn.functional as F
from kasper import KASPER
from losses import (
    l1_sparsity,
    unverified_regime_balance_penalty
)
from regime_detection import contrastive_loss

def audit_gradient_starvation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "best_kasper.pth"
    train_x_path = "data/spy_train_X.npy"
    train_y_path = "data/spy_train_y.npy"
    val_x_path = "data/spy_val_X.npy"
    val_y_path = "data/spy_val_y.npy"

    print("=" * 80)
    print("STEP 1.1 AUDIT: CHECKPOINT WEIGHT & REFINEMENT HEAD INSPECTION")
    print("=" * 80)

    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path)
    y_val = np.load(val_y_path)

    model = KASPER(num_inputs=X_train.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
    model.fit_knots(torch.tensor(X_train, dtype=torch.float32).to(device))

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found.")

    model.eval()

    # 1. Inspect Layer 2 Effective Weights & Pruning State
    eff_w = model.layer2.effective_weights()
    raw_w = model.layer2.weights
    theta_raw = model.layer2.theta_raw.detach().cpu().numpy()
    theta = model.layer2.sparsity_threshold().detach().cpu().numpy()

    zero_count = (eff_w == 0).sum().item()
    total_count = eff_w.numel()
    sparsity_pct = (zero_count / total_count) * 100.0

    print(f"\n  Layer 2 Weights Shape:           {eff_w.shape}")
    print(f"  Theta Raw:                      {np.array2string(theta_raw, precision=4)}")
    print(f"  Effective Sparsity Threshold θ:  {np.array2string(theta, precision=4)}")
    print(f"  Pruned Weights (eff_w == 0):     {zero_count} / {total_count} ({sparsity_pct:.2f}%)")
    print(f"  Effective Weights Max:           {eff_w.abs().max().item():.6e}")
    print(f"  Effective Weights Mean:          {eff_w.abs().mean().item():.6e}")

    # 2. Inspect Layer 2 Weights State
    print(f"\n  Non-Zero Forecast Weights: {(eff_w != 0).sum().item()} / {eff_w.numel()}")

    # Evaluate validation prediction value
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_hat_v, probs_v, z_v = model(X_val_t, tau=0.3, deterministic=True)
        val_pred_mean = y_hat_v.mean().item()
        val_pred_std = y_hat_v.std().item()

    print(f"\n  Validation Predictions: Mean = {val_pred_mean:+.6f}, Std = {val_pred_std:.6e}")

    print("\n" + "=" * 80)
    print("STEP 1.2 AUDIT: PER-LOSS GRADIENT NORM COMPARISON ON LAYER 2 & REFINEMENT")
    print("=" * 80)

    model.train()
    model.zero_grad()

    x_b = torch.tensor(X_train[:64], dtype=torch.float32).to(device)
    y_b = torch.tensor(y_train[:64], dtype=torch.float32).to(device)

    y_hat, p, z = model(x_b, tau=0.5)

    # Individual loss components
    loss_huber = F.huber_loss(y_hat, y_b)
    loss_sparsity = 0.001 * l1_sparsity(model.layer1, model.layer2)
    loss_contrastive = 0.50 * contrastive_loss(z, p)
    loss_orth = 0.01 * model.layer2.orthogonality_loss()
    loss_balance = 0.50 * unverified_regime_balance_penalty(p)

    components = {
        "Huber (Forecast)": loss_huber,
        "Sparsity (L1)": loss_sparsity,
        "Contrastive (L1)": loss_contrastive,
        "Orthogonality (L2)": loss_orth,
        "Regime Balance": loss_balance,
    }

    def compute_grad_norms(loss_tensor):
        model.zero_grad()
        loss_tensor.backward(retain_graph=True)

        w_grad = model.layer2.weights.grad
        w_grad_norm = w_grad.norm().item() if w_grad is not None else 0.0

        router_grad = model.layer1.to_logits.weight.grad
        router_grad_norm = router_grad.norm().item() if router_grad is not None else 0.0

        return w_grad_norm, router_grad_norm

    print(f"{'Loss Component':22s} | {'Loss Value':12s} | {'Grad Norm (KAN2.weights)':24s} | {'Grad Norm (Router)':18s}")
    print("-" * 80)
    for name, loss_tensor in components.items():
        w_gn, r_gn = compute_grad_norms(loss_tensor)
        print(f"{name:22s} | {loss_tensor.item():12.6f} | {w_gn:24.6e} | {r_gn:18.6e}")
    print("-" * 80)

if __name__ == "__main__":
    audit_gradient_starvation()
