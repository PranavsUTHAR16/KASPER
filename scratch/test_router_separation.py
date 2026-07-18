import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kasper import KASPER
from losses import composite_loss

def test_router_dynamics(lambda_c=1.0, lambda_b=1.0, max_epochs=30):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load preprocessed SPY dataset
    X_train = torch.tensor(np.load("data/spy_train_X.npy"), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.load("data/spy_train_y.npy"), dtype=torch.float32).to(device)

    model = KASPER(num_inputs=X_train.shape[1], hidden_dim=64, num_regimes=3, num_knots=8).to(device)
    model.fit_knots(X_train)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"\n--- Testing Router Dynamics (lambda_c={lambda_c}, lambda_b={lambda_b}) ---")

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_b, y_b in loader:
            y_hat, probs, z = model(x_b, tau=0.5)
            # Pass probs.detach() to layer2 to prevent Huber loss from collapsing router logits
            y_hat_detached, _, _ = model.layer2(x_b, probs.detach(), z=z)
            loss_dict = composite_loss(
                y_hat=y_hat_detached, y_true=y_b, z=z, p=probs, regime_ids=probs.argmax(-1),
                kan1=model.layer1, kan2=model.layer2, include_balance=True,
                lambda_s=0.001, lambda_c=lambda_c, lambda_b=lambda_b
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        # Check deterministic evaluation routing on full X_train
        model.eval()
        with torch.no_grad():
            z_eval, probs_eval, logits_eval = model.layer1(X_train, tau=0.3, deterministic=True)

        logit_std = logits_eval.std().item()
        max_probs = probs_eval.max(dim=-1)[0]
        mean_conf = max_probs.mean().item()
        counts = torch.bincount(probs_eval.argmax(dim=-1), minlength=3).tolist()
        entropy = (-probs_eval * torch.log(probs_eval.clamp_min(1e-8))).sum(-1).mean().item()

        if epoch == 1 or epoch % 5 == 0 or epoch == max_epochs:
            print(f"Epoch {epoch:2d} | LogitStd: {logit_std:.4f} | MeanConf: {mean_conf:.4f} | "
                  f"Entropy: {entropy:.4f} | Counts: {counts}")

if __name__ == "__main__":
    test_router_dynamics(lambda_c=1.0, lambda_b=1.0)
    test_router_dynamics(lambda_c=1.0, lambda_b=3.0)
    test_router_dynamics(lambda_c=1.0, lambda_b=5.0)
