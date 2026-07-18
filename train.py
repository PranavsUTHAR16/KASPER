import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from kasper import KASPER
from losses import composite_loss

def main():
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Preprocessed SPY Data
    data_dir = "data"
    train_x_path = os.path.join(data_dir, "spy_train_X.npy")
    train_y_path = os.path.join(data_dir, "spy_train_y.npy")
    val_x_path = os.path.join(data_dir, "spy_val_X.npy")
    val_y_path = os.path.join(data_dir, "spy_val_y.npy")

    if not all(os.path.exists(p) for p in [train_x_path, train_y_path, val_x_path, val_y_path]):
        print("Error: Preprocessed NumPy data files not found in 'data/' directory.")
        print("Please run preprocess_spy.py first.")
        return

    print("Loading preprocessed SPY datasets...")
    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path)
    y_val = np.load(val_y_path)

    print(f" - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f" - X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")

    # 3. Standardize target values to match paper recommendations (Table 1)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    # Create DataLoaders
    # Compute training target normalization parameters (so Huber loss gradients balance L1 parameter sparsity)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    print(f"Training target normalization: mean = {y_mean:+.6f}, std = {y_std:.6f}")

    # Standardized target tensors for gradient balance
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    train_dataset = TensorDataset(X_tensor, torch.tensor(y_train_norm, dtype=torch.float32))
    val_dataset = TensorDataset(X_val_tensor, torch.tensor(y_val_norm, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 4. Instantiate Model & Fit Knots
    num_inputs = X_train.shape[1]

    print(f"\nInstantiating KASPER model with {num_inputs} input features...")
    model = KASPER(
        num_inputs=num_inputs,
        hidden_dim=64,
        num_regimes=3,
        grid_size=10,
        n_linear=3,
        n_cubic=2,
        dropout_rate=0.2,
        num_knots=8, # maps to n_basis in the new KASPER Layer 2 B-spline
        sparsity_threshold=1e-3
    ).to(device)

    # Crucial Step: Before training starts, call model.fit_knots on training data
    print("Fitting quantile knots on full training input tensor...")
    model.fit_knots(X_tensor.to(device))

    # 6. Training Setup — paper Table 1 exact spec
    # AdamW lr=0.001, weight_decay=1e-5, ReduceLROnPlateau factor=0.7, patience=7
    # Single joint optimizer from epoch 1 — no phase-based freezing.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=7)

    # 7. Early Stopping — patience=30 on validation Huber forecasting loss
    best_val_huber = float('inf')
    early_stopping_patience = 30
    epochs_no_improve = 0
    max_epochs = 100

    # Temperature Annealing parameters (Eq. 17: tau anneals over training)
    tau_start = 1.0
    tau_end = 0.3
    tau_decay = (tau_end / tau_start) ** (1.0 / max_epochs)

    print(f"Starting training loop (max {max_epochs} epochs, early-stop patience={early_stopping_patience})...")
    print("-" * 80)

    for epoch in range(1, max_epochs + 1):        # Calculate current tau temperature
        current_tau = max(tau_end, tau_start * (tau_decay ** (epoch - 1)))
        current_sparsity_lambda = 0.001  # Table 1: lambda_sparsity = 0.001

        # Balanced Loss Schedule with Normalized Target Training
        # lambda_c = 0.05, lambda_b = 0.05 maintains regime separation while allowing
        # normalized Huber loss gradients to dominate forecast head learning.
        current_lambda_c = 0.05
        current_lambda_b = 0.05

        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        # Loss component tracking for verbose logging
        comp_sums = {"huber": 0.0, "sparsity": 0.0, "contrastive": 0.0, "orth": 0.0, "balance": 0.0}

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward Pass: P_t^(r) (probs), z_i (embeddings), and prediction (y_hat)
            y_hat, probs, embeddings = model(x_batch, tau=current_tau)

            # Evaluate composite loss with diversity warm-up schedule
            loss_dict = composite_loss(
                y_hat=y_hat,
                y_true=y_batch,
                z=embeddings,
                p=probs,
                regime_ids=probs.argmax(dim=-1),
                kan1=model.layer1,
                kan2=model.layer2,
                include_balance=True,
                lambda_s=current_sparsity_lambda,
                lambda_c=current_lambda_c,
                lambda_b=current_lambda_b,
            )
            loss = loss_dict["total"]

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Ensure gradients are flowing to Layer 1 (Regime Detection)
            if model.layer1.to_logits.weight.grad is None:
                print("CRITICAL WARNING: No gradients flowing to Layer 1! p_i and z_i are detached.")
            else:
                grad_norm = model.layer1.to_logits.weight.grad.norm().item()
                if grad_norm == 0.0:
                    print("WARNING: Layer 1 gradients are exactly 0.0. Learning has stopped.")

            # Gradient Clipping at max_norm=0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            comp_sums["huber"] += loss_dict["huber"].item()
            comp_sums["sparsity"] += loss_dict["sparsity_l1"].item()
            comp_sums["contrastive"] += loss_dict["contrastive"].item()
            comp_sums["orth"] += loss_dict["orthogonal"].item()
            comp_sums["balance"] += loss_dict.get("balance_unverified", torch.tensor(0.0)).item()

        train_loss /= len(train_loader)
        for key in comp_sums:
            comp_sums[key] /= len(train_loader)

        # --- VALIDATION PHASE & STEP 1 DIAGNOSTICS ---
        model.eval()
        val_loss = 0.0
        val_comp_sums = {"huber": 0.0, "sparsity": 0.0, "contrastive": 0.0, "orth": 0.0, "balance": 0.0}
        all_val_logits = []
        all_val_probs = []

        with torch.no_grad():
            for x_val_b, y_val_b in val_loader:
                x_val_b = x_val_b.to(device)
                y_val_b = y_val_b.to(device)

                # Differentiable forward pass
                y_hat_v, probs_v, embeddings_v = model(x_val_b, tau=current_tau)
                _, _, logits_v = model.layer1(x_val_b, tau=current_tau)

                all_val_logits.append(logits_v.cpu())
                all_val_probs.append(probs_v.cpu())

                loss_dict_v = composite_loss(
                    y_hat=y_hat_v,
                    y_true=y_val_b,
                    z=embeddings_v,
                    p=probs_v,
                    regime_ids=probs_v.argmax(dim=-1),
                    kan1=model.layer1,
                    kan2=model.layer2,
                    include_balance=True,
                    lambda_s=current_sparsity_lambda,
                    lambda_c=current_lambda_c,
                    lambda_b=current_lambda_b,
                )
                loss_v = loss_dict_v["total"]

                val_loss += loss_v.item()
                val_comp_sums["huber"] += loss_dict_v["huber"].item()
                val_comp_sums["sparsity"] += loss_dict_v["sparsity_l1"].item()
                val_comp_sums["contrastive"] += loss_dict_v["contrastive"].item()
                val_comp_sums["orth"] += loss_dict_v["orthogonal"].item()
                val_comp_sums["balance"] += loss_dict_v.get("balance_unverified", torch.tensor(0.0)).item()

        val_loss /= len(val_loader)
        for key in val_comp_sums:
            val_comp_sums[key] /= len(val_loader)

        # Compute Step 1 Router Diagnostics (Logit Std and Mean Entropy)
        concat_val_logits = torch.cat(all_val_logits, dim=0)
        concat_val_probs = torch.cat(all_val_probs, dim=0)
        val_logit_std = concat_val_logits.std().item()
        val_entropy = (-concat_val_probs * torch.log(concat_val_probs.clamp_min(1e-8))).sum(dim=-1).mean().item()

        # --- SPARSITY MONITOR REPORT (Equation 22) ---
        with torch.no_grad():
            w_sparse = model.layer2.effective_weights()
            total_weights = w_sparse.numel()
            pruned_weights = (w_sparse == 0).sum().item()
            sparsity_pct = (pruned_weights / total_weights) * 100
            print(f"Epoch Sparsity Report: {pruned_weights}/{total_weights} weights pruned ({sparsity_pct:.2f}%)")

        # --- LEARNING RATE SCHEDULER UPDATE (on val_loss) ---
        scheduler.step(val_loss)

        # Phase label for logging
        phase_label = "EP"

        # Print detailed statistics every 5 epochs
        if epoch == 1 or epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{phase_label}] Epoch {epoch:3d}/{max_epochs:3d} | LR: {current_lr:.6f} | Tau: {current_tau:.4f} | "
                  f"LogitStd: {val_logit_std:.4f} | Entropy: {val_entropy:.4f} (max 1.0986) | "
                  f"Train: {train_loss:.4f} (H:{comp_sums['huber']:.4f} C:{comp_sums['contrastive']:.4f} O:{comp_sums['orth']:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (H:{val_comp_sums['huber']:.4f})")

        # --- AUTOMATED COLLAPSE CHECKS (Step 3) ---
        # Evaluate validation predictions for collapse assertion
        all_y_hat_v = []
        all_x_v = []
        for x_val_b, y_val_b in val_loader:
            x_val_b = x_val_b.to(device)
            with torch.no_grad():
                y_hat_v_b, _, _ = model(x_val_b, tau=0.3, deterministic=True)
                all_y_hat_v.append(y_hat_v_b.cpu())
                all_x_v.append(x_val_b.cpu())
        val_y_hat_np = torch.cat(all_y_hat_v, dim=0).numpy() * y_std + y_mean
        val_x_np = torch.cat(all_x_v, dim=0).numpy()

        val_pred_std = val_y_hat_np.std()
        max_feat_corr = max(
            abs(np.corrcoef(val_x_np[:, j], val_y_hat_np)[0, 1])
            if val_pred_std > 1e-12 else 0.0
            for j in range(val_x_np.shape[1])
        )

        # --- EARLY STOPPING & WEIGHT SAVING (based on validation Huber forecasting error) ---
        # Note: Must use val_huber (not composite loss) so L1 parameter decay doesn't trick early stopping.
        val_huber = val_comp_sums["huber"]
        if val_huber < best_val_huber:
            # Step 3 Assertions: Must pass both Router Collapse & Forecast Collapse checks before saving checkpoint
            is_valid_checkpoint = True
            if val_entropy >= 1.0:
                is_valid_checkpoint = False
                if epoch % 5 == 0:
                    print(f"  [CHECKPOINT REJECTED] Router entropy = {val_entropy:.4f} >= 1.0 (Router collapsed)")

            if val_pred_std <= 1e-6:
                is_valid_checkpoint = False
                if epoch % 5 == 0:
                    print(f"  [CHECKPOINT REJECTED] Forecast std = {val_pred_std:.6e} <= 1e-6 (Forecast head collapsed)")

            if max_feat_corr <= 0.02:
                is_valid_checkpoint = False
                if epoch % 5 == 0:
                    print(f"  [CHECKPOINT REJECTED] Max feature correlation = {max_feat_corr:.4f} <= 0.02 (No feature conditioning)")

            if is_valid_checkpoint:
                best_val_huber = val_huber
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_kasper.pth")
                print(f"  --> Saved new best checkpoint at Epoch {epoch:3d} (Val Huber: {val_huber:.6f}, Pred Std: {val_pred_std:.6e}, Max Corr: {max_feat_corr:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("-" * 80)
                print(f"Early stopping triggered at epoch {epoch}. "
                      f"No val_huber improvement for {early_stopping_patience} epochs.")
                print(f"Best Val Huber achieved: {best_val_huber:.6f}")
                break

    print("-" * 80)
    print("Training process finished.")
    if os.path.exists("best_kasper.pth"):
        print("Best model weights successfully saved to 'best_kasper.pth'.")

if __name__ == "__main__":
    main()
