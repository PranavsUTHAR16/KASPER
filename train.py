import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from kasper import KASPER
from losses import KasperCompositeLoss

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
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)

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
        num_knots=5,
        sparsity_threshold=1e-3
    ).to(device)

    # Crucial Step: Before training starts, call model.fit_knots on training data
    print("Fitting quantile knots on full training input tensor...")
    model.fit_knots(X_tensor.to(device))

    # 5. Instantiate Composite Loss Function
    criterion = KasperCompositeLoss(
        lambda_s=0.001,
        lambda_c=0.01,
        lambda_o=0.01,
        lambda_b=0.05
    ).to(device)

    # 6. Optimizer & Scheduler Configuration
    # Table 1: AdamW with lr=0.001 and weight_decay=1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=7)

    # 7. Early Stopping Configuration
    best_val_loss = float('inf')
    early_stopping_patience = 15
    epochs_no_improve = 0
    max_epochs = 100

    # Temperature Annealing parameters
    tau_start = 2.0
    tau_end = 0.5
    tau_decay = (tau_end / tau_start) ** (1.0 / max_epochs)

    print(f"\nStarting training loop (max {max_epochs} epochs)...")
    print("-" * 80)

    for epoch in range(1, max_epochs + 1):
        # Calculate current tau temperature
        current_tau = max(tau_end, tau_start * (tau_decay ** (epoch - 1)))

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

            # Evaluate composite loss
            loss, components = criterion(
                predictions=y_hat,
                targets=y_batch,
                embeddings=embeddings,
                probs=probs,
                model=model
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Ensure gradients are flowing to Layer 1 (Regime Detection)
            if model.layer1.regime_proj.weight.grad is None:
                print("CRITICAL WARNING: No gradients flowing to Layer 1! p_i and z_i are detached.")
            else:
                grad_norm = model.layer1.regime_proj.weight.grad.norm().item()
                if grad_norm == 0.0:
                    print("WARNING: Layer 1 gradients are exactly 0.0. Learning has stopped.")

            # Gradient Clipping at max_norm=0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            for key in comp_sums:
                comp_sums[key] += components[key]

        train_loss /= len(train_loader)
        for key in comp_sums:
            comp_sums[key] /= len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_comp_sums = {"huber": 0.0, "sparsity": 0.0, "contrastive": 0.0, "orth": 0.0, "balance": 0.0}

        with torch.no_grad():
            for x_val_b, y_val_b in val_loader:
                x_val_b = x_val_b.to(device)
                y_val_b = y_val_b.to(device)

                # Differentiable forward pass
                # Pass decaying tau during validation to keep behavior aligned
                y_hat_v, probs_v, embeddings_v = model(x_val_b, tau=current_tau)

                loss_v, components_v = criterion(
                    predictions=y_hat_v,
                    targets=y_val_b,
                    embeddings=embeddings_v,
                    probs=probs_v,
                    model=model
                )

                val_loss += loss_v.item()
                for key in val_comp_sums:
                    val_comp_sums[key] += components_v[key]

        val_loss /= len(val_loader)
        for key in val_comp_sums:
            val_comp_sums[key] /= len(val_loader)

        # --- SPARSITY MONITOR REPORT (Equation 22) ---
        with torch.no_grad():
            w_raw = model.layer2.w
            theta = model.layer2.sparsity_threshold
            w_sparse = torch.sign(w_raw) * torch.relu(torch.abs(w_raw) - theta)
            total_weights = w_sparse.numel()
            pruned_weights = (w_sparse == 0).sum().item()
            sparsity_pct = (pruned_weights / total_weights) * 100
            print(f"Epoch Sparsity Report: {pruned_weights}/{total_weights} weights pruned ({sparsity_pct:.2f}%)")

        # --- LEARNING RATE SCHEDULER UPDATE ---
        scheduler.step(val_loss)

        # Print detailed statistics every 5 epochs
        if epoch == 1 or epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{max_epochs:3d} | LR: {current_lr:.6f} | Tau: {current_tau:.4f} | "
                  f"Train Loss: {train_loss:.4f} (Huber: {comp_sums['huber']:.4f}, Sparsity: {comp_sums['sparsity']:.1f}) | "
                  f"Val Loss: {val_loss:.4f} (Huber: {val_comp_sums['huber']:.4f}, Balance: {val_comp_sums['balance']:.4f})")

        # --- EARLY STOPPING & WEIGHT SAVING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model state dictionary
            torch.save(model.state_dict(), "best_kasper.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("-" * 80)
                print(f"Early stopping triggered at epoch {epoch}. No validation loss improvement for {early_stopping_patience} epochs.")
                print(f"Best Validation Loss achieved: {best_val_loss:.6f}")
                break

    print("-" * 80)
    print("Training process finished.")
    if os.path.exists("best_kasper.pth"):
        print("Best model weights successfully saved to 'best_kasper.pth'.")

if __name__ == "__main__":
    main()
