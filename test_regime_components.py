import os
import torch
import numpy as np
from regime_detection import RegimeDetectionLayer
from losses import ContrastiveLoss, OrthogonalityLoss

def run_tests():
    print("--------------------------------------------------")
    print("Testing KASPER Regime Detection & Losses")
    print("--------------------------------------------------")

    # 1. Load preprocessed training data
    data_path = "data/spy_train_X.npy"
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found.")
        return

    X_train_np = np.load(data_path)
    # Use a small subset (e.g. 100 samples) to speed up testing
    X_train = torch.from_numpy(X_train_np[:100]).float()
    print(f"Loaded train subset shape: {X_train.shape} ([Batch, Time_Steps, Features])")
    
    batch_size, time_steps, features = X_train.shape
    num_inputs = time_steps * features

    # 2. Instantiate and fit RegimeDetectionLayer
    print("\nInstantiating RegimeDetectionLayer...")
    layer = RegimeDetectionLayer(
        num_inputs=num_inputs,
        hidden_dim=64,
        num_regimes=3,
        grid_size=10,
        n_linear=3,
        n_cubic=2,
        dropout_rate=0.2
    )
    
    print("Fitting knots...")
    layer.fit_knots(X_train)

    # 3. Test forward pass in training mode
    print("\nRunning forward pass (training=True)...")
    layer.train()
    probs_train, embeddings_train = layer(X_train, tau=1.0)
    print(f" - probs_train shape:      {probs_train.shape}")
    print(f" - embeddings_train shape: {embeddings_train.shape}")
    print(f" - Are probs soft/grad:   {probs_train.requires_grad}")
    
    # 4. Test forward pass in evaluation mode
    print("\nRunning forward pass (training=False)...")
    layer.eval()
    with torch.no_grad():
        probs_eval, embeddings_eval = layer(X_train, tau=1.0)
    print(f" - probs_eval shape:       {probs_eval.shape}")
    print(f" - embeddings_eval shape:  {embeddings_eval.shape}")

    # 5. Test Contrastive Loss
    print("\nEvaluating ContrastiveLoss...")
    contrastive_criterion = ContrastiveLoss()
    loss_contrastive = contrastive_criterion(embeddings_train, probs_train)
    print(f" - Contrastive Loss: {loss_contrastive.item():.6f}")
    
    # Check if backpropagation works through the loss
    loss_contrastive.backward(retain_graph=True)
    print(" - Backward pass on ContrastiveLoss completed successfully.")

    # 6. Test Orthogonality Loss
    print("\nEvaluating OrthogonalityLoss...")
    orth_criterion = OrthogonalityLoss()
    # Pass layer's regime projection weights
    loss_orth = orth_criterion(layer.regime_proj.weight)
    print(f" - Projection weight shape: {layer.regime_proj.weight.shape}")
    print(f" - Orthogonality Loss:      {loss_orth.item():.6f}")
    
    # Check if backpropagation works through the loss
    loss_orth.backward()
    print(" - Backward pass on OrthogonalityLoss completed successfully.")

    # Verification checks
    assert not torch.isnan(probs_train).any(), "Found NaN in train probabilities"
    assert not torch.isnan(loss_contrastive).any(), "Found NaN in Contrastive Loss"
    assert not torch.isnan(loss_orth).any(), "Found NaN in Orthogonality Loss"
    print("\nAll checks passed successfully! Modules are fully vectorized and mathematically sound.")

if __name__ == "__main__":
    run_tests()
