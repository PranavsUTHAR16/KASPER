import torch
from kasper import KASPER
from losses import composite_loss

def test_master_pipeline():
    print("--------------------------------------------------")
    print("Testing KASPER Master Model & Composite Loss Pipeline")
    print("--------------------------------------------------")

    # Dimensions
    batch_size = 32
    num_inputs = 14
    num_regimes = 3

    # Generate dummy input and target tensors
    phi_t = torch.randn(batch_size, num_inputs)
    targets = torch.randn(batch_size)

    print(f"Inputs:")
    print(f" - Features phi_t shape: {phi_t.shape}")
    print(f" - Targets shape:        {targets.shape}")

    # 1. Instantiate the model
    print("\nInstantiating KASPER model...")
    model = KASPER(
        num_inputs=num_inputs,
        hidden_dim=64,
        num_regimes=num_regimes,
        grid_size=10,
        n_linear=3,
        n_cubic=2,
        dropout_rate=0.2,
        num_knots=5,
        sparsity_threshold=1e-3
    )

    # 2. Fit knots
    print("Fitting knots on training data...")
    model.fit_knots(phi_t)

    # 4. Run forward pass
    print("\nRunning model forward pass (train mode)...")
    model.train()
    predictions, probs, embeddings = model(phi_t, tau=1.0)
    print(f" - Predictions shape: {predictions.shape} (Expected: [{batch_size}])")
    print(f" - Probs shape:       {probs.shape} (Expected: [{batch_size}, {num_regimes}])")
    print(f" - Embeddings shape:  {embeddings.shape} (Expected: [{batch_size}, 64])")

    # 5. Compute loss
    print("\nComputing composite loss...")
    loss_dict = composite_loss(
        y_hat=predictions,
        y_true=targets,
        z=embeddings,
        p=probs,
        regime_ids=probs.argmax(dim=-1),
        kan1=model.layer1,
        kan2=model.layer2,
        include_balance=True
    )
    total_loss = loss_dict["total"]

    print(f" - Total Loss: {total_loss.item():.6f}")
    print(" - Component losses:")
    for name, val in loss_dict.items():
        if name != "total":
            print(f"   * {name:12s}: {val.item():.6f}")

    # 6. Run backward pass to verify gradient flow
    print("\nRunning backward pass...")
    total_loss.backward()

    # Check if parameters of both Layer 1 and Layer 2 have gradients
    layer1_has_grad = all(p.grad is not None for p in model.layer1.parameters() if p.requires_grad)
    layer2_has_grad = all(p.grad is not None for p in model.layer2.parameters() if p.requires_grad)

    print(f" - Layer 1 has gradients: {layer1_has_grad}")
    print(f" - Layer 2 has gradients: {layer2_has_grad}")

    # Checks
    assert predictions.shape == (batch_size,), f"Unexpected predictions shape: {predictions.shape}"
    assert total_loss.shape == (), f"Total loss must be a scalar, got shape: {total_loss.shape}"
    assert layer1_has_grad, "Some Layer 1 weights did not receive gradients."
    assert layer2_has_grad, "Some Layer 2 weights did not receive gradients."

    print("\nSuccess! KASPER master model and composite loss verification completed successfully.")

if __name__ == "__main__":
    test_master_pipeline()
