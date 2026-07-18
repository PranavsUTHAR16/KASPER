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

    # 8. Regression Test: Verify composite loss Table 1 default weights
    import losses
    assert losses.LAMBDA_CONTRASTIVE == 0.01, f"Expected LAMBDA_CONTRASTIVE=0.01 (Table 1), got {losses.LAMBDA_CONTRASTIVE}"
    assert losses.LAMBDA_SPARSITY == 0.001, f"Expected LAMBDA_SPARSITY=0.001 (Table 1), got {losses.LAMBDA_SPARSITY}"
    assert losses.LAMBDA_ORTHOGONAL == 0.01, f"Expected LAMBDA_ORTHOGONAL=0.01 (Table 1), got {losses.LAMBDA_ORTHOGONAL}"
    assert losses.LAMBDA_BALANCE == 0.05, f"Expected LAMBDA_BALANCE=0.05 (Table 1), got {losses.LAMBDA_BALANCE}"
    print(" - Table 1 loss weight defaults verified.")

    # 9. Regression Test: Verify l1_sparsity touches parameters in both KAN-1 and KAN-2
    model.zero_grad()
    l1_loss = losses.l1_sparsity(model.layer1, model.layer2)
    l1_loss.backward()
    kan1_has_l1_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layer1.parameters())
    kan2_has_l1_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layer2.parameters())
    assert kan1_has_l1_grad, "l1_sparsity did not compute gradients for Layer 1!"
    assert kan2_has_l1_grad, "l1_sparsity did not compute gradients for Layer 2!"
    print(" - L1 sparsity parameter scope verified (touches both kan1 and kan2).")

    # 10. Regression Test: Verify KAN-2 attention and refinement module
    assert hasattr(model.layer2, 'attn_proj'), "KAN-2 is missing attn_proj module!"
    assert hasattr(model.layer2, 'refinement_head'), "KAN-2 is missing refinement_head module!"
    attn_grad = model.layer2.attn_proj.weight.grad is not None
    refine_grad = model.layer2.refinement_head[0].weight.grad is not None
    assert attn_grad and refine_grad, "KAN-2 attention/refinement parameters did not receive gradients!"
    # 11. Regression Test: Verify evaluation consistency (effective_weights / sparsity threshold is unchanged during eval)
    w_effective_before = model.layer2.effective_weights().clone().detach()
    # Perform eval pass
    model.eval()
    with torch.no_grad():
        _ = model(phi_t)
    w_effective_after = model.layer2.effective_weights().clone().detach()
    assert torch.equal(w_effective_before, w_effective_after), "Evaluation path mutated effective_weights or sparsity threshold!"
    print(" - Evaluation consistency verified (sparsity threshold/effective_weights preserved).")

    print("\nSuccess! All KASPER master model and composite loss verification tests passed successfully.")

if __name__ == "__main__":
    test_master_pipeline()
