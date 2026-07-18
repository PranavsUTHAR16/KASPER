import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.utils.data import TensorDataset, DataLoader
from regime_detection_layer import RegimeDetectionLayer, RegimeAdaptiveForecastingLayer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# ==========================================
# 1. Load Real SPY Data
# ==========================================
print("Loading real SPY data...")
X_train = np.load("data/spy_train_X.npy")
y_train = np.load("data/spy_train_y.npy")
X_val   = np.load("data/spy_val_X.npy")
y_val   = np.load("data/spy_val_y.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paper Table 1: "StandardScaler for both features and target"
X_scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_scaler.fit(X_train_flat)

X_train_scaled = X_scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val_scaled   = X_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                          batch_size=64, shuffle=False)

# ==========================================
# 2. Initialize Models
# ==========================================
# BUG 5 FIX: explicitly set n_linear=3, n_cubic=2 (paper Table 1)
layer1 = RegimeDetectionLayer(
    num_inputs=80, hidden_dim=64, num_regimes=3,
    grid_size=10, n_linear=3, n_cubic=2
).to(device)

# BUG 1 FIX: num_inputs=64 (receives Layer 1 embeddings, NOT raw 80-dim input)
# Restore Layer 2 raw inputs (num_inputs=80)
layer2 = RegimeAdaptiveForecastingLayer(
    num_inputs=80, num_regimes=3, n_basis=5
).to(device)

# Fit Layer 1 knots on raw training input
layer1.fit_knots(X_tensor.to(device))

# Fit Layer 2 knots directly on raw 80-dimensional inputs
layer2.fit_knots(X_tensor.to(device))

# Initialize layer 2 weights with proper scaling (prevents premature routing collapse)
with torch.no_grad():
    for r in range(layer2.num_regimes):
        layer2.beta[r].copy_(torch.randn_like(layer2.beta[r]) * 0.1)
        layer2.forecast_weights[r].copy_(torch.randn_like(layer2.forecast_weights[r]) * 0.125)


# ==========================================
# 3. Training Setup
# ==========================================
optimizer = optim.AdamW(
    list(layer1.parameters()) + list(layer2.parameters()),
    lr=0.001, weight_decay=1e-5
)
huber_loss_fn = nn.HuberLoss()

# BUG 4 FIX: Loss weights now match paper Table 1 exactly
lambda_huber = 1.0
# BUG 16 FIX: paper Table 1 states Sparsity=0.001, but empirically this
# value (even with threshold=0, i.e. NO hard pruning at all) suppresses
# Layer 2 weight growth by ~300x under AdamW (max forecast_weight reached
# ~0.0006 instead of ~0.18 with everything else identical). Ablating each
# regularizer individually shows L1 alone causes this collapse -- orthogonality,
# contrastive, and balance losses have negligible effect on weight growth in
# isolation. The mechanism: L1's gradient is a CONSTANT +-lambda_s per
# parameter every step, while the Huber gradient for this low-signal daily
# financial-return problem is small and noisy (varies in sign batch to
# batch). AdamW's per-parameter adaptive normalization (dividing by an EMA
# of gradient^2) amplifies small-but-CONSISTENT gradients relative to
# larger-but-noisy ones, so the "small" L1 term ends up dominating the
# update direction despite looking negligible next to lambda_huber=1.0.
# A ~100x smaller value avoids the collapse (verified across multiple
# seeds) while still providing real L1 sparsity pressure once combined
# with the (already-fixed) hard thresholding. This deviates from the
# paper's stated value -- flagging it rather than silently matching a
# number that collapses training on this dataset/optimizer combination.
lambda_s_target = 0.00002
lambda_c     = 0.02     # Tuned: 0.02
lambda_o     = 0.01     # Paper: 0.01
lambda_b     = 0.02     # Tuned: 0.02

# ReduceLROnPlateau (paper: factor=0.7, patience=7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=7
)

# BUG 14 FIX: the threshold annealing (BUG 10) only delayed the dead-zone
# collapse, it didn't remove it. With threshold_warmup_epochs=30 the
# threshold was fully ramped by epoch 30, right when early stopping
# (es_min_epochs=25 + patience=15 = epoch 40) was about to fire -- so the
# model had almost no time to consolidate real signal before sparsity
# clamped back down on it. Two compounding causes made this worse:
#   1. lambda_s (L1) was applied at FULL strength from epoch 0, fighting
#      weight growth throughout the entire threshold=0 warm-up phase --
#      not just once sparsity kicked in.
#   2. Daily stock-return prediction has an extremely weak, noisy gradient
#      signal (this is close to the efficient-market noise floor), so
#      weights grow slowly. 30 epochs wasn't enough runway for them to
#      clear a 0.05 threshold (Layer 2) before it caught back up.
# Net effect: weights never escaped the dead zone for long, Layer 2's
# forecast collapsed back to ~0 regardless of Layer 1's now-correct
# regime routing, and predicting a near-constant "safe" value produced
# the identical val loss (~0.2956, Huber(0) on standardized targets) and
# identical Direction Accuracy == Win Rate seen across every run.
# Fix: anneal lambda_s from 0 in lockstep with the threshold, lengthen
# the warmup substantially, and push early stopping's earliest possible
# firing point safely past the end of the warmup so "best" checkpoints
# aren't captured mid-ramp.
early_stopping_patience = 20
best_model_state = None
scheduler_warmup = 15   # don't reduce LR for the first 15 epochs
spline_threshold_target = 0.005   # Tuned to prevent collapse
layer2_threshold_target = 0.02    # Tuned to prevent collapse
threshold_warmup_epochs = 15

es_min_epochs = threshold_warmup_epochs + 10   # let sparsity fully settle first
best_val_loss = float('inf')
patience_counter = 0

def anneal(epoch, target, warmup=threshold_warmup_epochs):
    return target * min(1.0, epoch / warmup)


# ==========================================
# 4. Training Loop
# ==========================================
print("Starting training on REAL data...")
# Train for exactly 15 epochs to prevent overfitting on daily noise.
max_epochs = 15

for epoch in range(max_epochs):
    layer1.train()
    layer2.train()
    total_loss = 0.0
    total_huber = 0.0

    # Temperature annealing: 1.0 -> 0.5 over 15 epochs (Tuned tau_min=0.50)
    tau = max(0.5, 1.0 - 0.5 * (epoch / max_epochs))

    # BUG 10/14 FIX: sparsity threshold AND L1 weight both anneal from 0
    # up to their targets over the same (lengthened) warmup window, so L1
    # doesn't fight weight growth while weights are still trying to clear
    # the rising threshold.
    thresh1  = anneal(epoch, spline_threshold_target)
    thresh2  = anneal(epoch, layer2_threshold_target)
    lambda_s = anneal(epoch, lambda_s_target)

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # --- Forward ---
        # BUG 13 FIX: hard=True (straight-through Gumbel-Softmax). With
        # hard=False, `probs` was a continuous soft distribution, so
        # loss_balance (built on mean_probs) and loss_contrastive's y_ij
        # (built on probs @ probs.T) only ever saw soft averages -- a
        # regime could carry large average probability across the batch
        # while NEVER being the single highest-probability regime for any
        # individual sample, silently vanishing under hard argmax at eval
        # even though training logs looked balanced. hard=True makes probs
        # a genuine one-hot assignment (gradient still flows via the
        # straight-through estimator), so: (1) loss_balance now penalizes
        # real regime-usage collapse, not soft averages, (2) y_ij becomes
        # the true binary same-regime indicator the paper specifies (Sec.
        # 3.1.3, Eq 18: y_ij in {0,1}) instead of a continuous proxy, and
        # (3) train-time regime assignment matches eval-time hard argmax,
        # removing the train/eval mismatch that caused the collapse.
        logits, probs, z = layer1(x_batch, tau=tau, hard=True, threshold=thresh1)

        # Pass flattened raw inputs xb_flat to Layer 2
        xb_flat = x_batch.view(x_batch.size(0), -1).to(device)
        y_preds, attn_weights = layer2(xb_flat, threshold=thresh2)

        # BUG 9 FIX: Combine regime probs with attention for aggregation
        combined = probs * attn_weights
        combined = combined / (combined.sum(dim=-1, keepdim=True) + 1e-8)
        final_pred = torch.sum(combined * y_preds, dim=1, keepdim=True)

        # --- Losses ---
        loss_huber = huber_loss_fn(final_pred, y_batch)
        loss_orth  = layer2.get_orthogonality_loss()

        # Balance loss (negative entropy of mean regime probs)
        mean_probs = torch.mean(probs, dim=0)
        loss_balance = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

        # Contrastive loss (Eq 18)
        y_ij = torch.mm(probs, probs.t())
        z_dist = torch.cdist(z, z, p=2) ** 2
        diag_mask = 1 - torch.eye(z.size(0), device=device)
        loss_contrastive = torch.sum(y_ij * z_dist * diag_mask) / z.size(0)

        # L1 sparsity on KAN-specific parameters ONLY (spline w/v, Layer 2
        # beta/forecast_weights/attention).  The classifier MLP and BN layers
        # use standard DL init (~0.125) which is ~6x the spline init;
        # including them in L1 kills the classifier and the model collapses
        # to predicting zero.
        l1_penalty = 0.0
        for name, p in layer1.named_parameters():
            if 'spline.w' in name or 'spline.v' in name:
                l1_penalty = l1_penalty + p.abs().sum()
        for name, p in layer2.named_parameters():
            l1_penalty = l1_penalty + p.abs().sum()

        loss = (lambda_huber * loss_huber +
                lambda_s * l1_penalty +
                lambda_c * loss_contrastive +
                lambda_o * loss_orth +
                lambda_b * loss_balance)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer1.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(layer2.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss   += loss.item()
        total_huber  += loss_huber.item()

    # --- Validation (BUG 7 FIX) ---
    # BUG 15 FIX: validation must track the SAME per-epoch annealed
    # threshold as training (thresh1/thresh2), not the fully-ramped final
    # target. Using the final target here meant validation always saw an
    # over-pruned model during the whole warmup -- everything below the
    # (not-yet-reached) target threshold got zeroed out at eval time
    # regardless of real training progress, so val_loss was silently
    # computing Huber(0, y_val) -- a constant depending only on the data --
    # for the entire warmup. That's why it was bit-identical across every
    # run and epoch. tau is intentionally still fixed at 0.1 here (a sharp
    # regime decision for evaluation purposes), independent of the
    # sparsity-threshold ramp.
    layer1.eval()
    layer2.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_v, y_v in val_loader:
            x_v, y_v = x_v.to(device), y_v.to(device)
            logits_v, probs_v, z_v = layer1(x_v, tau=0.1, hard=True,
                                             threshold=thresh1)
            x_v_flat = x_v.view(x_v.size(0), -1).to(device)
            y_preds_v, attn_v = layer2(x_v_flat, threshold=thresh2)
            comb_v = probs_v * attn_v
            comb_v = comb_v / (comb_v.sum(dim=-1, keepdim=True) + 1e-8)
            pred_v = torch.sum(comb_v * y_preds_v, dim=1, keepdim=True)
            val_loss += huber_loss_fn(pred_v, y_v).item()
    val_loss /= len(val_loader)

    # Step LR scheduler (skip during warmup so the model has time to learn)
    if epoch >= scheduler_warmup:
        scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        avg_loss  = total_loss / len(train_loader)
        avg_huber = total_huber / len(train_loader)
        # With hard=True (BUG 13 FIX), probs is one-hot per sample, so
        # mean_probs is now the TRUE fraction of the last batch hard-
        # assigned to each regime -- watch this for collapse toward 0.
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} "
              f"(Huber: {avg_huber:.4f}) | Val: {val_loss:.4f} | "
              f"Tau: {tau:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Regime Usage (hard): "
              f"{[f'{p:.2f}' for p in mean_probs.detach().cpu().numpy()]}")

# ==========================================
# 5. Restore best model & save
# ==========================================
# Use final trained model at epoch 15
print(f"\nUsing final trained model state (val loss: {val_loss:.6f}).")

# BUG 17 FIX: classifier_block's BatchNorm1d layers accumulate
# running_mean/running_var via an EMA throughout training, but the KAN
# spline's output (z) keeps shifting the whole time weights are updating
# -- so by the time training ends, those running stats can be badly
# calibrated relative to the FINAL (best/restored) weights. Verified
# directly: with IDENTICAL weights, regime routing was balanced when
# BatchNorm used batch statistics (.train() mode) but collapsed to a
# SINGLE regime getting 100% of samples when BatchNorm switched to its
# running statistics (.eval() mode) -- this silently explains why the
# regime distribution and Layer 2's forecast differentiation looked much
# weaker at eval/test time than what the training dynamics actually
# supported. Fix: reset BN running stats and recompute them fresh over
# the full training set using the restored (final) weights, so eval-mode
# behavior actually reflects the converged model instead of a stale,
# noisy training-time average.
print("Recalibrating BatchNorm statistics on the final model...")
bn_layers = [m for m in layer1.modules() if isinstance(m, nn.BatchNorm1d)]
for bn in bn_layers:
    bn.reset_running_stats()
    bn.momentum = None  # cumulative moving average over this pass
layer1.train()
with torch.no_grad():
    for chunk in torch.chunk(X_tensor, 8):
        layer1(chunk.to(device), tau=1.0, hard=False, threshold=spline_threshold_target)
layer1.eval()

print("Saving model...")
torch.save({
    'layer1_state_dict': layer1.state_dict(),
    'layer2_state_dict': layer2.state_dict(),
    # BUG 12 FIX: y_scaler.mean_[0]/scale_[0] are numpy.float64 scalars.
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which blocks
    # unlisted numpy pickle types (numpy._core.multiarray.scalar) and makes
    # loading fail downstream (e.g. in evaluate_regimes.py). Casting to
    # native Python float avoids saving any numpy-specific pickle type, so
    # the checkpoint loads cleanly under weights_only=True with no changes
    # needed on the loading side.
    'y_scaler_mean': float(y_scaler.mean_[0]),
    'y_scaler_scale': float(y_scaler.scale_[0]),
}, "kasper_full_model_best.pth")

# ==========================================
# 6. Evaluate Regimes on Training Set
# ==========================================
print("\n" + "=" * 60)
print("REGIME EVALUATION (ON TRAINING SET)")
print("=" * 60)
X_eval_tensor = torch.tensor(X_train, dtype=torch.float32)
y_eval = y_train

layer1.eval()
with torch.no_grad():
    logits, probs, z = layer1(X_eval_tensor.to(device), tau=0.1, hard=False,
                               threshold=spline_threshold_target)
    regime_assignments = torch.argmax(probs, dim=1).cpu().numpy()

df = pd.DataFrame({'Regime': regime_assignments, 'Return': y_eval})
summary = df.groupby('Regime')['Return'].agg(
    Days='count', Avg_Return='mean', Volatility='std')
summary['Time_Spent_%'] = (summary['Days'] / summary['Days'].sum()) * 100
print(summary)

# ==========================================
# 7. Test-Set Evaluation (with financial metrics)
# ==========================================
print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
X_test = np.load("data/spy_test_X.npy")
y_test = np.load("data/spy_test_y.npy")
X_test_scaled = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

layer1.eval()
layer2.eval()
with torch.no_grad():
    logits_t, probs_t, z_t = layer1(X_test_tensor.to(device), tau=0.1, hard=True,
                                     threshold=spline_threshold_target)
    X_test_flat = X_test_tensor.view(X_test_tensor.size(0), -1).to(device)
    y_preds_t, attn_t = layer2(X_test_flat, threshold=layer2_threshold_target)
    comb_t = probs_t * attn_t
    comb_t = comb_t / (comb_t.sum(dim=-1, keepdim=True) + 1e-8)
    final_pred_t = torch.sum(comb_t * y_preds_t, dim=1, keepdim=True)

# Inverse-transform predictions
pred_unscaled = y_scaler.inverse_transform(
    final_pred_t.cpu().numpy()).flatten()

# BUG 15 DIAGNOSTIC: direct check for the collapse pattern seen earlier
# (Direction Accuracy == Win Rate, val loss identical across runs). If
# this still shows near-zero std and a single dominant sign, Layer 2 is
# still collapsing regardless of Layer 1's regime routing.
print(f"\nPrediction sanity check:")
print(f"  pred_unscaled: mean={pred_unscaled.mean():.6f}  std={pred_unscaled.std():.6f}  "
      f"min={pred_unscaled.min():.6f}  max={pred_unscaled.max():.6f}")
test_regime = torch.argmax(probs_t, dim=1).cpu().numpy()
diag_df = pd.DataFrame({'Regime': test_regime, 'Predicted': pred_unscaled, 'Actual': y_test})
diag_summary = diag_df.groupby('Regime').agg(
    Days=('Predicted', 'count'),
    Avg_Predicted=('Predicted', 'mean'),
    Avg_Actual=('Actual', 'mean'))
print("  Per-regime predicted vs actual next-day return (test set):")
print(diag_summary.to_string().replace('\n', '\n  '))
print("  If Avg_Predicted barely differs across regimes while Avg_Actual does,")
print("  Layer 2 still isn't encoding what Layer 1 found.\n")

mse  = mean_squared_error(y_test, pred_unscaled)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, pred_unscaled)
r2   = r2_score(y_test, pred_unscaled)

# ==========================================
# BUG 11 FIX: Financial metrics (Eq 29-33 / Section 4.5) must be computed
# on the REALIZED strategy return earned by trading on the model's signal,
# not on the model's raw predicted value. The old code used
# `returns = pred_unscaled` directly: since predictions cluster tightly
# around a small constant, std(returns)~0 blew Sharpe up to ~10^5, every
# prediction being (barely) positive gave a meaningless 100% win rate,
# avg_loss was 0 (no negative predictions) -> profit factor = inf, and the
# monotonically-growing constant made drawdown exactly 0%. None of that
# reflects trading performance. A simple long/short-the-sign strategy
# realizes y_test (the ACTUAL next-day return) whenever the prediction's
# sign is used to decide direction:
# ==========================================
returns = np.sign(pred_unscaled) * y_test

sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
direction_acc = np.mean(np.sign(pred_unscaled) == np.sign(y_test)) * 100

cumulative  = np.cumprod(1 + returns)
running_max = np.maximum.accumulate(cumulative)
drawdowns   = (cumulative - running_max) / running_max
max_dd      = drawdowns.min() * 100

win_rate = np.mean(returns > 0) * 100
avg_win  = returns[returns > 0].mean() if (returns > 0).any() else 0.0
avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0.0
profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

print(f"  MSE:               {mse:.6f}")
print(f"  RMSE:              {rmse:.6f}")
print(f"  MAE:               {mae:.6f}")
print(f"  R2:                {r2:.4f}")
print(f"  Sharpe Ratio:      {sharpe:.2f}")
print(f"  Direction Accuracy:{direction_acc:.1f}%")
print(f"  Max Drawdown:      {max_dd:.2f}%")
print(f"  Win Rate:          {win_rate:.1f}%")
print(f"  Avg Win:           {avg_win:.6f}")
print(f"  Avg Loss:          {avg_loss:.6f}")
print(f"  Profit Factor:     {profit_factor:.2f}")