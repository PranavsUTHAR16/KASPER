import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from kasper import KASPER

# Core financial features from preprocessed SPY dataset (14 features)
FEATURE_NAMES = [
    "HL_Spread", "OC_Spread", "Log_Return_1d", "Log_Return_7d", 
    "Log_Return_High_1d", "Log_Return_Low_1d", "Log_Return_Open_1d", 
    "Log_Return_Volume_1d", "Rolling_Volatility_21d", "Volatility_Ratio_21d", 
    "ATR_21d", "Velocity", "Acceleration", "Delta_Volume"
]

def calculate_financial_metrics(strategy_returns, actual_returns, y_hat, active_regimes, is_dummy=False, dataset_type="test"):
    """
    Calculates and prints global and regime-specific forecasting/financial metrics,
    and generates visual plots.
    
    Args:
        strategy_returns (np.ndarray): Realized strategy returns, shape (N,).
        actual_returns (np.ndarray): Actual market returns, shape (N,).
        y_hat (np.ndarray): Model predicted returns, shape (N,).
        active_regimes (np.ndarray): Hard regime assignments, shape (N,).
        is_dummy (bool): If True, skips saving to the final artifact directory path.
        dataset_type (str): "train" or "test".
        
    Returns:
        dict: Calculated metrics.
    """
    # 1. Global Financial Metrics
    direction_acc = np.mean(np.sign(y_hat) == np.sign(actual_returns)) * 100.0
    win_rate = np.mean(strategy_returns > 0) * 100.0
    cum_returns = (np.prod(1.0 + strategy_returns) - 1.0) * 100.0
    actual_cum_returns = (np.prod(1.0 + actual_returns) - 1.0) * 100.0

    std_returns = np.std(strategy_returns)
    sharpe = (np.mean(strategy_returns) / (std_returns + 1e-8)) * np.sqrt(252.0) if std_returns > 0 else 0.0

    cumulative_equity = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_equity)
    running_max = np.where(running_max <= 0, 1e-8, running_max)
    drawdowns = (cumulative_equity - running_max) / running_max
    max_dd = drawdowns.min() * 100.0

    # 2. Global Machine Learning Metrics
    global_r2 = r2_score(actual_returns, y_hat)
    global_mse = mean_squared_error(actual_returns, y_hat)

    print("\n" + "=" * 60)
    print(f"GLOBAL PERFORMANCE SUMMARY ({dataset_type.upper()} SET)")
    print("=" * 60)
    print(f"{'Global R^2 Score':30s} | {global_r2:.6f}")
    print(f"{'Global Mean Squared Error':30s} | {global_mse:.6f}")
    print(f"{'Direction Accuracy (%)':30s} | {direction_acc:.4f}%")
    print(f"{'Win Rate (%)':30s} | {win_rate:.4f}%")
    print(f"{'Strategy Cumulative Return (%)':30s} | {cum_returns:.4f}%")
    print(f"{'Market Cumulative Return (%)':30s} | {actual_cum_returns:.4f}%")
    print(f"{'Annualized Sharpe Ratio':30s} | {sharpe:.4f}")
    print(f"{'Max Drawdown (%)':30s} | {max_dd:.4f}%")
    print("=" * 60)

    # 3. Regime-Specific Metrics Breakdown
    print("\n" + "=" * 60)
    print(f"REGIME-SPECIFIC PERFORMANCE BREAKDOWN ({dataset_type.upper()} SET)")
    print("=" * 60)

    for r in range(3):
        mask = (active_regimes == r)
        count = np.sum(mask)
        print(f"\n>>> Regime {r} (Sample Count: {count})")
        if count == 0:
            print("    No active samples in this regime.")
            continue

        r_actual = actual_returns[mask]
        r_pred = y_hat[mask]
        r_strat = strategy_returns[mask]

        # Machine Learning Metrics (requires at least 2 samples for R^2)
        r_r2 = r2_score(r_actual, r_pred) if count > 1 else float('nan')
        r_mse = mean_squared_error(r_actual, r_pred)

        # Financial Metrics
        r_dir_acc = np.mean(np.sign(r_pred) == np.sign(r_actual)) * 100.0
        r_std = np.std(r_strat)
        r_sharpe = (np.mean(r_strat) / (r_std + 1e-8)) * np.sqrt(252.0) if r_std > 0 else 0.0
        r_win_rate = np.mean(r_strat > 0) * 100.0

        print(f"    R^2 Score                  : {r_r2:.6f}")
        print(f"    Mean Squared Error         : {r_mse:.6f}")
        print(f"    Directional Accuracy (%)   : {r_dir_acc:.4f}%")
        print(f"    Win Rate (%)               : {r_win_rate:.4f}%")
        print(f"    Annualized Sharpe Ratio    : {r_sharpe:.4f}")

    print("=" * 60)

    # 4. Generate Visualizations
    scatter_filename = f"regime_scatter_{dataset_type}.png" if dataset_type != "test" else "regime_scatter.png"
    equity_filename = f"regime_equity_{dataset_type}.png" if dataset_type != "test" else "regime_equity.png"

    # Visualization 1: Regime Distribution and Scatter Plot
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot A: Regime Distribution (Pie Chart)
    counts = [np.sum(active_regimes == r) for r in range(3)]
    labels = [f'Regime {r}' for r in range(3) if counts[r] > 0]
    sizes = [c for c in counts if c > 0]
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
    active_colors = [colors_pie[r] for r in range(3) if counts[r] > 0]
    
    if len(sizes) > 0:
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=active_colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax1.set_title(f"Regime Distribution in {dataset_type.capitalize()} Set")

    # Subplot B: Predicted vs Actual Scatter Plot
    colors_scatter = ['red', 'gray', 'green']
    for r in range(3):
        r_mask = (active_regimes == r)
        if np.sum(r_mask) > 0:
            ax2.scatter(actual_returns[r_mask], y_hat[r_mask], 
                        color=colors_scatter[r], label=f'Regime {r}', alpha=0.6, edgecolors='none')
            
    # Add identity line (y = x)
    min_val = min(actual_returns.min(), y_hat.min())
    max_val = max(actual_returns.max(), y_hat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75, zorder=0, label='y=x')
    
    ax2.set_xlabel("Actual Returns")
    ax2.set_ylabel("Predicted Returns")
    ax2.set_title(f"Predicted vs Actual ({dataset_type.capitalize()} Set | Global $R^2$: {global_r2:.6f})")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(scatter_filename, bbox_inches='tight', dpi=150)
    plt.close(fig1)

    # Visualization 2: Cumulative Returns with Regime Highlighting
    fig2, ax = plt.subplots(figsize=(12, 6))

    strat_val = np.cumprod(1.0 + strategy_returns)
    mkt_val = np.cumprod(1.0 + actual_returns)
    days = np.arange(len(strategy_returns))

    ax.plot(days, strat_val, color='blue', linewidth=2, label='Strategy (KASPER)')
    ax.plot(days, mkt_val, color='black', linewidth=1.5, linestyle='--', label='Buy & Hold (Market)')

    # Shading backgrounds based on active regimes
    # Colors: Red for Regime 0, Light Gray for Regime 1, Light Green for Regime 2
    colors_shade = ['#ffcccc', '#f2f2f2', '#ccffcc']
    for t in range(len(active_regimes)):
        ax.axvspan(t, t + 1, color=colors_shade[active_regimes[t]], alpha=0.3, linewidth=0)

    # Create custom legend entries for the shading colors
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Strategy (KASPER)'),
        plt.Line2D([0], [0], color='black', lw=1.5, ls='--', label='Buy & Hold'),
        Patch(facecolor='#ffcccc', edgecolor='none', alpha=0.3, label='Regime 0 (Bear)'),
        Patch(facecolor='#f2f2f2', edgecolor='none', alpha=0.3, label='Regime 1 (Neutral)'),
        Patch(facecolor='#ccffcc', edgecolor='none', alpha=0.3, label='Regime 2 (Bull)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Growth of $1 Investment")
    ax.set_title(f"Strategy Growth vs Buy-and-Hold ({dataset_type.capitalize()} Set)")
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(equity_filename, bbox_inches='tight', dpi=150)
    plt.close(fig2)

    # Copy plots to the artifacts folder if NOT in dummy mode
    if not is_dummy:
        artifact_dir = "/Users/prana/.gemini/antigravity-ide/brain/4506f1b5-39d0-49cf-bed3-73c70d3ceba5"
        if os.path.exists(artifact_dir):
            import shutil
            shutil.copy(scatter_filename, os.path.join(artifact_dir, scatter_filename))
            shutil.copy(equity_filename, os.path.join(artifact_dir, equity_filename))
            print(f" - {dataset_type.capitalize()} visualizations successfully copied to artifacts directory.")

    return {
        "global_r2": global_r2,
        "global_mse": global_mse,
        "direction_acc": direction_acc,
        "win_rate": win_rate,
        "cum_returns": cum_returns,
        "sharpe": sharpe,
        "max_dd": max_dd
    }


def evaluate_model(dataset_type="test"):
    print("--------------------------------------------------")
    print(f"KASPER Model Inference and Financial Evaluation ({dataset_type.upper()} Set)")
    print("--------------------------------------------------")

    # Paths
    weights_path = "best_kasper.pth"
    train_y_path = "data/spy_train_y.npy"
    if dataset_type == "train":
        x_path = "data/spy_train_X.npy"
        y_path = "data/spy_train_y.npy"
    else:
        x_path = "data/spy_test_X.npy"
        y_path = "data/spy_test_y.npy"

    if not os.path.exists(weights_path):
        print(f"Error: Model weights file '{weights_path}' not found.")
        print("Please train the model first by running train.py.")
        return

    if not all(os.path.exists(p) for p in [train_y_path, x_path, y_path]):
        print("Error: Required NumPy data files not found in 'data/' directory.")
        return

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Fit Scaler on training target to recover scale parameters
    print("Loading training target data to fit scaler...")
    y_train = np.load(train_y_path)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train.reshape(-1, 1))

    # 3. Load Data
    print(f"Loading {dataset_type} data...")
    X_data = np.load(x_path)
    y_data = np.load(y_path)
    print(f" - X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
    
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=False)

    # 4. Instantiate and Load KASPER Model
    num_inputs = X_data.shape[1]

    print("\nInstantiating KASPER model...")
    model = KASPER(
        num_inputs=num_inputs,
        hidden_dim=64,
        num_regimes=3,
        grid_size=10,
        n_linear=3,
        n_cubic=2,
        dropout_rate=0.2,
        num_knots=8,
        sparsity_threshold=1e-3
    ).to(device)

    # Fit knots using training data to maintain identical scaling bounds
    train_x_path = "data/spy_train_X.npy"
    if os.path.exists(train_x_path):
        print("Fitting quantile knots using training features...")
        model.fit_knots(torch.tensor(np.load(train_x_path), dtype=torch.float32).to(device))
    else:
        print("Warning: Training features not found. Fitting knots.")
        model.fit_knots(X_tensor.to(device))

    print(f"Loading weights from '{weights_path}'...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # Disable sparsity thresholding during evaluation to analyze unpruned weights by filling theta_raw with a large negative value
    model.layer2.theta_raw.data.fill_(-100.0)
    model.eval()

    # 5. Run Model Inference
    print(f"\nRunning model inference on {dataset_type} set...")
    all_y_hat = []
    all_probs = []
    
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            # Eval pass: returns predictions, probs, embeddings
            y_hat_b, probs_b, _ = model(x_batch, tau=1.0)
            all_y_hat.append(y_hat_b.cpu().numpy())
            all_probs.append(probs_b.cpu().numpy())

    # Concatenate batch predictions and probabilities
    y_hat_scaled = np.concatenate(all_y_hat, axis=0).flatten()
    probs_np = np.concatenate(all_probs, axis=0)

    # Compute hard regime assignments (1D array)
    active_regimes = np.argmax(probs_np, axis=1)

    # 6. Inverse-scale predictions back to the physical return scale
    y_hat_unscaled = y_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).flatten()

    # 7. Trading Strategy Simulation
    # Positions: 1 (Long) if predicted return is positive, else -1 (Short)
    positions = np.where(y_hat_unscaled > 0.0, 1.0, -1.0)
    
    # Realized returns = position * actual_return
    strategy_returns = positions * y_data

    # 8. Calculate Financial/ML Metrics and Plot
    calculate_financial_metrics(strategy_returns, y_data, y_hat_unscaled, active_regimes, is_dummy=False, dataset_type=dataset_type)


if __name__ == "__main__":
    # --- Part 1: Mock Data Verification ---
    print("==================================================")
    print("MOCK DATA VERIFICATION TEST")
    print("==================================================")
    
    # Generate 100 days of mock returns, predictions, and probs (Batch of 100, 3 regimes)
    np.random.seed(42)
    mock_actual = np.random.normal(loc=0.0005, scale=0.01, size=100) # Slightly positive mean return
    mock_pred = mock_actual + np.random.normal(loc=0.0, scale=0.005, size=100) # Positive correlation with actual
    
    # Generate mock probabilities for 3 regimes
    mock_probs = np.random.dirichlet(alpha=[1, 1, 1], size=100)
    mock_active_regimes = np.argmax(mock_probs, axis=1)
    
    # Position decision
    mock_positions = np.where(mock_pred > 0.0, 1.0, -1.0)
    mock_strat_returns = mock_positions * mock_actual
    
    print("Evaluating financial metrics on mock data...")
    calculate_financial_metrics(mock_strat_returns, mock_actual, mock_pred, mock_active_regimes, is_dummy=True, dataset_type="test")
    
    # --- Part 2: Real Model Evaluation ---
    if os.path.exists("best_kasper.pth"):
        print("\n\n" + "=" * 50)
        print("REAL MODEL EVALUATION (SPY TRAINING DATASET)")
        print("=" * 50)
        evaluate_model(dataset_type="train")
        
        print("\n\n" + "=" * 50)
        print("REAL MODEL EVALUATION (SPY TEST DATASET)")
        print("=" * 50)
        evaluate_model(dataset_type="test")
    else:
        print("\nNote: 'best_kasper.pth' not found. Skipping real test set evaluation.")
