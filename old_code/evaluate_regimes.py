import torch
import numpy as np
import pandas as pd
from regime_detection_layer import RegimeDetectionLayer

# 1. Load Test Data
print("Loading test data...")
X_test = np.load("data/spy_test_X.npy")
y_test = np.load("data/spy_test_y.npy")

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 2. Load Trained Model (Only need Layer 1 to evaluate regimes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegimeDetectionLayer(num_inputs=80, hidden_dim=64, num_regimes=3).to(device)

# Load the FULL checkpoint, but extract only Layer 1
checkpoint = torch.load("kasper_full_model_best.pth", map_location=device)
model.load_state_dict(checkpoint['layer1_state_dict'])
model.eval()

# 3. Run Inference
print("Running regime detection on test set...")
with torch.no_grad():
    logits, probs, z = model(X_test_tensor.to(device), tau=0.5, hard=False)
    regime_assignments = torch.argmax(probs, dim=1).cpu().numpy()

# 4. Analyze Regimes
print("\n" + "="*50)
print("REGIME ANALYSIS (TEST SET)")
print("="*50)

df_analysis = pd.DataFrame({
    'Regime': regime_assignments,
    'Next_Day_Return': y_test
})

summary = df_analysis.groupby('Regime')['Next_Day_Return'].agg(
    Days='count',
    Avg_Return='mean',
    Volatility='std',
    Win_Rate=lambda x: (x > 0).mean() * 100
)
summary['Time_Spent_%'] = (summary['Days'] / summary['Days'].sum()) * 100

print("\nRegime Statistics:")
print(summary.to_string(float_format="{:.4f}".format))

# 5. Interpretation
print("\n" + "="*50)
print("HOW TO INTERPRET THIS:")
print("="*50)
for regime in sorted(df_analysis['Regime'].unique()):
    row = summary.loc[regime]
    avg_ret = row['Avg_Return']
    vol = row['Volatility']
    win_rate = row['Win_Rate']
    
    if avg_ret > 0.0005 and win_rate > 55:
        regime_type = "BULLISH (Upward Trend)"
    elif avg_ret < -0.0005 and win_rate < 45:
        regime_type = "BEARISH (Downward Trend)"
    else:
        regime_type = "NEUTRAL (Sideways/Choppy)"
        
    print(f"\nRegime {regime} -> {regime_type}")
    print(f"  - Spent {row['Time_Spent_%']:.2f}% of time in this regime.")
    print(f"  - Average Next Day Return: {avg_ret:.5f}")
    print(f"  - Volatility (Std Dev):    {vol:.5f}")
    print(f"  - Win Rate (Up days):      {win_rate:.2f}%")