import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

def preprocess_spy_data():
    print("Downloading SPY daily close data from yfinance...")
    # Fetch SPY daily OHLCV data from 2018-01-01 to present
    df = yf.download("SPY", start="2018-01-01")
    
    # Flatten MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    print(f"Downloaded {len(df)} rows of raw data.")
    
    # Extract series and align
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    open_p = df["Open"]
    volume = df["Volume"]
    
    # 1. Price spreads (Eq 13: HL_t = ln(H_t/L_t), OC_t = ln(C_t/O_t))
    hl_spread = np.log(high / low)
    oc_spread = np.log(close / open_p)
    
    # 2. Log-returns
    r1 = np.log(close / close.shift(1))
    r7 = np.log(close / close.shift(7))
    r1_high = np.log(high / high.shift(1))
    r1_low = np.log(low / low.shift(1))
    r1_open = np.log(open_p / open_p.shift(1))
    r1_volume = np.log((volume + 1e-8) / (volume.shift(1) + 1e-8))
    
    # 3. 21-day rolling volatility computed on past returns only (closed-left window)
    w = 21
    epsilon = 1e-8
    
    r1_shifted = r1.shift(1)
    r_bar = r1_shifted.rolling(window=w).mean()
    sigma = r1_shifted.rolling(window=w).std(ddof=0)
    
    # Volatility Ratio VRt
    vr = sigma / (r_bar.abs() + epsilon)
    
    # 4. Average True Range (ATR) on past returns only (respecting closed-left)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.shift(1).rolling(window=w).mean()
    
    # 5. Price momentum (Velocity and Acceleration)
    velocity = close - close.shift(1)
    acceleration = velocity - velocity.shift(1)
    
    # 6. Volume dynamics
    vol_shifted = volume.shift(1)
    vol_bar = vol_shifted.rolling(window=w).mean()
    sigma_vol = vol_shifted.rolling(window=w).std(ddof=0)
    
    delta_vol = (volume - vol_bar) / (sigma_vol + epsilon)
    volume_state_ratio = volume / (vol_bar + epsilon)
    
    # 7. Regime-context indicators (63-day window)
    regime_window = 63
    volatility_regime = sigma / (sigma.rolling(window=regime_window).mean() + epsilon)
    momentum_regime = r1.rolling(window=regime_window).mean() / (r1.rolling(window=regime_window).std(ddof=0) + epsilon)
    acceleration_regime = acceleration.rolling(window=regime_window).mean()
    
    # 9. Target variable yt = (Ct+1 - Ct) / Ct
    target = close.pct_change(1).shift(-1)
    
    # Assemble feature DataFrame
    features = pd.DataFrame({
        "HL_Spread": hl_spread,
        "OC_Spread": oc_spread,
        "Log_Return_1d": r1,
        "Log_Return_7d": r7,
        "Log_Return_High_1d": r1_high,
        "Log_Return_Low_1d": r1_low,
        "Log_Return_Open_1d": r1_open,
        "Log_Return_Volume_1d": r1_volume,
        "Rolling_Mean_Return_21d": r_bar,
        "Rolling_Volatility_21d": sigma,
        "Volatility_Ratio_21d": vr,
        "ATR_21d": atr,
        "Velocity": velocity,
        "Acceleration": acceleration,
        "Delta_Volume": delta_vol,
        "Volume_State_Ratio": volume_state_ratio,
        "Volatility_Regime_63d": volatility_regime,
        "Momentum_Regime_63d": momentum_regime,
        "Acceleration_Regime_63d": acceleration_regime,
        "Target_Return_Next_Day": target
    })
    
    # ======================================================================
    # BUG 8 FIX: Add actual temporal dummies (Paper Algorithm 1, Step 2;
    # Fig 2 "One-Hot Encoding").  The original code concatenated features
    # with itself (a no-op).  We now create day-of-week (4 dummies, Mon=0
    # dropped to avoid collinearity) and month (11 dummies, Jan dropped).
    # ======================================================================
    dow = features.index.dayofweek                         # 0=Mon .. 4=Fri
    month = features.index.month                           # 1=Jan  .. 12=Dec

    for d in range(1, 5):                                  # Tue–Fri dummies
        features[f"DOW_{d}"] = (dow == d).astype(float)
    for m in range(2, 13):                                 # Feb–Dec dummies
        features[f"Month_{m}"] = (month == m).astype(float)

    # Drop rows containing NaNs due to rolling window warmups and next-day target shift
    features_clean = features.dropna()
    print(f"Prepared feature dataframe. Cleaned shape: {features_clean.shape}")
    
    # Chronological Split of Raw Data (70% Train, 15% Val, 15% Test)
    n_samples = len(features_clean)
    train_end = int(0.70 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_df = features_clean.iloc[:train_end]
    val_df = features_clean.iloc[train_end:val_end]
    test_df = features_clean.iloc[val_end:]
    
    # 10. Feature Selection with SelectKBest (Algorithm 1, Step 5)
    # Fitted ONLY on training set to prevent data leakage
    candidate_cols = [
        "HL_Spread", "OC_Spread", "Log_Return_1d", "Log_Return_7d", 
        "Log_Return_High_1d", "Log_Return_Low_1d", "Log_Return_Open_1d", 
        "Log_Return_Volume_1d", "Rolling_Volatility_21d", "Volatility_Ratio_21d", 
        "ATR_21d", "Velocity", "Acceleration", "Delta_Volume", "Volume_State_Ratio",
        "Volatility_Regime_63d", "Momentum_Regime_63d", "Acceleration_Regime_63d",
        # Temporal dummies (Bug 8 fix)
        "DOW_1", "DOW_2", "DOW_3", "DOW_4",
        "Month_2", "Month_3", "Month_4", "Month_5", "Month_6",
        "Month_7", "Month_8", "Month_9", "Month_10", "Month_11", "Month_12",
    ] 
    
    # 10. No Feature Selection (Unstarve the Network - use all 33 candidate features)
    selected_features = candidate_cols
    
    # 11. Standardize Features (Algorithm 1, Step 6)
    # Fit ONLY on training set to prevent data leakage
    print("Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(train_df[selected_features])
    X_val = scaler.transform(val_df[selected_features])
    X_test = scaler.transform(test_df[selected_features])
    
    y_train = train_df["Target_Return_Next_Day"].values
    y_val = val_df["Target_Return_Next_Day"].values
    y_test = test_df["Target_Return_Next_Day"].values
    
    # Dates index references
    dates_train = train_df.index
    dates_val = val_df.index
    dates_test = test_df.index
    
    print("\n2D Cross-Sectional Splitting (Chronological):")
    print(f" - X_train shape: {X_train.shape}, y_train shape: {y_train.shape} ({dates_train[0].date()} to {dates_train[-1].date()})")
    print(f" - X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape} ({dates_val[0].date()} to {dates_val[-1].date()})")
    print(f" - X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape} ({dates_test[0].date()} to {dates_test[-1].date()})")
    
    # Save 2D arrays to .npy in a structured data/ directory
    os.makedirs("data", exist_ok=True)
    np.save("data/spy_train_X.npy", X_train)
    np.save("data/spy_train_y.npy", y_train)
    np.save("data/spy_val_X.npy", X_val)
    np.save("data/spy_val_y.npy", y_val)
    np.save("data/spy_test_X.npy", X_test)
    np.save("data/spy_test_y.npy", y_test)
    
    # Save CSV files for reference
    train_df.to_csv("data/spy_train.csv")
    val_df.to_csv("data/spy_val.csv")
    test_df.to_csv("data/spy_test.csv")
    features_clean.to_csv("data/spy_full_features.csv")
    print("\nSaved all split datasets (both .npy and .csv) to the 'data/' folder.")
    
    return selected_features

if __name__ == "__main__":
    selected_features = preprocess_spy_data()
    print(f"\nFinal Selected Features for KAN Input: {selected_features}")