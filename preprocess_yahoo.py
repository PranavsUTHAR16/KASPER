import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

def preprocess_yahoo_data():
    """
    Data Preprocessing pipeline per Section 4.1 & Table 1 of TMLR 02/2026 Paper:
    - Sourced from Yahoo Finance (SPY daily 2018-2023)
    - Closed-left rolling windows (no lookahead leakage)
    - 15 candidate engineered features
    - SelectKBest with f_regression selecting top 8 features (fitted on Train split only)
    - 70% / 15% / 15% temporal train/val/test split
    """
    print("Downloading SPY daily close data from yfinance (2018-01-01 to 2023-12-31)...")
    df = yf.download("SPY", start="2018-01-01", end="2023-12-31")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    print(f"Downloaded {len(df)} rows of raw data.")
    
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
    
    # Target variable yt = (Ct+1 - Ct) / Ct
    target = close.pct_change(1).shift(-1)
    
    # Assemble feature DataFrame (15 engineered features per Table 1)
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
        "Target_Return_Next_Day": target
    })
    
    features_clean = features.dropna()
    print(f"Prepared feature dataframe. Cleaned shape: {features_clean.shape}")
    
    # Chronological Split of Raw Data (70% Train, 15% Val, 15% Test) per Table 1
    n_samples = len(features_clean)
    train_end = int(0.70 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_df = features_clean.iloc[:train_end]
    val_df = features_clean.iloc[train_end:val_end]
    test_df = features_clean.iloc[val_end:]
    
    # 15 candidate engineered features
    candidate_cols = [
        "HL_Spread", "OC_Spread", "Log_Return_1d", "Log_Return_7d", 
        "Log_Return_High_1d", "Log_Return_Low_1d", "Log_Return_Open_1d", 
        "Log_Return_Volume_1d", "Rolling_Mean_Return_21d", "Rolling_Volatility_21d", 
        "Volatility_Ratio_21d", "ATR_21d", "Velocity", "Acceleration", "Delta_Volume"
    ]
    
    # Feature Selection: Top 8 features via SelectKBest with f_regression (Sec. 4.1, Table 1)
    # Fitted strictly on training split to prevent data leakage
    selector = SelectKBest(score_func=f_regression, k=8)
    selector.fit(train_df[candidate_cols], train_df["Target_Return_Next_Day"])
    selected_indices = selector.get_support(indices=True)
    selected_features = [candidate_cols[i] for i in selected_indices]
    print(f"\nSelectKBest (f_regression, k=8) selected features: {selected_features}")
    
    # Standardize Features (Algorithm 1, Step 6)
    # Fit ONLY on training set to prevent data leakage
    print("Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(train_df[selected_features])
    X_val = scaler.transform(val_df[selected_features])
    X_test = scaler.transform(test_df[selected_features])
    
    y_train = train_df["Target_Return_Next_Day"].values
    y_val = val_df["Target_Return_Next_Day"].values
    y_test = test_df["Target_Return_Next_Day"].values
    
    dates_train = train_df.index
    dates_val = val_df.index
    dates_test = test_df.index
    
    print("\n2D Cross-Sectional Splitting (Chronological 70/15/15):")
    print(f" - X_train shape: {X_train.shape}, y_train shape: {y_train.shape} ({dates_train[0].date()} to {dates_train[-1].date()})")
    print(f" - X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape} ({dates_val[0].date()} to {dates_val[-1].date()})")
    print(f" - X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape} ({dates_test[0].date()} to {dates_test[-1].date()})")
    
    os.makedirs("data", exist_ok=True)
    np.save("data/spy_train_X.npy", X_train)
    np.save("data/spy_train_y.npy", y_train)
    np.save("data/spy_val_X.npy", X_val)
    np.save("data/spy_val_y.npy", y_val)
    np.save("data/spy_test_X.npy", X_test)
    np.save("data/spy_test_y.npy", y_test)
    
    # Also save with yahoo_ prefix for explicit naming
    np.save("data/yahoo_train_X.npy", X_train)
    np.save("data/yahoo_train_y.npy", y_train)
    np.save("data/yahoo_val_X.npy", X_val)
    np.save("data/yahoo_val_y.npy", y_val)
    np.save("data/yahoo_test_X.npy", X_test)
    np.save("data/yahoo_test_y.npy", y_test)
    
    train_df.to_csv("data/spy_train.csv")
    val_df.to_csv("data/spy_val.csv")
    test_df.to_csv("data/spy_test.csv")
    features_clean.to_csv("data/spy_full_features.csv")
    
    # Save selected feature names list for reference
    with open("data/selected_features.txt", "w") as f:
        f.write("\n".join(selected_features))
        
    print("\nSaved all split datasets to 'data/' folder.")
    return selected_features

if __name__ == "__main__":
    selected_features = preprocess_yahoo_data()
    print(f"\nFinal Selected 8 Features for KAN Input: {selected_features}")
