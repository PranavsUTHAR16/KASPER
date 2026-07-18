import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

def audit_feature_redundancy():
    df = pd.read_csv("data/spy_full_features.csv")

    # Use training split only (N=1040) for strict zero-lookahead discipline
    train_df = df.iloc[:1040].copy()
    feature_cols = [c for c in train_df.columns if c not in ["Date", "Target_Return_Next_Day"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["Target_Return_Next_Day"].values

    # 1. Compute f_regression scores & p-values for all 16 candidate features
    f_scores, p_values = f_regression(X_train, y_train)
    f_rank_df = pd.DataFrame({
        "Feature": feature_cols,
        "F_Score": f_scores,
        "p_value": p_values
    }).sort_values(by="F_Score", ascending=False).reset_index(drop=True)

    print("=" * 90)
    print("STEP 1 AUDIT: ALL 16 CANDIDATE FEATURES RANKED BY INDIVIDUAL F-REGRESSION SCORE")
    print("=" * 90)
    for idx, row in f_rank_df.iterrows():
        is_selected = " [SELECTED BY K=8]" if idx < 8 else ""
        print(f"{idx+1:2d}. {row['Feature']:25s} | F-Score: {row['F_Score']:8.4f} | p-value: {row['p_value']:.4e}{is_selected}")

    # 2. Pairwise Correlation Matrix
    corr_df = pd.DataFrame(np.corrcoef(X_train.T), index=feature_cols, columns=feature_cols)

    print("\n" + "=" * 90)
    print("STEP 1 AUDIT: PAIRWISE CORRELATIONS IN SAME-DAY RETURN CLUSTER")
    print("=" * 90)
    return_cluster = ["Log_Return_1d", "Log_Return_High_1d", "Log_Return_Low_1d", "Log_Return_Open_1d"]
    cluster_corr = corr_df.loc[return_cluster, return_cluster]
    print(cluster_corr.round(4))

    print("\n" + "=" * 90)
    print("STEP 1 AUDIT: CORRELATION OF HL_SPREAD VS SELECTED RETURN CLUSTER & OC_SPREAD")
    print("=" * 90)
    target_feats = return_cluster + ["OC_Spread", "HL_Spread", "Volatility_Ratio_21d", "ATR_21d", "Volume_State_Ratio"]
    print(corr_df.loc[target_feats, target_feats].round(4))

if __name__ == "__main__":
    audit_feature_redundancy()
