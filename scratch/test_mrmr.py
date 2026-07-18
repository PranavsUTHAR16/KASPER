import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

def mrmr_feature_selection(X, y, feature_names, k=8, penalty_weight=10.0):
    f_scores, _ = f_regression(X, y)
    corr_matrix = np.abs(np.corrcoef(X.T))

    selected_indices = []
    remaining_indices = list(range(len(feature_names)))

    # Step 1: Select highest F-score feature
    first_choice = int(np.argmax(f_scores))
    selected_indices.append(first_choice)
    remaining_indices.remove(first_choice)

    # Step 2..k: Select feature maximizing (f_score - penalty * mean_correlation)
    for _ in range(1, k):
        best_score = -float('inf')
        best_idx = None

        for candidate in remaining_indices:
            relevance = f_scores[candidate]
            redundancy = np.mean([corr_matrix[candidate, sel] for sel in selected_indices])
            mrmr_score = relevance - penalty_weight * redundancy

            if mrmr_score > best_score:
                best_score = mrmr_score
                best_idx = candidate

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [feature_names[i] for i in selected_indices]

def run_test():
    df = pd.read_csv("data/spy_full_features.csv")
    train_df = df.iloc[:1040].copy()
    candidate_cols = [
        "HL_Spread", "OC_Spread", "Log_Return_1d", "Log_Return_7d",
        "Log_Return_High_1d", "Log_Return_Low_1d", "Log_Return_Open_1d",
        "Log_Return_Volume_1d", "Rolling_Mean_Return_21d", "Rolling_Volatility_21d",
        "Volatility_Ratio_21d", "ATR_21d", "Velocity", "Acceleration", "Delta_Volume"
    ]
    X_train = train_df[candidate_cols].values
    y_train = train_df["Target_Return_Next_Day"].values

    selected = mrmr_feature_selection(X_train, y_train, candidate_cols, k=8, penalty_weight=15.0)
    print("=" * 80)
    print("mRMR-Selected 8 Non-Redundant Features (train-only fit):")
    print("=" * 80)
    for idx, f in enumerate(selected, 1):
        print(f"  {idx}. {f}")

if __name__ == "__main__":
    run_test()
