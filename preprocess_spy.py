import preprocess_yahoo

if __name__ == "__main__":
    selected_features = preprocess_yahoo.preprocess_yahoo_data()
    print(f"\nFinal Selected 8 Features for KAN Input: {selected_features}")