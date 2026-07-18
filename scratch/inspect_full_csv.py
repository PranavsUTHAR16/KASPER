import pandas as pd

def check_csv():
    df = pd.read_csv("data/spy_full_features.csv")
    print("Shape of spy_full_features.csv:", df.shape)
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    check_csv()
