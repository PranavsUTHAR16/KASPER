import numpy as np
import pandas as pd
import os

def check_spy():
    data_dir = "data"
    print("Files in data directory:", os.listdir(data_dir) if os.path.exists(data_dir) else "No data dir")
    for f in ["spy_train_X.npy", "spy_train_y.npy", "spy_val_X.npy", "spy_val_y.npy", "spy_test_X.npy", "spy_test_y.npy"]:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            arr = np.load(p)
            print(f" - {f:18s}: shape {arr.shape}, min {arr.min():+.4f}, max {arr.max():+.4f}")

if __name__ == "__main__":
    check_spy()
