import pandas as pd
import os

INPUT_CSV = "data/driving_log.csv"
OUTPUT_CSV = "data/driving_log_fixed.csv"
IMG_FOLDER = "data/IMG"

data = pd.read_csv(INPUT_CSV, header=None)

def fix_path(path):
    filename = os.path.basename(str(path).strip())
    return os.path.join(IMG_FOLDER, filename)

# Columns 0, 1, 2 are center, left, right image paths
for col in [0, 1, 2]:
    data[col] = data[col].apply(fix_path)

data.to_csv(OUTPUT_CSV, header=False, index=False)
print(f"Saved fixed paths to {OUTPUT_CSV}")
print(f"Total rows: {len(data)}")
print("Example fixed center path:", data.iloc[0, 0])
