import pandas as pd
import matplotlib.pyplot as plt

# Load CSV without headers
data = pd.read_csv("data/driving_log.csv", header=None)

# Convert steering column to numbers
steering = pd.to_numeric(data[3], errors="coerce")

# Remove zero values
non_zero = steering[steering != 0]

print(non_zero.describe())

plt.figure(figsize=(10,5))
plt.hist(non_zero, bins=50)

print("Total samples:", len(steering))
print("Zero steering:", (steering == 0).sum())
print("Non-zero steering:", (steering != 0).sum())
print("Percentage zero:", (steering == 0).mean() * 100)
print("Range:", steering.min(), steering.max())

plt.xlabel("Steering Angle")
plt.ylabel("Count")
plt.title("Non-zero Steering Distribution")

plt.show()