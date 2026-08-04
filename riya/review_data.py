import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---- EDIT THIS to point to your driving_log.csv ----
DATA_PATH = '.'  # driving_log.csv is in the project root  # folder containing driving_log.csv and IMG/
CSV_PATH = os.path.join(DATA_PATH, 'driving_log.csv')

# The Udacity simulator writes no header row, columns are always in this order:
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv(CSV_PATH, names=columns)

print(f'Total samples: {len(data)}')
print(data.head())

# Keep only the filename, not the full recording-machine path
# (the simulator saves absolute paths from YOUR machine, which we'll need
#  to fix before training anyway, so let's check that now too)
sample_path = data['center'].iloc[0]
print(f'\nExample center image path as stored in CSV:\n{sample_path}')

# ---- Plot histogram of steering angles ----
num_bins = 25
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5

plt.figure(figsize=(10, 5))
plt.bar(center, hist, width=0.05)
plt.title('Steering Angle Distribution')
plt.xlabel('Steering Angle')
plt.ylabel('Number of Samples')
plt.axhline(y=np.mean(hist), color='r', linestyle='--', label='Average per bin')
plt.legend()
plt.tight_layout()
plt.savefig('steering_histogram.png')
print('\nSaved histogram to steering_histogram.png')
plt.show()
