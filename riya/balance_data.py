import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = 'driving_log.csv'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv(CSV_PATH, names=columns)

print(f'Original total samples: {len(data)}')

# ---- Balance the dataset ----
num_bins = 25
samples_per_bin = 200   # <-- adjust this cap as needed; try 200-400

hist, bins = np.histogram(data['steering'], num_bins)

remove_indices = []
for j in range(num_bins):
    bin_indices = []
    for i in range(len(data['steering'])):
        if bins[j] <= data['steering'][i] <= bins[j + 1]:
            bin_indices.append(i)
    # shuffle so we remove randomly, not just the last N driven
    bin_indices = np.array(bin_indices)
    np.random.shuffle(bin_indices)
    remove_indices.extend(bin_indices[samples_per_bin:])

print(f'Removing {len(remove_indices)} over-represented samples')
data_balanced = data.drop(data.index[remove_indices]).reset_index(drop=True)
print(f'Balanced total samples: {len(data_balanced)}')

# Save the balanced CSV for use in training later
data_balanced.to_csv('driving_log_balanced.csv', index=False, header=False)
print('Saved balanced data to driving_log_balanced.csv')

# ---- Plot before/after comparison ----
hist_balanced, _ = np.histogram(data_balanced['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(center, hist, width=0.05)
axes[0].set_title('Before Balancing')
axes[0].set_xlabel('Steering Angle')
axes[0].set_ylabel('Number of Samples')

axes[1].bar(center, hist_balanced, width=0.05)
axes[1].set_title('After Balancing')
axes[1].set_xlabel('Steering Angle')

plt.tight_layout()
plt.savefig('steering_histogram_balanced.png')
print('Saved comparison chart to steering_histogram_balanced.png')
plt.show()
