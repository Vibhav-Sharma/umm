# ============================================
# Mini Project: Self Organizing Maps (SOM)
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
# pip install minisom
from minisom import MiniSom
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
# Create 3 distinct clusters
cluster1 = np.random.normal(loc=[2, 8], scale=0.8, size=(n // 3, 2))
cluster2 = np.random.normal(loc=[8, 2], scale=0.8, size=(n // 3, 2))
cluster3 = np.random.normal(loc=[8, 8], scale=0.8, size=(n // 3, 2))
X = np.vstack([cluster1, cluster2, cluster3])
labels = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n // 3))

df = pd.DataFrame({'Feature_1': X[:, 0], 'Feature_2': X[:, 1], 'Label': labels})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

# Step 3: Normalize data (no train_test_split for SOM)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train SOM
som_grid_rows = 10
som_grid_cols = 10
som = MiniSom(som_grid_rows, som_grid_cols, X_scaled.shape[1],
              sigma=1.5, learning_rate=0.5, random_seed=42)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=500)

# Step 5: Map data to SOM
win_map = som.win_map(X_scaled)
bmu_indices = np.array([som.winner(x) for x in X_scaled])

# Step 6: Evaluation
print(f"\nSOM Grid Size: {som_grid_rows} x {som_grid_cols}")
print(f"Quantization Error: {som.quantization_error(X_scaled):.4f}")
print(f"Topographic Error:  {som.topographic_error(X_scaled):.4f}")

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# U-Matrix (Distance Map)
axes[0].imshow(som.distance_map().T, cmap='bone_r', origin='lower')
axes[0].set_title('SOM U-Matrix (Distance Map)')
axes[0].set_xlabel('SOM X')
axes[0].set_ylabel('SOM Y')

# Scatter plot on SOM grid colored by original labels
colors = ['steelblue', 'tomato', 'mediumseagreen']
markers = ['o', 's', '^']
for i, (x, label) in enumerate(zip(X_scaled, labels)):
    w = som.winner(x)
    axes[1].scatter(w[0] + 0.5 + np.random.uniform(-0.2, 0.2),
                    w[1] + 0.5 + np.random.uniform(-0.2, 0.2),
                    color=colors[label], marker=markers[label], s=20, alpha=0.6)

axes[1].set_xlim([0, som_grid_cols])
axes[1].set_ylim([0, som_grid_rows])
axes[1].set_title('SOM: Data Points Mapped to Grid')
axes[1].set_xlabel('SOM X')
axes[1].set_ylabel('SOM Y')

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='Cluster 0'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='tomato', markersize=10, label='Cluster 1'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='mediumseagreen', markersize=10, label='Cluster 2')]
axes[1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
