# ============================================
# Mini Project: K-Mode Clustering
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
# pip install kmodes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 200
colors = np.random.choice(['Red', 'Blue', 'Green'], n)
sizes = np.random.choice(['Small', 'Medium', 'Large'], n)
shapes = np.random.choice(['Circle', 'Square', 'Triangle'], n)
textures = np.random.choice(['Smooth', 'Rough'], n)

df = pd.DataFrame({
    'Color': colors,
    'Size': sizes,
    'Shape': shapes,
    'Texture': textures
})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

# Step 3: No train_test_split (Unsupervised Learning)

# Step 4: Train model
kmode = KModes(n_clusters=3, init='Huang', n_init=5, random_state=42)
clusters = kmode.fit_predict(df)

# Step 5: Assign clusters
df['Cluster'] = clusters

# Step 6: Evaluation
print(f"\nCluster Centroids (Modes):")
print(pd.DataFrame(kmode.cluster_centroids_, columns=['Color', 'Size', 'Shape', 'Texture']))
print(f"\nCost (Dissimilarity): {kmode.cost_}")
print(f"\nCluster Distribution:\n{df['Cluster'].value_counts()}")

# Elbow Method for K-Modes
costs = []
K_range = range(1, 8)
for k in K_range:
    km = KModes(n_clusters=k, init='Huang', n_init=5, random_state=42)
    km.fit(df[['Color', 'Size', 'Shape', 'Texture']])
    costs.append(km.cost_)

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cluster distribution bar chart
df['Cluster'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato', 'mediumseagreen'], edgecolor='k')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Count')
axes[0].set_title('K-Mode Cluster Distribution')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Elbow curve
axes[1].plot(K_range, costs, 'bo-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Cost (Dissimilarity)')
axes[1].set_title('Elbow Method for K-Modes')

plt.tight_layout()
plt.show()
