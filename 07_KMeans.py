# ============================================
# Mini Project: K-Means Clustering
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
# Create 3 clusters
cluster1 = np.random.normal(loc=[2, 2], scale=0.8, size=(n // 3, 2))
cluster2 = np.random.normal(loc=[7, 7], scale=0.8, size=(n // 3, 2))
cluster3 = np.random.normal(loc=[12, 2], scale=0.8, size=(n // 3, 2))
X = np.vstack([cluster1, cluster2, cluster3])

df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

# Step 3: No train_test_split (Unsupervised Learning)

# Step 4: Train model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(df)

# Step 5: Predict
df['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 6: Evaluation
print(f"\nCluster Centers:\n{centroids}")
print(f"\nInertia (WCSS): {kmeans.inertia_:.4f}")
print(f"Silhouette Score: {silhouette_score(df[['Feature_1', 'Feature_2']], df['Cluster']):.4f}")

# Elbow Method
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df[['Feature_1', 'Feature_2']])
    wcss.append(km.inertia_)

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cluster plot
scatter = axes[0].scatter(df['Feature_1'], df['Feature_2'], c=df['Cluster'], cmap='viridis', alpha=0.6, edgecolors='k')
axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='k', label='Centroids')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('K-Means Clustering (K=3)')
axes[0].legend()

# Elbow curve
axes[1].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('WCSS (Inertia)')
axes[1].set_title('Elbow Method')

plt.tight_layout()
plt.show()
