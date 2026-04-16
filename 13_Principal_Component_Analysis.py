# ============================================
# Mini Project: Principal Component Analysis (PCA)
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
feature1 = np.random.normal(5, 2, n)
feature2 = feature1 * 0.8 + np.random.normal(0, 1, n)     # Correlated with feature1
feature3 = np.random.normal(10, 3, n)
feature4 = feature3 * 0.5 + np.random.normal(0, 1.5, n)   # Correlated with feature3
feature5 = np.random.normal(0, 1, n)                        # Independent noise
labels = np.where(feature1 + feature3 > 15, 1, 0)           # For coloring

df = pd.DataFrame({
    'Feature_1': feature1,
    'Feature_2': feature2,
    'Feature_3': feature3,
    'Feature_4': feature4,
    'Feature_5': feature5,
    'Label': labels
})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

# Step 3: Standardize features (no train_test_split for PCA)
X = df[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Transform data
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Label'] = labels

# Step 6: Evaluation
print(f"\nExplained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Explained Variance:  {pca.explained_variance_ratio_.sum():.4f}")
print(f"\nPCA Components (Loadings):")
loadings = pd.DataFrame(pca.components_, columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'], index=['PC1', 'PC2'])
print(loadings)

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA scatter plot
scatter = axes[0].scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Label'], cmap='viridis', alpha=0.6, edgecolors='k')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('PCA: 2D Projection')
plt.colorbar(scatter, ax=axes[0], label='Label')

# Explained variance bar chart
pca_full = PCA()
pca_full.fit(X_scaled)
axes[1].bar(range(1, 6), pca_full.explained_variance_ratio_, color='steelblue', edgecolor='k', alpha=0.7)
axes[1].plot(range(1, 6), np.cumsum(pca_full.explained_variance_ratio_), 'ro-', linewidth=2, label='Cumulative')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Scree Plot')
axes[1].legend()

plt.tight_layout()
plt.show()
