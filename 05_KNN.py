# ============================================
# Mini Project: K-Nearest Neighbors (KNN)
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
# Three clusters for 3 classes
X1 = np.concatenate([np.random.normal(2, 1, n // 3), np.random.normal(6, 1, n // 3), np.random.normal(10, 1, n // 3)])
X2 = np.concatenate([np.random.normal(2, 1, n // 3), np.random.normal(8, 1, n // 3), np.random.normal(3, 1, n // 3)])
y = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n // 3))

df = pd.DataFrame({'Feature_1': X1, 'Feature_2': X2, 'Target': y})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Target'].value_counts()}")

# Step 3: Split data
X = df[['Feature_1', 'Feature_2']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Step 4: Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("\n--- Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Step 7: Visualization
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test['Feature_1'], X_test['Feature_2'], c=y_pred, cmap='viridis', alpha=0.7, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Classification (K=5)')
plt.colorbar(scatter, label='Predicted Class')
plt.tight_layout()
plt.show()
