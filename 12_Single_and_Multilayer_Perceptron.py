# ============================================
# Mini Project: Single and Multilayer Perceptron
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
X1 = np.concatenate([np.random.normal(2, 1, n // 2), np.random.normal(6, 1, n // 2)])
X2 = np.concatenate([np.random.normal(3, 1, n // 2), np.random.normal(7, 1, n // 2)])
y = np.array([0] * (n // 2) + [1] * (n // 2))

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
# --- Single Layer Perceptron ---
slp_model = Perceptron(max_iter=1000, random_state=42)
slp_model.fit(X_train, y_train)

# --- Multilayer Perceptron ---
mlp_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# Step 5: Predict
y_pred_slp = slp_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test)

# Step 6: Evaluation
print("\n--- Single Layer Perceptron ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_slp):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_slp)}")

print("\n--- Multilayer Perceptron ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_mlp)}")
print(f"\nClassification Report (MLP):\n{classification_report(y_test, y_pred_mlp)}")

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
h = 0.2

# Single Layer Perceptron decision boundary
x_min, x_max = X['Feature_1'].min() - 1, X['Feature_1'].max() + 1
y_min, y_max = X['Feature_2'].min() - 1, X['Feature_2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_slp = slp_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z_slp, alpha=0.3, cmap='coolwarm')
axes[0].scatter(X_test['Feature_1'], X_test['Feature_2'], c=y_test, cmap='coolwarm', edgecolors='k', alpha=0.7)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Single Layer Perceptron')

# Multilayer Perceptron decision boundary
Z_mlp = mlp_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z_mlp, alpha=0.3, cmap='coolwarm')
axes[1].scatter(X_test['Feature_1'], X_test['Feature_2'], c=y_test, cmap='coolwarm', edgecolors='k', alpha=0.7)
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('Multilayer Perceptron (16, 8)')

plt.tight_layout()
plt.show()
