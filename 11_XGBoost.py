# ============================================
# Mini Project: XGBoost
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# pip install xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
feature1 = np.random.uniform(0, 10, n)
feature2 = np.random.uniform(0, 10, n)
feature3 = np.random.uniform(0, 10, n)
# Non-linear decision boundary
y = ((feature1 ** 2 + feature2 > 30) | (feature3 > 7)).astype(int)

df = pd.DataFrame({
    'Feature_1': feature1,
    'Feature_2': feature2,
    'Feature_3': feature3,
    'Target': y
})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Target'].value_counts()}")

# Step 3: Split data
X = df[['Feature_1', 'Feature_2', 'Feature_3']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Step 4: Train model
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("\n--- Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Feature importance
importances = model.feature_importances_
features = ['Feature_1', 'Feature_2', 'Feature_3']
print(f"\nFeature Importances:")
for f, imp in zip(features, importances):
    print(f"  {f}: {imp:.4f}")

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature importance
axes[0].barh(features, importances, color=['steelblue', 'tomato', 'mediumseagreen'], edgecolor='k')
axes[0].set_xlabel('Importance')
axes[0].set_title('XGBoost Feature Importance')

# Actual vs Predicted
axes[1].scatter(range(len(y_test)), y_test, color='steelblue', alpha=0.6, label='Actual', marker='o')
axes[1].scatter(range(len(y_pred)), y_pred, color='tomato', alpha=0.4, label='Predicted', marker='x')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Class')
axes[1].set_title('XGBoost: Actual vs Predicted')
axes[1].legend()

plt.tight_layout()
plt.show()
