# ============================================
# Mini Project: Random Forest
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
age = np.random.randint(18, 70, n)
income = np.random.randint(20000, 120000, n)
credit_score = np.random.randint(300, 850, n)
# Target: Loan Approved if income > 50000 and credit_score > 600
approved = ((income > 50000) & (credit_score > 600)).astype(int)

df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Credit_Score': credit_score,
    'Approved': approved
})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Approved'].value_counts()}")

# Step 3: Split data
X = df[['Age', 'Income', 'Credit_Score']]
y = df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
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
features = ['Age', 'Income', 'Credit_Score']
print(f"\nFeature Importances:")
for f, imp in zip(features, importances):
    print(f"  {f}: {imp:.4f}")

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature importance bar chart
axes[0].barh(features, importances, color=['steelblue', 'tomato', 'mediumseagreen'], edgecolor='k')
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest Feature Importance')

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Rejected', 'Approved'])
axes[1].set_yticklabels(['Rejected', 'Approved'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()
