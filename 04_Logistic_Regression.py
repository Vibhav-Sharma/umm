# ============================================
# Mini Project: Logistic Regression
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 200
study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(40, 100, n)
# Target: Pass = 1 if study_hours > 4 and attendance > 60, with some noise
prob = 1 / (1 + np.exp(-(0.8 * study_hours + 0.05 * attendance - 7)))
passed = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Passed': passed
})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Passed'].value_counts()}")

# Step 3: Split data
X = df[['Study_Hours', 'Attendance']]
y = df['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Step 4: Train model
model = LogisticRegression(random_state=42)
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
plt.scatter(X_test['Study_Hours'], X_test['Attendance'], c=y_pred, cmap='bwr', alpha=0.7, edgecolors='k')
plt.xlabel('Study Hours')
plt.ylabel('Attendance (%)')
plt.title('Logistic Regression Classification')
plt.colorbar(label='Predicted Class (0=Fail, 1=Pass)')
plt.tight_layout()
plt.show()
