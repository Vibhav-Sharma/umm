# ============================================
# Mini Project: Linear and Multiple Linear Regression
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
X1 = np.random.rand(200, 1) * 10          # Feature 1: Area
X2 = np.random.rand(200, 1) * 5           # Feature 2: Rooms
X3 = np.random.rand(200, 1) * 20          # Feature 3: Age
y = 3 * X1.flatten() + 7 * X2.flatten() - 1.5 * X3.flatten() + np.random.randn(200) * 2

df = pd.DataFrame({
    'Area': X1.flatten(),
    'Rooms': X2.flatten(),
    'Age': X3.flatten(),
    'Price': y
})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

# Step 3: Split data
X = df[['Area', 'Rooms', 'Age']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Step 4: Train model
# --- Simple Linear Regression (using only 'Area') ---
simple_model = LinearRegression()
simple_model.fit(X_train[['Area']], y_train)

# --- Multiple Linear Regression (using all features) ---
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Step 5: Predict
y_pred_simple = simple_model.predict(X_test[['Area']])
y_pred_multi = multi_model.predict(X_test)

# Step 6: Evaluation
print("\n--- Simple Linear Regression (Area only) ---")
print(f"MSE:  {mean_squared_error(y_test, y_pred_simple):.4f}")
print(f"R2:   {r2_score(y_test, y_pred_simple):.4f}")

print("\n--- Multiple Linear Regression (All Features) ---")
print(f"MSE:  {mean_squared_error(y_test, y_pred_multi):.4f}")
print(f"R2:   {r2_score(y_test, y_pred_multi):.4f}")
print(f"Coefficients: {multi_model.coef_}")
print(f"Intercept:    {multi_model.intercept_:.4f}")

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Simple Linear Regression plot
axes[0].scatter(X_test['Area'], y_test, color='steelblue', alpha=0.6, label='Actual')
axes[0].scatter(X_test['Area'], y_pred_simple, color='tomato', alpha=0.6, label='Predicted')
axes[0].set_xlabel('Area')
axes[0].set_ylabel('Price')
axes[0].set_title('Simple Linear Regression')
axes[0].legend()

# Multiple Linear Regression: Actual vs Predicted
axes[1].scatter(y_test, y_pred_multi, color='mediumseagreen', alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title('Multiple Linear Regression: Actual vs Predicted')

plt.tight_layout()
plt.show()
