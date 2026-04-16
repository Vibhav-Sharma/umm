# ============================================
# Mini Project: Decision Tree (ID3 and CART)
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Create synthetic dataset
np.random.seed(42)
n = 300
temperature = np.random.randint(15, 45, n)
humidity = np.random.randint(20, 100, n)
wind_speed = np.random.randint(0, 50, n)
# Target: Play = 1 if temp < 35 and humidity < 70 and wind < 30
play = ((temperature < 35) & (humidity < 70) & (wind_speed < 30)).astype(int)

df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind_Speed': wind_speed,
    'Play': play
})
print("Dataset Head:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Play'].value_counts()}")

# Step 3: Split data
X = df[['Temperature', 'Humidity', 'Wind_Speed']]
y = df['Play']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Step 4: Train model
# --- ID3 uses entropy as criterion ---
id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
id3_model.fit(X_train, y_train)

# --- CART uses gini as criterion ---
cart_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
cart_model.fit(X_train, y_train)

# Step 5: Predict
y_pred_id3 = id3_model.predict(X_test)
y_pred_cart = cart_model.predict(X_test)

# Step 6: Evaluation
print("\n--- ID3 (Entropy) Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_id3):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_id3)}")

print("\n--- CART (Gini) Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_cart):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_cart)}")

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

plot_tree(id3_model, feature_names=['Temperature', 'Humidity', 'Wind_Speed'],
          class_names=['No Play', 'Play'], filled=True, rounded=True, ax=axes[0])
axes[0].set_title('ID3 Decision Tree (Entropy)')

plot_tree(cart_model, feature_names=['Temperature', 'Humidity', 'Wind_Speed'],
          class_names=['No Play', 'Play'], filled=True, rounded=True, ax=axes[1])
axes[1].set_title('CART Decision Tree (Gini)')

plt.tight_layout()
plt.show()
