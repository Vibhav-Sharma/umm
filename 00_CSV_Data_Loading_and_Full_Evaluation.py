# ============================================
# How to Load Data from CSV File
# + Full Evaluation (Confusion Matrix, F-Score, Precision, Recall, etc.)
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# Step 2A: Create a sample CSV file (for demo)
# ============================================
# In real projects, you already have a CSV. This step creates one for demonstration.
np.random.seed(42)
n = 500
demo_data = pd.DataFrame({
    'Age': np.random.randint(18, 70, n),
    'Income': np.random.randint(20000, 120000, n),
    'Credit_Score': np.random.randint(300, 850, n),
    'Loan_Amount': np.random.randint(5000, 50000, n),
    'Employment_Years': np.random.randint(0, 30, n),
    'Approved': ((np.random.randint(20000, 120000, n) > 50000) &
                 (np.random.randint(300, 850, n) > 600)).astype(int)
})
# Save to CSV
demo_data.to_csv('sample_dataset.csv', index=False)
print(">>> sample_dataset.csv created successfully!\n")

# ============================================
# Step 2B: Load data from CSV file
# ============================================
# -------------------------------------------
# METHOD 1: Load from local file
# -------------------------------------------
df = pd.read_csv('sample_dataset.csv')

# -------------------------------------------
# METHOD 2: Load from Google Drive (Colab)
# -------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/MyDrive/your_folder/your_file.csv')

# -------------------------------------------
# METHOD 3: Upload file manually in Colab
# -------------------------------------------
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv('your_file.csv')

# -------------------------------------------
# METHOD 4: Load from URL
# -------------------------------------------
# url = 'https://raw.githubusercontent.com/user/repo/main/data.csv'
# df = pd.read_csv(url)

# ============================================
# Step 2C: Explore the dataset
# ============================================
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"\nShape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:")
print(df.describe())
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nTarget Distribution:\n{df['Approved'].value_counts()}")

# Step 3: Split data
X = df.drop('Approved', axis=1)   # Features (all columns except target)
y = df['Approved']                 # Target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC

# ============================================
# Step 6: FULL EVALUATION METRICS
# ============================================
print("\n" + "=" * 50)
print("COMPLETE EVALUATION REPORT")
print("=" * 50)

# --- Accuracy ---
acc = accuracy_score(y_test, y_pred)
print(f"\n1. Accuracy:          {acc:.4f}")

# --- Precision ---
prec = precision_score(y_test, y_pred)
print(f"2. Precision:         {prec:.4f}")

# --- Recall (Sensitivity) ---
rec = recall_score(y_test, y_pred)
print(f"3. Recall:            {rec:.4f}")

# --- F1-Score ---
f1 = f1_score(y_test, y_pred)
print(f"4. F1-Score:          {f1:.4f}")

# --- F-Beta Scores ---
from sklearn.metrics import fbeta_score
f05 = fbeta_score(y_test, y_pred, beta=0.5)
f2 = fbeta_score(y_test, y_pred, beta=2)
print(f"5. F0.5-Score:        {f05:.4f}")
print(f"6. F2-Score:          {f2:.4f}")

# --- Specificity ---
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print(f"7. Specificity:       {specificity:.4f}")

# --- Matthews Correlation Coefficient ---
mcc = matthews_corrcoef(y_test, y_pred)
print(f"8. MCC:               {mcc:.4f}")

# --- ROC-AUC Score ---
roc_auc = roc_auc_score(y_test, y_prob)
print(f"9. ROC-AUC:           {roc_auc:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
print(f"\n--- Confusion Matrix ---")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")
print(f"\n{cm}")

# --- Full Classification Report ---
print(f"\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# --- All metrics in a summary table ---
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F0.5-Score',
               'F2-Score', 'Specificity', 'MCC', 'ROC-AUC'],
    'Score': [acc, prec, rec, f1, f05, f2, specificity, mcc, roc_auc]
})
print("\n--- Summary Table ---")
print(metrics_df.to_string(index=False))

# ============================================
# Step 7: Visualization
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'], ax=axes[0, 0], cbar=True,
            annot_kws={'size': 16, 'fontweight': 'bold'})
axes[0, 0].set_xlabel('Predicted', fontsize=12)
axes[0, 0].set_ylabel('Actual', fontsize=12)
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
axes[0, 1].fill_between(fpr, tpr, alpha=0.1, color='blue')
axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)

# Plot 3: All Metrics Bar Chart
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#16a085']
bars = axes[1, 0].bar(metrics_df['Metric'], metrics_df['Score'], color=colors, edgecolor='k', alpha=0.85)
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_title('All Evaluation Metrics', fontsize=14, fontweight='bold')
axes[1, 0].set_xticklabels(metrics_df['Metric'], rotation=45, ha='right')
for bar, val in zip(bars, metrics_df['Score']):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4: Feature Importance
importances = model.feature_importances_
feat_names = X.columns
sorted_idx = np.argsort(importances)
axes[1, 1].barh(feat_names[sorted_idx], importances[sorted_idx], color='steelblue', edgecolor='k')
axes[1, 1].set_xlabel('Importance', fontsize=12)
axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
