import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

# 1. Load the EDA Dataset
# Assuming 'enzyme_graph_metrics.csv' was created in the previous EDA step
df = pd.read_csv('enzyme_graph_metrics.csv')

# 2. Preprocessing
# Select our 5 macroscopic structural features
X = df[['node_count', 'edge_count', 'radius', 'diameter', 'mean_eigen_cent']]
y = df['ec_class']

# Disconnected graphs might return NaN for radius/diameter.
# We impute missing values with the median of the respe~ctive columns.
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data (80% Training, 20% Testing)
# Stratify=y ensures all 6 EC classes are balanced in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Model Initialization and Training
# We use 300 trees, balance the classes, and enable OOB tracking so we can
# visualize how performance changes as more trees are added to the forest.
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    oob_score=True,
    warm_start=True,
)

# 3A. OOB Learning Curve
# Fit the forest a chunk at a time and record both training accuracy and
# out-of-bag accuracy, which acts as a built-in validation-style estimate.
n_estimators_range = list(range(10, 301, 10))
oob_accuracies: list[float] = []
train_accuracies: list[float] = []

for n in n_estimators_range:
    rf_model.set_params(n_estimators=n)
    rf_model.fit(X_train, y_train)
    oob_accuracies.append(rf_model.oob_score_)
    train_accuracies.append(accuracy_score(y_train, rf_model.predict(X_train)))

# 4. Predictions and Evaluation
y_pred = rf_model.predict(X_test)

print(f"=== Baseline Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Visualizations

# 5A. Confusion Matrix
ec_classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=ec_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ec_classes,
            yticklabels=ec_classes)
plt.title('Baseline Random Forest Confusion Matrix', fontsize=15)
plt.xlabel('Predicted EC Class', fontsize=12)
plt.ylabel('Actual (True) EC Class', fontsize=12)
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: rf_confusion_matrix.png")

# 5B. Learning Curve
plt.figure(figsize=(9, 5))
plt.plot(n_estimators_range, train_accuracies,
         label='Train Accuracy', color='#4CAF50', linewidth=2)
plt.plot(n_estimators_range, oob_accuracies,
         label='OOB Accuracy (validation proxy)', color='#F44336',
         linewidth=2, linestyle='--')
plt.title('Baseline Random Forest Learning Curve', fontsize=15)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('rf_learning_curve_accuracy.png', dpi=150)
plt.show()
print("Saved: rf_learning_curve_accuracy.png")

# 5C. Feature Importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
plt.title('Feature Importance (Which structural metrics matter most?)', fontsize=15)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Structural Metric', fontsize=12)
plt.tight_layout()
plt.savefig('rf_feature_importances.png', dpi=150)
plt.show()
print("Saved: rf_feature_importances.png")
