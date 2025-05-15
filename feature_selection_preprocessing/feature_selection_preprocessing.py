import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier

# Part 1: Load and normalize dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dataset summary
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])
print("Class Distribution (0=malignant, 1=benign):")
print(pd.Series(y).value_counts())

# Part 2A: Chi-Square Feature Selection
chi2_selector = SelectKBest(score_func=chi2, k="all")
X_chi2 = chi2_selector.fit_transform(X_scaled, y)
chi2_scores = chi2_selector.scores_

# Chi2 Feature Importance Visualization
plt.figure(figsize=(10, 4))
plt.barh(range(len(chi2_scores)), chi2_scores, align='center')
plt.yticks(np.arange(len(chi2_scores)), feature_names)
plt.xlabel("Chi-Square Score")
plt.title("Chi-Square Feature Scores")
plt.tight_layout()
plt.savefig("feature_selection_preprocessing/chi2_scores.png")

top5_chi2 = np.argsort(chi2_scores)[-5:][::-1]
top5_chi2_features = feature_names[top5_chi2]

# Part 2B: Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)
lasso_coeffs = lasso.coef_

top_lasso_features = feature_names[lasso_coeffs != 0]

# Part 2C: ExtraTreesClassifier
tree_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
tree_clf.fit(X_scaled, y)
tree_importances = tree_clf.feature_importances_

# Tree Feature Importance Visualization
plt.figure(figsize=(10, 4))
plt.barh(range(len(tree_importances)), tree_importances, align='center')
plt.yticks(np.arange(len(tree_importances)), feature_names)
plt.xlabel("Feature Importance")
plt.title("ExtraTrees Feature Importances")
plt.tight_layout()
plt.savefig("feature_selection_preprocessing/tree_feature_importances.png")

top5_tree = np.argsort(tree_importances)[-5:][::-1]
top5_tree_features = feature_names[top5_tree]

# Part 3: Comparison & Reflection
print("\n[Top 5 Features by Chi-Square]")
print(top5_chi2_features)

print("\n[Non-zero Coefficients in Lasso]")
print(top_lasso_features)

print("\n[Top 5 Features by ExtraTreesClassifier]")
print(top5_tree_features)

"""
Reflection:

1. Common features across methods:
- 'mean concave points'
- 'worst concave points'

2. Features selected by some but not all:
- Lasso selected fewer features (sparse model).
- ExtraTrees and Chi2 both ranked 'worst area' and 'worst perimeter' highly,
  but Lasso did not include them.

3. Most trustworthy method: ExtraTreesClassifier
- Handles non-linear relationships and feature interactions.
- Less sensitive to scaling.
- Produces stable and interpretable feature importances.
"""
