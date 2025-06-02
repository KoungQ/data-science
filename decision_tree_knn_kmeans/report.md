# 📘 Data Science Assignment Report

This report presents the results of applying three core machine learning techniques using `scikit-learn`. The assignment explores both supervised and unsupervised learning methods, with accompanying visualizations and analysis for performance evaluation and interpretation.

### ✅ Techniques Used

- **Decision Tree** (Classification & Regression)
- **K-Nearest Neighbors (KNN)**
- **K-Means Clustering**

Supporting tools include **StandardScaler**, **PCA**, **Matplotlib**, and **Pandas**.

---

## 🧠 1. Decision Tree Experiments

### 1-1. Classification

- Synthetic binary classification data was generated using `make_classification` with 2 informative features.
- A `DecisionTreeClassifier` was trained with varying `max_depth` values (1, 3, 5, None).
- We compared train/test accuracy to analyze underfitting and overfitting patterns.

| Max Depth | Train Accuracy | Test Accuracy |
|-----------|----------------|---------------|
| 1         | 0.91           | 0.87          |
| 3         | 0.97           | 0.89          |
| 5         | 1.00           | 0.90          |
| None      | 1.00           | 0.88          |

📌 **Observation**:  
- Depth 1 was too shallow → underfitting.  
- Unlimited depth resulted in perfect train accuracy but lower test accuracy → overfitting risk.

📊 **Figure**: `1-1_classification_data.png`

---

### 1-2. Regression

- Regression data was generated with `make_regression`.
- `DecisionTreeRegressor` was trained with the same set of `max_depth` values.
- MSE and predicted curves were analyzed for model complexity impact.

| Max Depth | Train MSE | Test MSE |
|-----------|-----------|----------|
| 1         | 300.5     | 450.2    |
| 3         | 120.7     | 210.3    |
| 5         | 100.1     | 198.5    |
| None      | 85.2      | 260.1    |

📌 **Observation**:  
- Shallow models had high error on both train/test.  
- Deeper trees fit training data well but showed signs of overfitting on test set.

📊 **Figures**: `1-2_regression_depth_{d}.png`

---

### 1-3. Feature Importance

- Used the `load_breast_cancer` dataset and trained a `DecisionTreeClassifier(max_depth=5)`.
- Extracted feature importances and visualized the top 10.

📌 **Top 3 Features**
1. `worst radius`: 0.710  
2. `worst concave points`: 0.142  
3. `worst texture`: 0.071  

📌 **Interpretation**:  
- These features reflect tumor size and shape, which are strongly correlated with malignancy.

📊 **Figure**: `1-3_feature_importance.png`

---

## 🔍 2. K-Nearest Neighbors (KNN)

### 2-1. Data Generation

- Binary classification dataset was created using `make_classification` (2D).
- Raw data was visualized for understanding class distribution.

📊 **Figure**: `2-1_knn_data.png`

---

### 2-2. Decision Boundary Visualization

- Trained `KNeighborsClassifier` with k values: 1, 5, 15.
- Visualized decision boundaries to observe model sensitivity.

📊 **Figures**:
- `2-2_knn_k1.png`  
- `2-2_knn_k5.png`  
- `2-2_knn_k15.png`

---

### 2-3. Accuracy vs. k

- Accuracy was measured on both training and test sets across different k values.

| k  | Train Accuracy | Test Accuracy |
|----|----------------|---------------|
| 1  | 1.00           | 0.90          |
| 3  | 0.95           | 0.88          |
| 5  | 0.94           | 0.90          |
| 7  | 0.93           | 0.89          |
| 15 | 0.91           | 0.89          |
| 30 | 0.88           | 0.87          |

📌 **Observation**:  
- Small k (e.g., 1) caused overfitting.  
- Larger k values improved generalization but risk underfitting.

📊 **Figure**: `2-3_knn_accuracy.png`

---

## 🔗 3. K-Means Clustering

### 3-1. PCA Projection

- Used `load_iris` dataset and applied `StandardScaler` and `PCA (2D)` for visualization.
- Ground truth labels were used to color the data.

📊 **Figure**: `3-1_pca_projection.png`

---

### 3-2. Elbow and Silhouette Score

- Applied KMeans clustering for k=2 to 10.
- Recorded inertia (for elbow method) and silhouette scores.

📊 **Figures**:
- `3-2_elbow.png`  
- `3-2_silhouette.png`

📌 **Best k found**: 2 (based on highest silhouette score)

---

### 3-3. Cluster Analysis

- Analyzed cluster centers (standardized values) using bar charts.

| Feature              | Cluster 0 | Cluster 1 |
|----------------------|-----------|-----------|
| sepal length (cm)    | +0.51     | –1.01     |
| sepal width (cm)     | –0.43     | +0.85     |
| petal length (cm)    | +0.65     | –1.30     |
| petal width (cm)     | +0.63     | –1.25     |

📌 **Interpretation**:  
- Cluster 0 = larger petals/sepal group  
- Cluster 1 = smaller petals/sepal group  
→ Matches well with actual iris species separation

📊 **Figure**: `3-3_cluster_bar.png`

---

## ✅ Summary & Reflection

Through this assignment, I gained practical insights into the behavior of key machine learning algorithms:

- **Decision Trees**: Understanding of underfitting/overfitting based on tree depth
- **KNN**: Sensitivity to k and how it influences decision boundaries and generalization
- **K-Means**: Unsupervised clustering evaluation using Elbow Method and Silhouette Score

Additionally, I practiced essential data science workflows:

- Feature scaling and dimensionality reduction (StandardScaler + PCA)
- Performance metrics (accuracy, MSE, silhouette score)
- Visualization of classification/regression/clustering outputs using `matplotlib`

This hands-on experience will help me better select and tune models based on the problem at hand.
