import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import (
    make_classification, make_regression,
    load_breast_cancer, load_iris
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# 결과 저장 디렉토리
output_dir = "decision_tree_knn_kmeans"
os.makedirs(output_dir, exist_ok=True)

### 1. Decision Tree
# 1-1 Classification
X1, y1 = make_classification(n_samples=300, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap='bwr', edgecolor='k')
plt.title("1-1. Classification Data")
plt.savefig(f"{output_dir}/1-1_classification_data.png")
plt.close()

clf_depths = [1, 3, 5, None]
clf_results = []
for d in clf_depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X1_train, y1_train)
    train_acc = accuracy_score(y1_train, model.predict(X1_train))
    test_acc = accuracy_score(y1_test, model.predict(X1_test))
    clf_results.append((str(d), train_acc, test_acc))
df_clf = pd.DataFrame(clf_results, columns=["Max Depth", "Train Accuracy", "Test Accuracy"])
print("\n[1-1] Decision Tree Classification Results")
print(df_clf)

# 1-2 Regression
X2, y2 = make_regression(n_samples=300, n_features=1, noise=15, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
reg_results = []
x_grid = np.linspace(X2.min(), X2.max(), 300).reshape(-1, 1)

for d in clf_depths:
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(X2_train, y2_train)
    train_mse = mean_squared_error(y2_train, model.predict(X2_train))
    test_mse = mean_squared_error(y2_test, model.predict(X2_test))
    reg_results.append((str(d), train_mse, test_mse))

    y_pred_grid = model.predict(x_grid)
    plt.scatter(X2_train, y2_train, label='Train', alpha=0.6)
    plt.scatter(X2_test, y2_test, label='Test', alpha=0.6)
    plt.plot(x_grid, y_pred_grid, color='red', label='Prediction')
    plt.title(f"1-2. Regression (depth={d})")
    plt.legend()
    plt.savefig(f"{output_dir}/1-2_regression_depth_{d}.png")
    plt.close()

df_reg = pd.DataFrame(reg_results, columns=["Max Depth", "Train MSE", "Test MSE"])
print("\n[1-2] Decision Tree Regression Results")
print(df_reg)

# 1-3 Feature Importance
data_bc = load_breast_cancer()
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(data_bc.data, data_bc.target)
importances = model.feature_importances_
features = data_bc.feature_names
indices = np.argsort(importances)[::-1]
plt.barh(range(10), importances[indices[:10]][::-1])
plt.yticks(range(10), features[indices[:10]][::-1])
plt.title("1-3. Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(f"{output_dir}/1-3_feature_importance.png")
plt.close()
top3 = [(features[indices[i]], importances[indices[i]]) for i in range(3)]
print("\n[1-3] Top 3 Important Features")
for f, v in top3:
    print(f"{f}: {v:.3f}")

### 2. KNN Classification
X3, y3 = make_classification(n_samples=300, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1, random_state=0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0)
plt.scatter(X3[:, 0], X3[:, 1], c=y3, cmap='bwr', edgecolor='k')
plt.title("2-1. KNN Data")
plt.savefig(f"{output_dir}/2-1_knn_data.png")
plt.close()

def plot_boundary(model, X, y, filename):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    h = 0.1
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(filename.split("/")[-1])
    plt.savefig(filename)
    plt.close()

knn_results = []
for k in [1, 3, 5, 7, 15, 30]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X3_train, y3_train)
    train_acc = accuracy_score(y3_train, model.predict(X3_train))
    test_acc = accuracy_score(y3_test, model.predict(X3_test))
    knn_results.append((k, train_acc, test_acc))
    if k in [1, 5, 15]:
        plot_boundary(model, X3, y3, f"{output_dir}/2-2_knn_k{k}.png")

df_knn = pd.DataFrame(knn_results, columns=["k", "Train Accuracy", "Test Accuracy"])
print("\n[2] KNN Accuracy")
print(df_knn)

plt.plot(df_knn["k"], df_knn["Train Accuracy"], label="Train", marker='o')
plt.plot(df_knn["k"], df_knn["Test Accuracy"], label="Test", marker='s')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.title("2-3. Accuracy vs k")
plt.savefig(f"{output_dir}/2-3_knn_accuracy.png")
plt.close()

### 3. K-Means Clustering
iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='Set1', edgecolor='k')
plt.title("3-1. PCA Projection (True Labels)")
plt.savefig(f"{output_dir}/3-1_pca_projection.png")
plt.close()

inertias = []
sil_scores = []
best_k = 0
best_score = -1
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    sil_scores.append(score)
    if score > best_score:
        best_k = k
        best_score = score

plt.plot(range(2, 11), inertias, marker='o')
plt.title("3-2. Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.savefig(f"{output_dir}/3-2_elbow.png")
plt.close()

plt.plot(range(2, 11), sil_scores, marker='s', color='green')
plt.title("3-2. Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")
plt.savefig(f"{output_dir}/3-2_silhouette.png")
plt.close()

final_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
final_model.fit(X_scaled)
centers = pd.DataFrame(final_model.cluster_centers_, columns=iris.feature_names)
centers.index = [f"Cluster {i}" for i in range(best_k)]
print(f"\n[3-3] Best k: {best_k}")
print(centers)

centers.T.plot(kind='bar')
plt.title("3-3. Cluster Feature Means")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/3-3_cluster_bar.png")
plt.close()
