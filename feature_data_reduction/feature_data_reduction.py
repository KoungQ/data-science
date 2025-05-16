import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# 시각화 저장 폴더 생성
os.makedirs("clean_fix_scale", exist_ok=True)

# 데이터 로딩
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# [Part 1] 초기 원본 feature 시각화
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X['alcohol'], y=X['malic_acid'], hue=y, palette='Set1')
plt.title("Original Features: Alcohol vs Malic Acid")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.tight_layout()
plt.savefig("feature_data_reduction/original_scatter.png")
plt.close()

# [Part 1] PCA 적용 (2차원 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA 시각화
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1')
plt.title("PCA Result (PC1 vs PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("feature_data_reduction/pca_scatter.png")
plt.close()

# PCA 설명된 분산 비율 출력
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# [Part 2-1] Random Sampling (30%)
X_random, _, y_random, _ = train_test_split(X_scaled, y, test_size=0.7, random_state=42)
pca_random = PCA(n_components=2).fit_transform(X_random)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=pca_random[:, 0], y=pca_random[:, 1], hue=y_random, palette='Set1')
plt.title("Random Sampling (30%) PCA")
plt.tight_layout()
plt.savefig("feature_data_reduction/random_sample_pca.png")
plt.close()

# [Part 2-1] Stratified Sampling (30%)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for _, sample_idx in sss.split(X_scaled, y):
    X_stratified = X_scaled[sample_idx]
    y_stratified = y.iloc[sample_idx]

pca_stratified = PCA(n_components=2).fit_transform(X_stratified)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=pca_stratified[:, 0], y=pca_stratified[:, 1], hue=y_stratified, palette='Set1')
plt.title("Stratified Sampling (30%) PCA")
plt.tight_layout()
plt.savefig("feature_data_reduction/stratified_sample_pca.png")
plt.close()

# [Part 2-2] Binning
alcohol = X['alcohol']
malic_acid = X['malic_acid']

X['alcohol_eq_width'] = pd.cut(alcohol, bins=3, labels=["low", "mid", "high"])
X['alcohol_eq_freq'] = pd.qcut(alcohol, q=3, labels=["low", "mid", "high"])

# Equal-width binning 시각화
plt.figure(figsize=(6, 5))
sns.scatterplot(x=alcohol, y=malic_acid, hue=X['alcohol_eq_width'], palette='Set2')
plt.title("Equal-Width Binning: Alcohol vs Malic Acid")
plt.tight_layout()
plt.savefig("feature_data_reduction/equal_width_binning.png")
plt.close()

# Equal-frequency binning 시각화
plt.figure(figsize=(6, 5))
sns.scatterplot(x=alcohol, y=malic_acid, hue=X['alcohol_eq_freq'], palette='Set2')
plt.title("Equal-Frequency Binning: Alcohol vs Malic Acid")
plt.tight_layout()
plt.savefig("feature_data_reduction/equal_freq_binning.png")
plt.close()
