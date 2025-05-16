# Data Science Assignment: Feature & Data Reduction

## Dataset
- Source: `sklearn.datasets.load_wine()`
- Total Features: 13
- Target: Wine class (0, 1, 2)

---

## Part 1: Feature Reduction using PCA

### 1. Original Feature Visualization
- Selected Features: `alcohol`, `malic_acid`
- Created a scatter plot with color-coded class labels.
- Saved to: `original_scatter.png`

### 2. PCA Transformation
- Applied PCA with all 13 features reduced to 2 components.
- Scatter plot with PC1 vs PC2.
- Saved to: `pca_scatter.png`

### 3. Explained Variance Ratio
- PC1: 36.2%  
- PC2: 19.2%  
→ Together they capture 55.4% of the total variance.

---

## Part 2-1: Data Reduction (Instance Reduction)

### Random Sampling (30%)
- Selected 30% of data randomly (without considering class distribution).
- PCA applied and visualized.
- Saved to: `random_sample_pca.png`

### Stratified Sampling (30%)
- Sampled 30% while preserving class balance.
- PCA applied and visualized.
- Saved to: `stratified_sample_pca.png`

### Analysis
- Random Sampling may distort class representation (imbalanced classes).
- Stratified Sampling better preserves original class separability and structure in PCA plot.
- Thus, stratified sampling is preferred for classification tasks or fair evaluation.

---

## Part 2-2: Data Reduction (Data Binning)

### Binning Strategy
- Feature: `alcohol`
- Compared two binning methods:
  - Equal-width Binning: Divides alcohol range into 3 equal intervals.
  - Equal-frequency Binning: Splits alcohol into 3 bins with equal number of samples.

### Visualization
- Binned `alcohol` vs `malic_acid` scatter plots
- Saved to: `equal_width_binning.png`  
- Saved to: `equal_freq_binning.png`

### Discussion
- Equal-width Binning:
  - Simple but may result in unbalanced bin sizes if the feature is skewed.
- Equal-frequency Binning:
  - Ensures even distribution across bins, but bin width varies.
- Insight:
  - Equal-frequency is useful for classification or models sensitive to imbalance.
  - Equal-width may miss important patterns if data is clustered.