import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import pandas as pd

# Step 1: Data Setup
np.random.seed(42)
n = 150
age = np.random.normal(40, 10, n)
income = np.random.normal(60000, 15000, n)
purchases = np.random.exponential(300, n)
clicks = np.random.poisson(5, n)

# Inject missing values
income[5] = np.nan
purchases[10] = np.nan

# Inject outliers
income[7] = 300000
purchases[3] = 5000

# Step 2: Handle missing values
income[np.isnan(income)] = np.nanmean(income)
purchases[np.isnan(purchases)] = np.nanmedian(purchases)

# Step 3: Handle outliers using IQR clipping
def iqr_clip(arr):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return np.clip(arr, lower, upper)

income_clipped = iqr_clip(income)
purchases_clipped = iqr_clip(purchases)

# Step 4: Feature Scaling
# a. Min-Max Scaling for age
age_minmax = MinMaxScaler().fit_transform(age.reshape(-1, 1)).flatten()

# b. Z-score Standardization for income
income_zscore = StandardScaler().fit_transform(income_clipped.reshape(-1, 1)).flatten()

# c. Log Transformation for purchases
purchases_log = np.log1p(purchases_clipped)

# d. Robust Scaling for income
income_robust = RobustScaler().fit_transform(income_clipped.reshape(-1, 1)).flatten()

# e. Vector Normalization for [age, income, clicks]
vector_data = np.vstack([age, income_clipped, clicks]).T
normalized_vector = Normalizer().fit_transform(vector_data)

# Step 5: Visualization
# Histogram of purchases before and after log transform
plt.figure(figsize=(8, 4))
plt.hist(purchases_clipped, bins=30, alpha=0.5, label='Original Purchases')
plt.hist(purchases_log, bins=30, alpha=0.5, label='Log Transformed Purchases')
plt.legend()
plt.title("Histogram of Purchases: Original vs Log Transformed")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("clean_fix_scale/purchases_histogram.png")

# Box plot of income before and after robust scaling
plt.figure(figsize=(6, 4))
plt.boxplot([income_clipped, income_robust], labels=["Original Income", "Robust Scaled Income"])
plt.title("Boxplot of Income: Original vs Robust Scaled")
plt.tight_layout()
plt.savefig("clean_fix_scale/income_boxplot.png")

# Optional: Print summary of transformed results (first 5 rows)
df = pd.DataFrame({
    "MinMax Age": age_minmax[:5],
    "Z-Score Income": income_zscore[:5],
    "Log Purchases": purchases_log[:5],
    "Robust Income": income_robust[:5],
    "Norm Age": normalized_vector[:5, 0],
    "Norm Income": normalized_vector[:5, 1],
    "Norm Clicks": normalized_vector[:5, 2],
})
print(df)
