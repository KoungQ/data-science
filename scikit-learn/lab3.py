# Lab 3 - Essential Tasks Only

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Create output folder for figures
os.makedirs("scikit-learn/figures-lab", exist_ok=True)

# 1. Load dataset
print("[1] Load Dataset")
df = pd.read_csv("scikit-learn/bmi_data_lab3.csv")

# 2. Clean outliers and missing values
df.loc[(df["Height (Inches)"] < 50) | (df["Height (Inches)"] > 84), "Height (Inches)"] = np.nan
df.loc[(df["Weight (Pounds)"] < 50) | (df["Weight (Pounds)"] > 300), "Weight (Pounds)"] = np.nan
df.loc[(df["BMI"] < 0) | (df["BMI"] > 4), "BMI"] = np.nan
df_clean = df.dropna().copy()
df_clean["BMI"] = df_clean["BMI"].astype(int)

# 3. Plot BMI-wise height histogram
g = sns.FacetGrid(df_clean, col="BMI", col_wrap=3)
g.map_dataframe(sns.histplot, x="Height (Inches)", bins=10)
plt.tight_layout()
plt.savefig("scikit-learn/figures-lab/height_hist_by_bmi.png")
plt.clf()

# 4. Plot BMI-wise weight histogram
g = sns.FacetGrid(df_clean, col="BMI", col_wrap=3)
g.map_dataframe(sns.histplot, x="Weight (Pounds)", bins=10)
plt.tight_layout()
plt.savefig("scikit-learn/figures-lab/weight_hist_by_bmi.png")
plt.clf()

# 5. Linear regression and ze calculation
X = df_clean[["Height (Inches)"]]
y = df_clean["Weight (Pounds)"]
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
e = y - y_pred
ze = (e - np.mean(e)) / np.std(e)

# 6. ze histogram
sns.histplot(ze, bins=10, kde=True)
plt.title("Normalized Error (ze) Distribution")
plt.xlabel("ze")
plt.ylabel("Frequency")
plt.axvline(x=1.0, color='red', linestyle='--', label='alpha = +1.0')
plt.axvline(x=-1.0, color='red', linestyle='--', label='alpha = -1.0')
plt.legend()
plt.tight_layout()
plt.savefig("scikit-learn/figures-lab/ze_histogram.png")
plt.clf()

# 7. Predict BMI based on ze
alpha = 1.0
bmi_pred = [0 if z < -alpha else 4 if z > alpha else None for z in ze]

# 8. Scatter plot with regression and outliers
normal = df_clean[[p is None for p in bmi_pred]]
outliers = df_clean[[p is not None for p in bmi_pred]]

plt.figure(figsize=(8, 6))
plt.scatter(normal["Height (Inches)"], normal["Weight (Pounds)"], color='blue', label="Normal")
plt.scatter(outliers["Height (Inches)"], outliers["Weight (Pounds)"], color='red', label="Outlier (BMI=0 or 4)")

x_vals = np.linspace(df_clean["Height (Inches)"].min(), df_clean["Height (Inches)"].max(), 100)
y_vals = reg.predict(x_vals.reshape(-1, 1))
plt.plot(x_vals, y_vals, color='black', label="Linear Regression")

plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.title("Outlier Detection via Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scikit-learn/figures-lab/scatter_outlier_regression.png")
plt.clf()