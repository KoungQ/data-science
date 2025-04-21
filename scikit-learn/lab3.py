# Lab 3 - Enhanced with Metadata Print, Gender-Based Regression, and Outlier Adjustment

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Create output folder for figures
os.makedirs("scikit-learn/figures-lab", exist_ok=True)

# 1. Load dataset
print("[1] Load Dataset")
df = pd.read_csv("scikit-learn/bmi_data_lab3.csv")
print(df.info())
print(df.describe())
print(df.dtypes)

# 2. Clean outliers and missing values
print("[2] Clean Outliers and Missing Values")
df.loc[(df["Height (Inches)"] < 50) | (df["Height (Inches)"] > 84), "Height (Inches)"] = np.nan
df.loc[(df["Weight (Pounds)"] < 50) | (df["Weight (Pounds)"] > 300), "Weight (Pounds)"] = np.nan
df.loc[(df["BMI"] < 0) | (df["BMI"] > 4), "BMI"] = np.nan
df_clean = df.dropna().copy()
df_clean["BMI"] = df_clean["BMI"].astype(int)

# 3. Plot BMI-wise height histogram
print("[3] Plot Height Histogram by BMI")
g = sns.FacetGrid(df_clean, col="BMI", col_wrap=3)
g.map_dataframe(sns.histplot, x="Height (Inches)", bins=10)
plt.tight_layout()
plt.savefig("scikit-learn/figures-lab/height_hist_by_bmi.png")
plt.clf()

# 4. Plot BMI-wise weight histogram
print("[4] Plot Weight Histogram by BMI")
g = sns.FacetGrid(df_clean, col="BMI", col_wrap=3)
g.map_dataframe(sns.histplot, x="Weight (Pounds)", bins=10)
plt.tight_layout()
plt.savefig("scikit-learn/figures-lab/weight_hist_by_bmi.png")
plt.clf()

# 5. Linear regression and ze calculation
print("[5] Perform Linear Regression and Calculate ze")
X = df_clean[["Height (Inches)"]]
y = df_clean["Weight (Pounds)"]
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
e = y - y_pred
ze = (e - np.mean(e)) / np.std(e)

slope = reg.coef_[0]
intercept = reg.intercept_

# 6. ze histogram
print("[6] Plot ze Histogram")
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
print("[7] Predict Outlier BMI Labels")
alpha = 1.0
bmi_pred = [0 if z < -alpha else 4 if z > alpha else None for z in ze]
df_clean["bmi_pred"] = bmi_pred

# 8. Adjust Outliers to Fit Regression Line
print("[8] Adjust Outliers to Fit Regression Line")
normal = df_clean[df_clean["bmi_pred"].isna()].copy()
outliers = df_clean[df_clean["bmi_pred"].notna()].copy()
outliers["Weight (Pounds)"] = slope * outliers["Height (Inches)"] + intercept

adjusted_df = pd.concat([normal, outliers])

# 회귀선 준비
x_vals = np.linspace(adjusted_df["Height (Inches)"].min(), adjusted_df["Height (Inches)"].max(), 100)
x_vals_df = pd.DataFrame(x_vals, columns=["Height (Inches)"])
y_vals = reg.predict(x_vals_df)

# 보정 후 산점도 시각화
plt.figure(figsize=(8, 6))
plt.scatter(normal["Height (Inches)"], normal["Weight (Pounds)"], color='blue', label="Normal (Unchanged)")
plt.scatter(outliers["Height (Inches)"], outliers["Weight (Pounds)"], color='red', label="Outlier (Adjusted)")
plt.plot(x_vals, y_vals, color='black', label="Linear Regression")

plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.title("Outlier Adjusted to Regression Line")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scikit-learn/figures-lab/scatter_outlier_regression_adjusted.png")
plt.clf()

# 9. Scaling and plot results
print("[9] Plot Scaled Height vs Weight using Standard, MinMax, and Robust Scalers")
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

for name, scaler in scalers.items():
    scaled_data = scaler.fit_transform(adjusted_df[["Height (Inches)", "Weight (Pounds)"]])
    plt.figure(figsize=(6, 4))
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], alpha=0.5)
    plt.title(f"{name} - Scaled Height vs. Weight")
    plt.xlabel("Scaled Height")
    plt.ylabel("Scaled Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scikit-learn/figures-lab/{name.lower()}_height_weight.png")
    plt.clf()

# 10. Gender-based regression
print("[10] Gender-based Regression")
for gender in ["Female", "Male"]:
    df_gender = df_clean[df_clean["Sex"] == gender]
    if df_gender.empty:
        print(f"{gender} model: no data")
        continue
    Xg = df_gender[["Height (Inches)"]]
    yg = df_gender["Weight (Pounds)"]
    model = LinearRegression().fit(Xg, yg)
    slope_g = model.coef_[0]
    intercept_g = model.intercept_
    print(f"{gender} model: weight = {slope_g:.2f} * height + {intercept_g:.2f}")
