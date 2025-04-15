# Homework 3 - BMI Data Analysis (Essential Steps)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Create folder for required figures-hw
os.makedirs("scikit-learn/figures-hw", exist_ok=True)

# 1. Load the dataset
print("[1] Load Dataset")
df = pd.read_excel("scikit-learn/bmi_data_phw3.xlsx")

# 2. Basic exploration
print(df.info())
print(df.describe())
print(df.dtypes)

# 3. Convert BMI to string for categorical plots
df["BMI"] = df["BMI"].astype(str)

# 4. Plot Height histograms by BMI group
g1 = sns.FacetGrid(df, col="BMI", col_wrap=3, height=4)
g1.map_dataframe(sns.histplot, x="Height (Inches)", bins=10)
g1.fig.suptitle("Height Distribution by BMI", y=1.05)
plt.tight_layout()
plt.savefig("scikit-learn/figures-hw/height_hist_by_bmi.png")
plt.clf()

# 5. Plot Weight histograms by BMI group
g2 = sns.FacetGrid(df, col="BMI", col_wrap=3, height=4)
g2.map_dataframe(sns.histplot, x="Weight (Pounds)", bins=10)
g2.fig.suptitle("Weight Distribution by BMI", y=1.05)
plt.tight_layout()
plt.savefig("scikit-learn/figures-hw/weight_hist_by_bmi.png")
plt.clf()

# 6. Scaling comparison
scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler()
}

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(df[["Height (Inches)", "Weight (Pounds)"]])
    scaled_df = pd.DataFrame(scaled, columns=["Height", "Weight"])

    sns.histplot(scaled_df["Height"], bins=10, kde=True).set_title(f"{name.capitalize()} Scaler - Height")
    plt.tight_layout()
    plt.savefig(f"scikit-learn/figures-hw/{name}_height_scaled.png")
    plt.clf()

    sns.histplot(scaled_df["Weight"], bins=10, kde=True).set_title(f"{name.capitalize()} Scaler - Weight")
    plt.tight_layout()
    plt.savefig(f"scikit-learn/figures-hw/{name}_weight_scaled.png")
    plt.clf()

# 7. Regression and ze calculation (no gender split)
X = df[["Height (Inches)"]]
y = df["Weight (Pounds)"]
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
e = y - y_pred
ze = (e - np.mean(e)) / np.std(e)

# 8. ze histogram
sns.histplot(ze, bins=10, kde=True)
plt.title("Normalized Error (ze) Distribution")
plt.xlabel("ze")
plt.ylabel("Frequency")
plt.axvline(x=1.0, color='red', linestyle='--', label='alpha = +1.0')
plt.axvline(x=-1.0, color='red', linestyle='--', label='alpha = -1.0')
plt.legend()
plt.tight_layout()
plt.savefig("scikit-learn/figures-hw/ze_histogram.png")
plt.clf()

# 9. Predict BMI based on ze
alpha = 1.0
bmi_pred = [0 if z < -alpha else 4 if z > alpha else None for z in ze]

# 10. Scatter plot with regression line and outliers
normal = df[[p is None for p in bmi_pred]]
outliers = df[[p is not None for p in bmi_pred]]

plt.figure(figsize=(8, 6))
plt.scatter(normal["Height (Inches)"], normal["Weight (Pounds)"], color='blue', label="Normal")
plt.scatter(outliers["Height (Inches)"], outliers["Weight (Pounds)"], color='red', label="Outlier (BMI=0 or 4)")

x_vals = np.linspace(df["Height (Inches)"].min(), df["Height (Inches)"].max(), 100)
y_vals = reg.predict(x_vals.reshape(-1, 1))
plt.plot(x_vals, y_vals, color='black', label="Linear Regression")

plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.title("Outlier Detection via Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scikit-learn/figures-hw/scatter_outlier_regression.png")
plt.clf()

# 11. Gender-based regression
print("\n[Gender-based Regression]")
df_female = df[df["Sex"] == "Female"]
df_male = df[df["Sex"] == "Male"]

reg_f = LinearRegression().fit(df_female[["Height (Inches)"]], df_female["Weight (Pounds)"])
reg_m = LinearRegression().fit(df_male[["Height (Inches)"]], df_male["Weight (Pounds)"])

print(f"Female model: weight = {reg_f.coef_[0]:.2f} * height + {reg_f.intercept_:.2f}")
print(f"Male model:   weight = {reg_m.coef_[0]:.2f} * height + {reg_m.intercept_:.2f}")
