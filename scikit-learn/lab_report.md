# Lab 3: Regression-Based Outlier Detection using BMI Dataset

## 1. Dataset Description

We use a dataset containing the following columns:

- `Height (Inches)` — individual's height
- `Weight (Pounds)` — individual's weight
- `BMI` — body mass index category (integer between 0–4)

---

## 2. Data Cleaning

To remove invalid or extreme values:

- Heights below 50 or above 84 inches → set as NaN  
- Weights below 50 or above 300 pounds → set as NaN  
- BMI values not in the range [0, 4] → set as NaN  

After removing rows with missing values, the cleaned dataset is used for analysis.

---

## 3. Height Histogram by BMI

Each BMI category shows a distribution of heights.

**Saved as**: `height_hist_by_bmi.png`

<img src="figures-lab/height_hist_by_bmi.png" width="500"/>

---

## 4. Weight Histogram by BMI

Each BMI category shows a distribution of weights.

**Saved as**: `weight_hist_by_bmi.png`

<img src="figures-lab/weight_hist_by_bmi.png" width="500"/>

---

## 5. Linear Regression & Normalized Error (ze)

We fit a linear regression model:

