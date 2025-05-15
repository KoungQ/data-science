# Clean It, Fix It, Scale It - Assignment

This assignment implements data preprocessing and feature scaling using a synthetic dataset.  
The tasks include missing value handling, outlier detection and treatment, application of various scaling methods, and visualizations.

---

## 1. Dataset Setup

Synthetic data was generated using NumPy:

- `age`: Normally distributed around 40
- `income`: Normally distributed around 60,000
- `purchases`: Exponentially distributed (skewed)
- `clicks`: Poisson-distributed counts

Additionally:
- Injected missing values at `income[5]` and `purchases[10]`
- Injected outliers at `income[7] = 300000`, `purchases[3] = 5000`

---

## 2. Tasks Completed

- [x] Missing value handling:
  - Replaced missing `income` with mean
  - Replaced missing `purchases` with median
- [x] Outlier detection using IQR method
- [x] Outlier treatment via clipping
- [x] Feature scaling methods:
  - Min-Max Scaling (`age`)
  - Z-score Standardization (`income`)
  - Log Transformation (`purchases`)
  - Robust Scaling (`income`)
  - Vector Normalization (`age`, `income`, `clicks`)
- [x] Visualization:
  - Histogram of `purchases` before and after log transformation
  - Boxplot of `income` before and after robust scaling

---

## 3. Files Included

- `clean_it_fix_it_scale_it.py` : Full Python implementation
- `clean_fix_scale/purchases_histogram.png` : Purchases log transform histogram
- `clean_fix_scale/income_boxplot.png` : Income robust scaling boxplot

---

## 4. How to Run

Install dependencies if needed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Then run the Python file:

```bash
python clean_it_fix_it_scale_it.py
```

---

## 5. Output Preview

Sample of transformed values (first 5 rows):

| MinMax Age | Z-Score Income | Log Purchases | Robust Income | Norm Age | Norm Income | Norm Clicks |
|------------|----------------|----------------|----------------|-----------|--------------|--------------|
| ...        | ...            | ...            | ...            | ...       | ...          | ...          |

(See terminal output for actual values.)

---

## 6. Scaling Technique Explanations

### 1. Min-Max Scaling (Applied to `age`)
- **What it does**: Rescales values to [0, 1].
- **Why appropriate**: `age` has a normal range without extreme outliers. Min-Max preserves distribution shape while bounding values, suitable for neural nets or k-NN.

### 2. Z-score Standardization (Applied to `income`)
- **What it does**: Centers to mean 0 and scales to unit variance.
- **Why appropriate**: Standardization handles different scales well. Even with moderate outliers, Z-score is common for algorithms assuming normality (e.g., SVM).

### 3. Log Transformation (Applied to `purchases`)
- **What it does**: Compresses large values and reduces right-skew.
- **Why appropriate**: `purchases` are exponentially distributed with a long tail. Log transform normalizes distribution and reduces skew.

### 4. Robust Scaling (Applied to `income`)
- **What it does**: Uses median and IQR instead of mean/std.
- **Why appropriate**: Income has extreme outliers (e.g., 300,000). RobustScaler minimizes their influence, unlike Min-Max or Z-score.

### 5. Vector Normalization (Applied to `[age, income, clicks]`)
- **What it does**: Scales each row so its L2 norm equals 1.
- **Why appropriate**: Useful when comparing input vectors by direction rather than magnitude (e.g., cosine similarity). Ensures fair sample comparison.

---

## 7. Author

Department of Artificial Intelligence  
Gachon University – Spring 2025  
Data Science Lecture 03 – Assignment
