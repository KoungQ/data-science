# Regression & Classification Analysis Report

## Code Explanation

Below is a detailed description of each function and its main steps in the provided Python code.

---

### 1. `regression_analysis()` Function

1. **Data Loading & EDA**

   ```python
   diabetes = load_diabetes()
   X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
   y = diabetes.target
   ```

   * `X.describe()`: prints descriptive statistics for all features
   * Histograms and boxplots visualize feature distributions and outliers

2. **Simple Linear Regression (BMI → target)**

   ```python
   X_bmi = X[['bmi']].values
   slr = LinearRegression().fit(X_tr, y_tr)
   y_pred = slr.predict(X_te)
   ```

   * Split data 70/30 using BMI only
   * Compute R², MAE, RMSE
   * Plot scatter of actual vs predicted with regression line

3. **Multiple Linear Regression (all features)**

   ```python
   mlr = LinearRegression().fit(X_tr_all, y_tr_all)
   y_pred_all = mlr.predict(X_te_all)
   ```

   * Use all 10 features for training
   * Compute R², MAE, RMSE
   * Plot actual vs predicted values

4. **Polynomial Regression (BMI only, degree=2 and 3)**

   ```python
   for deg in (2,3):
       poly = PolynomialFeatures(degree=deg)
       pr = LinearRegression().fit(X_tr_p, y_tr_p)
   ```

   * Expand BMI feature to 2nd and 3rd degree
   * Compute metrics and plot fitted curves against actual data

5. **Multiple Polynomial Regression (BMI & BP, degree=2)**

   ```python
   X_2 = X[['bmi','bp']]
   mpr = LinearRegression().fit(X2_tr, y2_tr)
   ```

   * Include interaction between BMI and blood pressure
   * Compute metrics and plot actual vs predicted

---

### 2. `under_overfitting_analysis()` Function

* Generate synthetic sine-wave data with noise
* Train polynomial models of degree 1–15
* Compute train and test RMSE for each degree
* Plot RMSE curves to identify underfitting and overfitting regions

---

### 3. `logistic_classification()` Function

1. **Binary Logistic Regression**

   * Uses `make_classification` to generate 2-class data
   * Prints classification report (precision, recall, F1-score, accuracy)
   * Computes and plots ROC curve and AUC

2. **Multiclass Logistic Regression**

   * Uses `make_blobs` to generate 3-class data with OVR strategy
   * Prints classification report for each class
   * Computes and plots one-vs-all ROC curves and AUC values

---

### 4. `logistic_vs_polynomial()` Function

* Uses `make_moons` to generate nonlinearly separable data
* Trains base logistic and polynomial logistic (degree=3) models
* Compares AUC for both models
* Defines `plot_decision_boundary()` to visualize decision regions for each model

---

## Results Analysis

### A. Regression Model Performance

| Model                                 | R²     | MAE     | RMSE    |
| ------------------------------------- | ------ | ------- | ------- |
| Simple LR (BMI)                       | 0.2803 | 50.5931 | 62.3293 |
| Multiple LR (all features)            | 0.4773 | 41.9194 | 53.1202 |
| Poly LR (BMI, degree=2)               | 0.2766 | 50.6971 | 62.4923 |
| Poly LR (BMI, degree=3)               | 0.2770 | 50.6756 | 62.4737 |
| Multiple Poly LR (BMI & BP, degree=2) | 0.3354 | 48.5065 | 59.8971 |

* **Best Overall**: Multiple Linear Regression achieved the highest R² (0.4773) and lowest RMSE (53.12).
* **Polynomial Expansion** on BMI alone provided negligible improvement.
* **Interaction Terms** in multiple polynomial regression offered a modest performance boost.

### B. Underfitting vs Overfitting Analysis

* **Optimal Degree**: Degrees 5–7 yield the lowest test RMSE (\~0.39).
* **Underfitting**: Degrees 1–2 show high RMSE on both train and test sets.
* **Overfitting**: Degrees ≥10 show low train RMSE but sharply increasing test RMSE.

### C. Classification Model Evaluation

* **Binary Logistic Regression**:

  * Accuracy: 0.96
  * AUC: 0.984
  * Precision/Recall/F1: \~0.96 for both classes

* **Multiclass Logistic Regression (OVR)**:

  * One-vs-all AUC per class: 0.95–0.99

* **Logistic vs Polynomial Logistic on Moons**:

  * Base Logistic AUC: 0.9646
  * Polynomial Logistic AUC: 0.9884
  * Polynomial features improved AUC by \~2.4%.

---

## Conclusion

* **Feature Utilization**: Using multiple features and interaction terms significantly improves regression accuracy over single-feature models.
* **Model Complexity**: Polynomial degree must be selected carefully (e.g., via cross-validation) to balance bias and variance.
* **Classification**: Nonlinear feature expansion enhances logistic regression’s discriminative ability on complex datasets like `make_moons`.
