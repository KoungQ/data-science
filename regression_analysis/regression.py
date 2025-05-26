import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_classification, make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, roc_curve, auc
)

def regression_analysis():
    # 1. Load and EDA
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target

    # 1.1 Descriptive statistics
    print("=== EDA: Descriptive Statistics ===")
    print(X.describe(), "\n")

    # 1.1.1 Histograms
    X.hist(bins=30, figsize=(12, 10))
    plt.suptitle("Feature Histograms")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig('regression_analysis/feature_histograms.png')
    plt.show()

    # 1.1.2 Boxplots
    X.plot(kind='box', subplots=True, layout=(3,4), figsize=(12,10), sharex=False)
    plt.suptitle("Feature Boxplots")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig('regression_analysis/feature_boxplots.png')
    plt.show()

    # 1.2 Simple Linear Regression (bmi → target)
    X_bmi = X[['bmi']].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_bmi, y, train_size=0.7, random_state=42)
    slr = LinearRegression().fit(X_tr, y_tr)
    y_pred = slr.predict(X_te)
    print(">>> Simple Linear Regression <<<")
    print("R²:  ", r2_score(y_te, y_pred))
    print("MAE: ", mean_absolute_error(y_te, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_te, y_pred)), "\n")

    # 1.2.1 Visualization
    plt.figure(figsize=(8,6))
    plt.scatter(X_te, y_te, alpha=0.7, label='Actual')
    x_line = np.linspace(X_bmi.min(), X_bmi.max(), 100).reshape(-1,1)
    plt.plot(x_line, slr.predict(x_line), color='red', linewidth=2, label='Fit')
    plt.title("Simple Linear Regression: BMI → Target")
    plt.xlabel("BMI (standardized)")
    plt.ylabel("Disease Progression")
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_analysis/simple_linear_regression.png')
    plt.show()

    # 1.3 Multiple Linear Regression (all features)
    X_tr_all, X_te_all, y_tr_all, y_te_all = train_test_split(X, y, train_size=0.7, random_state=42)
    mlr = LinearRegression().fit(X_tr_all, y_tr_all)
    y_pred_all = mlr.predict(X_te_all)
    print(">>> Multiple Linear Regression <<<")
    print("R²:  ", r2_score(y_te_all, y_pred_all))
    print("MAE: ", mean_absolute_error(y_te_all, y_pred_all))
    print("RMSE:", np.sqrt(mean_squared_error(y_te_all, y_pred_all)), "\n")

    # 1.4 Polynomial Regression (bmi only, degree=2,3)
    for deg in (2,3):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_poly = poly.fit_transform(X_bmi)
        X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_poly, y, train_size=0.7, random_state=42)
        pr = LinearRegression().fit(X_tr_p, y_tr_p)
        y_pred_p = pr.predict(X_te_p)
        print(f">>> Polynomial Regression (deg={deg}) <<<")
        print("R²:  ", r2_score(y_te_p, y_pred_p))
        print("MAE: ", mean_absolute_error(y_te_p, y_pred_p))
        print("RMSE:", np.sqrt(mean_squared_error(y_te_p, y_pred_p)), "\n")

        # 1.4.1 Visualization
        plt.figure(figsize=(8,6))
        plt.scatter(X_bmi, y, alpha=0.3)
        xs = np.linspace(X_bmi.min(), X_bmi.max(), 200).reshape(-1,1)
        plt.plot(xs, pr.predict(poly.transform(xs)), label=f"deg={deg}")
        plt.title(f"Polynomial Regression (degree={deg})")
        plt.xlabel("BMI (standardized)")
        plt.ylabel("Disease Progression")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'regression_analysis/polynomial_deg{deg}.png')
        plt.show()

    # 1.5 Multiple Polynomial Regression (bmi & bp, degree=2)
    X_2 = X[['bmi','bp']]
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X2_poly = poly2.fit_transform(X_2)
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2_poly, y, train_size=0.7, random_state=42)
    mpr = LinearRegression().fit(X2_tr, y2_tr)
    y2_pred = mpr.predict(X2_te)
    print(">>> Multiple Polynomial Regression <<<")
    print("R²:  ", r2_score(y2_te, y2_pred))
    print("MAE: ", mean_absolute_error(y2_te, y2_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y2_te, y2_pred)), "\n")

def under_overfitting_analysis():
    # 2. Synthetic data for under/overfitting
    np.random.seed(42)
    X_syn = np.sort(5 * np.random.rand(100,1), axis=0)
    y_syn = np.sin(X_syn).ravel() + np.random.normal(0,0.5,100)
    X_tr, X_te, y_tr, y_te = train_test_split(X_syn, y_syn, train_size=0.7, random_state=42)

    train_err, test_err = [], []
    for deg in range(1,16):
        pf = PolynomialFeatures(degree=deg)
        X_tr_p = pf.fit_transform(X_tr)
        X_te_p = pf.transform(X_te)
        lr = LinearRegression().fit(X_tr_p, y_tr)
        train_err.append(np.sqrt(mean_squared_error(y_tr, lr.predict(X_tr_p))))
        test_err.append(np.sqrt(mean_squared_error(y_te, lr.predict(X_te_p))))

    print(">>> Under/Overfitting Analysis <<<")
    print("Train RMSE:", np.round(train_err,2))
    print("Test  RMSE:", np.round(test_err,2))

    # 2.1 Visualization
    plt.figure(figsize=(8,6))
    plt.plot(range(1,16), train_err, label='Train RMSE')
    plt.plot(range(1,16), test_err, label='Test RMSE')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE")
    plt.title("Underfitting vs Overfitting")
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_analysis/under_overfitting.png')
    plt.show()

def logistic_classification():
    # 3. Binary Logistic Regression
    X_bin, y_bin = make_classification(
        n_samples=300, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_bin, y_bin, train_size=0.7, random_state=42)
    log_bin = LogisticRegression().fit(Xb_tr, yb_tr)
    print(">>> Binary Classification Report <<<")
    print(classification_report(yb_te, log_bin.predict(Xb_te)))

    # ROC Curve
    fpr, tpr, _ = roc_curve(yb_te, log_bin.predict_proba(Xb_te)[:,1])
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary Logistic)")
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_analysis/roc_binary.png')
    plt.show()

def logistic_vs_polynomial():
    # 4. Logistic vs Polynomial Logistic (moons)
    X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=42)
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X_m, y_m, train_size=0.7, random_state=42)

    # Base logistic ROC
    log_base = LogisticRegression().fit(Xm_tr, ym_tr)
    fpr, tpr, _ = roc_curve(ym_te, log_base.predict_proba(Xm_te)[:,1])
    roc_auc = auc(fpr, tpr)
    print(">>> Base Logistic on Moons <<<")
    print("AUC:", roc_auc)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"Base AUC={roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Base Logistic)")
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_analysis/roc_base_moons.png')
    plt.show()

    # Polynomial logistic ROC
    pf3 = PolynomialFeatures(degree=3)
    X3_tr = pf3.fit_transform(Xm_tr)
    X3_te = pf3.transform(Xm_te)
    log_poly = LogisticRegression(max_iter=10000).fit(X3_tr, ym_tr)
    fpr2, tpr2, _ = roc_curve(ym_te, log_poly.predict_proba(X3_te)[:,1])
    roc_auc2 = auc(fpr2, tpr2)
    print(">>> Polynomial Logistic on Moons <<<")
    print("AUC:", roc_auc2)
    plt.figure(figsize=(6,6))
    plt.plot(fpr2, tpr2, label=f"Poly AUC={roc_auc2:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Polynomial Logistic)")
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_analysis/roc_poly_moons.png')
    plt.show()

if __name__ == "__main__":
    regression_analysis()
    under_overfitting_analysis()
    logistic_classification()
    logistic_vs_polynomial()