import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_classification, make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, label_binarize
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

    # 1.3.1 Visualization
    plt.figure(figsize=(6,6))
    plt.scatter(y_te_all, y_pred_all, alpha=0.7)
    plt.plot([y_te_all.min(), y_te_all.max()], [y_te_all.min(), y_te_all.max()], 'k--')
    plt.title("Multiple Linear Regression: Actual vs Predicted")
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.grid(True)
    plt.savefig('regression_analysis/multiple_lr_actual_vs_pred.png')
    plt.show()

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

    plt.figure(figsize=(6,6))
    plt.scatter(y2_te, y2_pred, alpha=0.7)
    plt.plot([y2_te.min(), y2_te.max()], [y2_te.min(), y2_te.max()], 'k--')
    plt.title("Multiple Polynomial Regression: Actual vs Predicted")
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.grid(True)
    plt.savefig('regression_analysis/multiple_poly_actual_vs_pred.png')
    plt.show()


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

    fpr, tpr, _ = roc_curve(yb_te, log_bin.predict_proba(Xb_te)[:,1])
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
    plt.title("ROC Curve (Binary Logistic)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_analysis/roc_binary.png')
    plt.show()

    # 3. Multiclass Logistic Regression
    X_mb, y_mb = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    Xmb_tr, Xmb_te, ymb_tr, ymb_te = train_test_split(X_mb, y_mb, train_size=0.7, random_state=42)
    log_multi = LogisticRegression(multi_class='ovr', max_iter=1000).fit(Xmb_tr, ymb_tr)
    print(">>> Multiclass Classification Report <<<")
    print(classification_report(ymb_te, log_multi.predict(Xmb_te)))

    ymb_te_bin = label_binarize(ymb_te, classes=[0,1,2])
    probas = log_multi.predict_proba(Xmb_te)
    plt.figure(figsize=(6,6))
    for i in range(3):
        fpr_i, tpr_i, _ = roc_curve(ymb_te_bin[:,i], probas[:,i])
        plt.plot(fpr_i, tpr_i, label=f"Class {i} (AUC={auc(fpr_i,tpr_i):.2f})")
    plt.title("Multiclass ROC Curve (OVA)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_analysis/roc_multiclass.png')
    plt.show()


def logistic_vs_polynomial():
    # 4. Logistic vs Polynomial Logistic (moons)
    X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=42)
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X_m, y_m, train_size=0.7, random_state=42)
    base_log = LogisticRegression().fit(Xm_tr, ym_tr)
    pf3 = PolynomialFeatures(degree=3)
    X3_tr, X3_te = pf3.fit_transform(Xm_tr), pf3.transform(Xm_te)
    poly_log = LogisticRegression(max_iter=10000).fit(X3_tr, ym_tr)

    def plot_decision_boundary(model, transf, X, y, fname, title):
        x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
        y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        if transf:
            grid = transf.transform(grid)
        Z = model.predict(grid).reshape(xx.shape)
        plt.figure(figsize=(6,6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', alpha=0.7)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.savefig(fname)
        plt.show()

    plot_decision_boundary(base_log, None, X_m, y_m,
                           'regression_analysis/db_base.png',
                           "Decision Boundary: Base Logistic")
    plot_decision_boundary(poly_log, pf3, X_m, y_m,
                           'regression_analysis/db_poly.png',
                           "Decision Boundary: Polynomial Logistic (deg=3)")


def regression_summary():
    # 6. Model comparison table without extra dependencies
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target
    results = []

    # Simple LR
    X_bmi = X[['bmi']]
    X_tr, X_te, y_tr, y_te = train_test_split(X_bmi, y, train_size=0.7, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    results.append({
        'Model': 'Simple LR (bmi)',
        'R²': r2_score(y_te, y_pred),
        'MAE': mean_absolute_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred))
    })

    # Multiple LR
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.7, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    results.append({
        'Model': 'Multiple LR (all features)',
        'R²': r2_score(y_te, y_pred),
        'MAE': mean_absolute_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred))
    })

    # Poly deg=2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_bmi)
    X_tr, X_te, y_tr, y_te = train_test_split(X_poly, y, train_size=0.7, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    results.append({
        'Model': 'Poly LR (bmi, deg=2)',
        'R²': r2_score(y_te, y_pred),
        'MAE': mean_absolute_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred))
    })

    # Poly deg=3
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X_bmi)
    X_tr, X_te, y_tr, y_te = train_test_split(X_poly, y, train_size=0.7, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    results.append({
        'Model': 'Poly LR (bmi, deg=3)',
        'R²': r2_score(y_te, y_pred),
        'MAE': mean_absolute_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred))
    })

    # Multiple Poly
    X_2 = X[['bmi','bp']]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_2)
    X_tr, X_te, y_tr, y_te = train_test_split(X_poly, y, train_size=0.7, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    results.append({
        'Model': 'Multiple Poly LR (bmi & bp, deg=2)',
        'R²': r2_score(y_te, y_pred),
        'MAE': mean_absolute_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred))
    })

    df = pd.DataFrame(results)
    print("=== Regression Model Comparison ===")
    print(df.to_string(index=False, float_format='%.4f'))


if __name__ == "__main__":
    regression_analysis()
    under_overfitting_analysis()
    logistic_classification()
    logistic_vs_polynomial()
    regression_summary()
