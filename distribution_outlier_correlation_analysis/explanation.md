# Data Exploration Report

## Author: 202135508 김경규
## Assignment 3

---

## Part 1: Distribution Analysis (Histogram)

### 1. Age
- **Distribution Shape**: Generally resembles a normal distribution, symmetric around the mean.
- **Outliers**: No clear outliers observed.

### 2. Income
- **Distribution Shape**: Mostly centered around the average, but shows a **right-skewed** distribution with a long tail on the right.
- **Outliers**: One extreme high-income value (250,000) is present → considered an outlier.

### 3. Purchase Amount
- **Distribution Shape**: Typical **exponential distribution** pattern with many values concentrated in the lower range.
- **Outliers**: One very large value (5,000) is present → considered an outlier.

---

## Part 2: Outlier Detection (Boxplot)

### 1. Income
- **Outliers**: A clear outlier appears on the right side of the boxplot.
- **Explanation**: This value corresponds to the previously mentioned 250,000, which greatly exceeds the typical income level.

### 2. Purchase Amount
- **Outliers**: Several outliers are present, and the overall distribution is skewed.
- **Explanation**: There are values significantly higher than the typical purchase amount.

---

## Part 3: Variable Relationships (Scatterplot)

### 1. Age vs Purchase Amount
- **Relationship**: No clear correlation observed.
- **Observation**: A wide range of purchase amounts across all age groups. No visible clusters or trends.

### 2. Income vs Purchase Amount
- **Relationship**: Possibly a weak positive correlation, but not definitive.
- **Observation**: One high-income individual shows a very large purchase amount, considered an outlier.

---

## Summary Conclusion

- **Data distributions** show general normality or exponential trends, but some variables include extreme outliers.
- **Outliers** should be handled prior to data analysis or modeling.
- **Relationships between variables** are limited and not easily explained by simple linear regression.

---