import numpy as np
import matplotlib.pyplot as plt

wt = np.random.uniform(40.0, 90.0, 100)
ht = np.random.randint(140, 201, 100)
bmi = wt / ((ht / 100.0) ** 2)

categories = ['Underweight', 'Healthy', 'Overweight', 'Obese']
counts = [
    np.sum(bmi < 18.5),
    np.sum((bmi >= 18.5) & (bmi < 25.0)),
    np.sum((bmi >= 25.0) & (bmi < 30.0)),
    np.sum(bmi >= 30.0)
]

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.bar(categories, counts, color='skyblue')
plt.title("BMI Category (Bar Chart)")

plt.subplot(2, 2, 2)
plt.hist(bmi, bins=4, color='green', edgecolor='black')
plt.title("BMI Distribution (Histogram)")

plt.subplot(2, 2, 3)
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140)
plt.title("BMI Category (Pie Chart)")

plt.subplot(2, 2, 4)
plt.scatter(ht, wt, color='red')
plt.title("Height vs Weight (Scatter Plot)")

plt.tight_layout()
plt.show()
