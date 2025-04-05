import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200

age = np.random.normal(loc=35, scale=10, size=n)
income = np.random.normal(loc=50000, scale=15000, size=n)
purchase_amount = np.random.exponential(scale=300, size=n)
purchase_count = np.random.poisson(lam=3, size=n)

income[5] = 250000
purchase_amount[10] = 5000

def plot_histogram(data, title, xlabel):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {title}')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_hist.png')
    plt.show()

plot_histogram(age, 'Age', 'Age')
plot_histogram(income, 'Income', 'Income')
plot_histogram(purchase_amount, 'Purchase Amount', 'Purchase Amount')

def plot_boxplot(data, title, ylabel):
    plt.figure(figsize=(8, 2))
    plt.boxplot(data, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightgreen'))
    plt.title(f'Boxplot of {title}')
    plt.xlabel(ylabel)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_boxplot.png')
    plt.show()

plot_boxplot(income, 'Income', 'Income')
plot_boxplot(purchase_amount, 'Purchase Amount', 'Purchase Amount')

def plot_scatter(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='tomato', alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

plot_scatter(age, purchase_amount, 'Age', 'Purchase Amount', 'Age vs Purchase Amount')
plot_scatter(income, purchase_amount, 'Income', 'Purchase Amount', 'Income vs Purchase Amount')
