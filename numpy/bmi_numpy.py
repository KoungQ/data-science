import numpy as np

wt = np.random.uniform(40.0, 90.0, 100)
ht = np.random.randint(140, 201, 100)
bmi = wt / ((ht / 100.0) ** 2)

print(bmi)
print(bmi[:10])