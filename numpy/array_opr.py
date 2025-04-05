import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c_add = a + b
c_sub = a - b
c_mul = a * b
c_div = a / b

c = [c_add, c_sub, c_mul, c_div]

print(c)