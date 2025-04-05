import numpy as np

a = np.ones((3, 3),float)
b = np.zeros((3, 3),float)
b = b + 2. * np.identity(3)
c = a + b

print(c)