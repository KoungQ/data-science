import numpy as np

dataset = np.array ( [ [ 2, 4, 6, 8, 3, 2, 5] ,
[ 7, 5, 3, 1, 6, 8, 0] ,
[ 1, 3, 2, 1, 0, 0, 8] ] )

print(np.max (dataset, axis=1) - np.min(dataset, axis=1))