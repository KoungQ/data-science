import numpy as np

a = np.array ( [ [ 2, 4, 6, 8],
                [ 1, 3, 5, 7] ] )

b = np.array ( [ [ 1, 2],
                    [ 2, 3],
                    [ 3, 4],
                    [ 4, 5] ] )
c = np.dot (a, b)

print(c)