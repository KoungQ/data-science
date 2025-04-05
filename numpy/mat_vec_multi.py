import numpy as np

a = np.array ( [ 2, 4, 6, 8] )
b = np.array ( [ [ 1, 2, 3, 4],
                    [ 2, 3, 4, 5],
                    [ 3, 4, 5, 6],
                    [ 4, 5, 6, 7] ] )
c = np.dot (a, b)

print (c)