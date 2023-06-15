import numpy as np

a = np.random.randn(3, 2)
b = np.random.randn(2, 1)

z = np.matmul(a, b)

relu = lambda x : max(0, x)




