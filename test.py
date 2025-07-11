import numpy as np
import dcor
import matplotlib.pyplot as plt

covar = np.array([[1, 3/5], [3/5, 2]])
mean = np.array([1, 1])
a = np.random.multivariate_normal(mean, covar, size=1000)
b = np.random.multivariate_normal(mean, covar, size=1000)

print(dcor.energy_distance(a, b))

plt.plot(a[:, 0], a[:, 1], 'o')
plt.plot(b[:, 0], b[:, 1], 'o')
plt.show()