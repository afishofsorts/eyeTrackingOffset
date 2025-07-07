from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

r = stats.norm.rvs(size=1000)
r2 = stats.norm.rvs(size=1000)

errDist = np.sqrt(r**2+r2**2)
pval = stats.kstest(errDist, stats.rayleigh.cdf).pvalue
print(pval)
plt.hist(errDist)
plt.show()