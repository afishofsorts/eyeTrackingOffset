import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

goldVar = np.load('saved\\gold_stats.npy')[1]
silverVars = np.load('saved\\silver_stats.npy')[1]

stndx = np.sqrt(1/(1/goldVar[0] + sum(1/silverVars[:, 0])))
stndy = np.sqrt(1/(1/goldVar[1] + sum(1/silverVars[:, 1])))

norm1 = stats.norm.rvs(size=10000)*stndx
norm2 = stats.norm.rvs(size=10000)*stndy

final = np.sqrt(norm1**2 + norm2**2)

bins = np.arange(0, 35, 1)
counts, bins, bars = plt.hist(final, bins)

total = 0
i = 0
while(total < sum(counts)*0.95):
    total = total + counts[i]
    i = i + 1

np.save('tests\\monteCarlo\\saved\\model_dist.npy', final)
print(i)