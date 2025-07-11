import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

goldStats = np.load('saved\\gold_stats.npy')
silverStats = np.load('saved\\silver_stats.npy')
vars = np.append([goldStats[1]], silverStats[1], axis=0)

outDists = np.zeros(6)
for k in range(6):
    norm1 = stats.norm.rvs(size=100000)*np.sqrt(vars[k, 0])
    norm2 = stats.norm.rvs(size=100000)*np.sqrt(vars[k, 1])

    final = np.sqrt(norm1**2 + norm2**2)

    bins = np.arange(0, 150, 1)
    counts, bins, bars = plt.hist(final, bins)

    total = 0
    i = 0
    while(total < sum(counts)*0.99):
        total = total + counts[i]
        i = i + 1
    outDists[k] = i

np.save('saved\\outlier_dists.npy', outDists)