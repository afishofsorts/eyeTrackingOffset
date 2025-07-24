import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

##############################################
# SIMULATION TO ESTIMATE ERROR RADIUS        #
##############################################

covar = np.load('saved\\pso_SPRD_covar.npy') # density peak covariance matrix

norm1 = stats.norm.rvs(size=10000)*np.sqrt(2/3*covar[0, 0]) # seperates out the x and y standard deviations
norm2 = stats.norm.rvs(size=10000)*np.sqrt(2/3*covar[1, 1])

final = np.sqrt(norm1**2 + norm2**2) # radius for each pair of x and y samples

bins = np.arange(0, 100, 1)
counts, bins, bars = plt.hist(final, bins)
plt.title('Monte Carlso Simulation of Error Radius for Speedometer Density Peak')
plt.xlabel('Radius (pixels)'); plt.ylabel('Frequency')
plt.savefig('tests\\monteCarlo\\saved\\radius_hist')
total = 0
i = 0
# checks at what point 99 percent of histogram is below i
while(total < sum(counts)*0.95):
    total = total + counts[i]
    i = i + 1

np.save('tests\\monteCarlo\\saved\\model_dist.npy', final)
print(i)