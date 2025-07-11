##############################################
# TESTS THE HISTOGRAM OF ERROR DISTANCES     #
##############################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

errDist = np.load('tests\\errDistTest\\saved\\err_data.npy')
assumed = np.load('tests\\monteCarlo\\saved\\model_dist.npy')


pval = stats.kstest(errDist, assumed).pvalue

plt.title(['Probability Density of Correction Offset Error Radius'])
plt.text(15, 0.07, 'K-Test P-value: ' + str(pval))
plt.hist([errDist, assumed], density='True')
plt.legend(['Manually Validated Good Data', 'Model'])
plt.ylabel('Frequency'); plt.xlabel('Distance from Original Data to Offset Correction (pixels)')
plt.savefig('tests\\errDistTest\\saved\\err_hist')
