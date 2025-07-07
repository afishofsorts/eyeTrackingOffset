##############################################
# TESTS THE HISTOGRAM OF ERROR DISTANCES     #
##############################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

errDist = np.load('tests\\errDistTest\\saved\\err_data.npy')

plt.hist(errDist)
plt.title('Error Radius Estimation')
plt.ylabel('Frequency'); plt.xlabel('Distance from Original Manually Validated Good Data to its Offset Correction (pixels)')
plt.show()
pval = stats.kstest(errDist, stats.rayleigh.cdf).pvalue

print(pval)
