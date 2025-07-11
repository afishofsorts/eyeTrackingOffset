import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scipy import stats
import dcor

##############################################
# TEST IF EACH REGIONAL COM IS NORMAL DIST   #
##############################################

def normalize(samples):
    return (samples- np.full(len(samples), np.average(samples)))/np.std(samples)

goldCOMS = np.load('saved\\gold_raw.npy')
silverCOMS = np.load('saved\\silver_raw.npy')

pvals = np.zeros((6, 2)); edists = np.zeros(6)
pvals[0, 0] = stats.kstest(normalize(goldCOMS[0, :]), stats.norm.cdf).pvalue
pvals[0, 1] = stats.kstest(normalize(goldCOMS[1, :]), stats.norm.cdf).pvalue
goldCOMS[0, :] = normalize(goldCOMS[0, :])
goldCOMS[1, :] = normalize(goldCOMS[1, :])
edists[0] = dcor.energy_distance(goldCOMS.T, stats.norm.rvs(size=goldCOMS.shape).T)
for i in range(5):
    pvals[i+1, 0] = stats.kstest(normalize(silverCOMS[i, 0, :]), stats.norm.cdf).pvalue
    pvals[i+1, 1] = stats.kstest(normalize(silverCOMS[i, 1, :]), stats.norm.cdf).pvalue
    silverCOMS[i, 0, :] = normalize(silverCOMS[i, 0, :])
    silverCOMS[i, 1, :] = normalize(silverCOMS[i, 1, :])
    edists[i+1] = dcor.energy_distance(silverCOMS[i, :, :].T, stats.norm.rvs(size=silverCOMS[i, :, :].shape).T)
    

print(pvals)
print(edists)