import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scipy import stats

##############################################
# TEST IF EACH REGIONAL COM IS NORMAL DIST   #
##############################################

def normalize(samples):
    return (samples- np.full(len(samples), np.average(samples)))/np.std(samples)

goldCOMS = np.load('saved\\gold_raw.npy')
silverCOMS = np.load('saved\\silver_raw.npy')

pvals = np.zeros((6, 2))
pvals[0, 0] = stats.kstest(normalize(goldCOMS[0, :]), stats.norm.cdf).pvalue
pvals[0, 1] = stats.kstest(normalize(goldCOMS[1, :]), stats.norm.cdf).pvalue
for i in range(5):
    pvals[i+1, 0] = stats.kstest(normalize(silverCOMS[i, 0, :]), stats.norm.cdf).pvalue
    pvals[i+1, 1] = stats.kstest(normalize(silverCOMS[i, 1, :]), stats.norm.cdf).pvalue

print(pvals)