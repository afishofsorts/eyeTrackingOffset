import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
import dcor

################################################################
# CALCULATES DIFFERENCE VECTORS BETWEEN REGIONAL DENSITY PEAKS #
################################################################

psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy') # regional density peaks
manualOutliers = np.array([9, 21, 23, 25, 31, 58]) # list of VDAs identified as miscalibrated despite associated offset or good tag

print(psoPeaks)
difs = np.zeros((4+3+2+1, 2, len(psoPeaks[0, 0, :])-6))
k = 0
# only checks one direction of difference vectors (if SP to RD is calculated, RD to SP is a duplicate)
for i in range(len(psoPeaks[0, 0, :])):
    if not i in manualOutliers:
        for j in range(4):
            difs[int(-j**2/2+9/2*j):int(-(j+1)**2/2+9/2*(j+1)), :, k] = cb.relVectors(psoPeaks[(j+1), :, i], psoPeaks[(j+2):, :, i])[1:, :]
        k = k + 1

edists = np.zeros(4+3+2+1)
for i in range(4+3+2+1):
    covar = np.cov(difs[i, :, :])
    print(i)
    print(np.sqrt(covar[0, 0]))
    print(np.sqrt(covar[1, 1]))
    print(covar[0, 1]/np.sqrt(covar[0, 0]*covar[1, 1]))
    mean = np.array([np.mean(difs[i, 0, :]), np.mean(difs[i, 1, :])])
    # compares difference vector distribution to normal with same mean and covariance using energy distances
    edists[i] = dcor.energy_distance(difs[i, :, :].T, np.random.multivariate_normal(mean, covar, size=len(difs[0, 0, :]))) 
    plt.plot(difs[i, 0, :], difs[i, 1, :], 'o', markersize=2)
print(edists)

np.save('saved\\pso_SPRD_covar', np.cov(difs[5, :, :]))
np.save('saved\\pso_SPRD_mean', np.array([np.mean(difs[5, 0, :]), np.mean(difs[5, 1, :])]))
plt.legend(['SM SP', 'SM RV', 'SM RD', 'SM PN', 'SP RV', 'SP RD', 'SP PN', 'RV RD', 'RV PN', 'RD PN'])
plt.xlabel('x (pixels)'); plt.ylabel('y (pixels)')
plt.title('Difference Vectors for Each Regional PSO COM')
plt.savefig('tests\\difVectors\\saved\\dif_spreads')
plt.close()

plt.plot(difs[5, 0, :], difs[5, 1, :], 'o', color='#8c564b')
plt.title('Speedometer to Road Density Peak Vector')
plt.xlabel('x (pixels)'); plt.ylabel('y (pixels)')
plt.savefig('tests\\difVectors\\saved\\dif_spread_SPRD')
plt.close()

plt.plot(difs[4, 0, :], difs[4, 1, :], 'o', color='tab:purple')
plt.title('Speedometer to Rearview Peak Vector')
plt.xlabel('x (pixels)'); plt.ylabel('y (pixels)')
plt.savefig('tests\\difVectors\\saved\\dif_spread_SPRV')
plt.close()

plt.plot(difs[7, 0, :], difs[7, 1, :], 'o', color='tab:gray')
plt.title('Road to Rearview Density Peak Vector')
plt.xlabel('x (pixels)'); plt.ylabel('y (pixels)')
plt.savefig('tests\\difVectors\\saved\\dif_spread_RDRV')





