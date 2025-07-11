import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
import dcor

coms = np.load('tests\\clusterTests\\saved\\pso_coms.npy')
manualOutliers = np.array([9, 21, 23, 25, 31, 58])

difs = np.zeros((4+3+2+1, 2, len(coms[0, 0, :])-6))
k = 0
for i in range(len(coms[0, 0, :])):
    if not i in manualOutliers:
        for j in range(4):
            difs[int(-j**2/2+9/2*j):int(-(j+1)**2/2+9/2*(j+1)), :, k] = cb.relVectors(coms[(j+1), :, i], coms[(j+2):, :, i])[1:, :]
        k = k + 1

edists = np.zeros(4+3+2+1)
for i in range(4+3+2+1):
    covar = np.cov(difs[i, :, :])
    print(covar[0, 1]/np.sqrt(covar[0, 0]*covar[1, 1]))
    mean = np.array([np.mean(difs[i, 0, :]), np.mean(difs[i, 1, :])])
    edists[i] = dcor.energy_distance(difs[i, :, :].T, np.random.multivariate_normal(mean, covar, size=len(difs[0, 0, :])))
    plt.plot(difs[i, 0, :], difs[i, 1, :], 'o', markersize=2)
print(edists)
plt.legend(['SM SP', 'SM RV', 'SM RD', 'SM PN', 'SP RV', 'SP RD', 'SP PN', 'RV RD', 'RV PN', 'RD PN'])
plt.title('Difference Vectors for Each Regional PSO COM')
plt.savefig('tests\\clusterTests\\saved\\dif_spreads')





