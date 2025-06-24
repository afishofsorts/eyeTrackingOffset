import numpy as np
import matplotlib.pyplot as plt

def genData(N, m, focusPts):
    musig = np.random.rand(2, m) + 0.2
    dists = np.zeros(shape=(2, N*m))
    for i in range(m):
        dists[0, (i*N):((i+1)*N)] = np.random.normal(musig[0, i], musig[1, i], size=(1, N)) + np.full(N, focusPts[:, i][0])
        dists[1, (i*N):((i+1)*N)] = np.random.normal(musig[0, i], musig[0, i], size=(1, N)) + np.full(N, focusPts[:, i][1])
    return dists

m = 5
N = 300
trials = 50
fnum = 1

focusPts = np.random.rand(2, m)*7
distSets = np.zeros(shape=(2, N*m, trials, fnum))
for j in range(fnum):
    for i in range(trials):
        distSets[:, :, i, j] = genData(N, m, focusPts)

np.save('offset\\data\\trueDistSets.npy', distSets)

for j in range(fnum):
    for i in range(trials):
        xoff = (np.random.rand() - 0.5)*2
        yoff = (np.random.rand() - 0.5)*2
        distSets[0, :, i, j] = distSets[0, :, i, j] + np.full(N*m, xoff)
        distSets[1, :, i, j] = distSets[1, :, i, j] + np.full(N*m, yoff)

np.save('offset\\data\\offDistSets.npy', distSets)