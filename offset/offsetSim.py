import numpy as np
import matplotlib.pyplot as plt


def genData(N, m):
    focusPts = np.random.rand(2, m)*7
    musig = np.random.rand(2, m) + 0.2
    dists = np.zeros(shape=(2, N*m))
    for i in range(m):
        dists[0, (i*N):((i+1)*N)] = np.random.normal(musig[0, i], musig[1, i], size=(1, N)) + np.full(N, focusPts[:, i][0])
        dists[1, (i*N):((i+1)*N)] = np.random.normal(musig[0, i], musig[0, i], size=(1, N)) + np.full(N, focusPts[:, i][1])
    return dists

m = 5
N = 300

dists = genData(N, m)

plt.plot(dists[0, :], dists[1, :], 'o')
plt.title('Original Simulated Data')
plt.savefig('offset\\pics\\orig_data.png')
plt.close()

yoff = (np.random.rand() - 0.5)*2
xoff = (np.random.rand() - 0.5)*2
# dispMatr = np.transpose(np.matmul(np.full((N*m, 2), offset), np.array([[0, 1], [0, 0]])))
    
plt.title('Original and Offset Simulated Data')
plt.plot(dists[0, :] + np.full(N*m, xoff), dists[1, :] + np.full(N*m, yoff), 'o')
plt.plot(dists[0, :], dists[1, :], 'o')
plt.savefig('offset\\pics\\off_data.png')


