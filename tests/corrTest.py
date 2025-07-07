import numpy as np
import matplotlib.pyplot as plt 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from commandBank import commandBank as cb

#######################################################
# STRIPPED DOWN SANITY TEST FOR CORRECTION ALGORITHM  #
#######################################################

avgComs = np.array([[1, 1], [3, 2], [1.5, 0]])
pts = np.array([[1.5, 1.5], [3.1, 2.1], [1.6, 0.1]])

difs = cb.relVectors(pts[0, :], pts[1:, :])
vars = np.array([[1,1],[10,10],[10,10]])
corrX1 = cb.meanComCorr(avgComs, difs, vars)
print(corrX1)
corr = pts + np.ones(len(pts[:, 0]))[None].T @ (corrX1 - pts[0, :])[None]
print(corr)
plt.plot(avgComs[:, 0], avgComs[:, 1], 'o')
plt.plot(pts[:, 0], pts[:, 1], 'o')
plt.plot(corr[:, 0], corr[:, 1], 'o')
plt.legend(['Standards', 'Measured', 'Corrected'])
plt.show()