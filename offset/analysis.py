import numpy as np
import matplotlib.pyplot as plt

trueDistSets = np.load('offset\\data\\trueDistSets.npy')
offDistSets = np.load('offset\\data\\offDistSets.npy')

for i in range(4):
    plt.plot(trueDistSets[0, :, i, 0], trueDistSets[1, :, i, 0], 'o')
plt.title('4 Original Data Trials For 1 Set of 5 Focus Points')
plt.savefig('offset\\pics\\orig_trials.png')