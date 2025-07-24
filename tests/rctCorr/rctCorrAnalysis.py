import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

cd = 'tests\\rctCorr'
dir_list = os.listdir(cd + '\\saved\\data')

isOutlier = np.load(cd + '\\saved\\data\\outlier_list.npy')

print(len(isOutlier))
print(dir_list)
for i in range(len(isOutlier)):
    data = pandas.read_csv(cd + '\\saved\\data\\' + str(dir_list[i])).to_numpy()[:, 1:]
    newData = cb.dataClean(data) # removes data  marked as invalid

    plt.rcParams["figure.figsize"] = [20.00, 10.0]
    img = plt.imread("saved\\Primary Care Calibration\\Exemplar Scene Cropped.jpg")
    fig, ax = plt.subplots()
    im = ax.imshow(img, extent=[0, 1920, 0, 1080])
    plt.plot(newData[:, 4], newData[:, 5], 'o', mec = 'black', mew='0.8', markersize=4, color='fuchsia')
    plt.plot(920, 312, 'o', mec='black', color='gold', markersize=13)
    plt.text(1350, 1000, 'isOutlier = ' + str(isOutlier[i]), color='w', size=20)
    plt.text(1350, 800, 'isOutlier = ' + str(i), color='w', size=20)
    plt.xlim(0, 1920); plt.ylim(0, 1080)
    
    plt.title('Outlier ' + str(dir_list[i][:-4]))
    plt.xlabel('x position (pixels)'); plt.ylabel('y position (pixels)')


    # Put a legend to the right of the current axis
    plt.savefig(cd + '\\saved\\figs\\' + str(dir_list[i][:-4]), bbox_inches='tight', pad_inches=0.5)
    plt.close()
        

