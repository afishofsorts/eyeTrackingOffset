import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

###############################################################################
# CORRECTS AND PLOTS SAMPLE EYE TRACKING DATA USING SPEEDOMETER DENSITY PEAK  #
###############################################################################

# grabbing directory names for each manually verified file
mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)
# tags for each participant's data
calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()

# marks which data sets are Good with a 1
inds = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    if calList[i, 1]=='Good' or calList[i, 1]=='Offset':
        inds[i] = 1

for i in range(len(dir_list)-1):
    data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
    
    corrCoords, isOutlier = cb.SPCorr(data)
    plt.rcParams["figure.figsize"] = [10.00, 5.0]
    img = plt.imread("saved\\Primary Care Calibration\\Calibration Shape.jpg")
    fig, ax = plt.subplots()
    im = ax.imshow(img, extent=[0, 1920, 0, 1080])
    plt.plot(corrCoords[:, 0], corrCoords[:, 1], 'o', markersize=1)
    plt.plot(920, 312, 'o', markersize=2)
    plt.title('PSO Speedometer Comparison for ' + str(dir_list[i+1][:-4]))
    plt.xlabel('x position (pixels)'); plt.ylabel('y position (pixels)')
    plt.text(1350, 1000, 'isOutlier = ' + str(isOutlier), color='w')
    plt.xlim(0, 1920); plt.ylim(0, 1080)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    # Put a legend to the right of the current axis
    ax.legend(['PSO SP', 'REF'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('tests\\mvpCorr\\saved\\pso_corr_' + str(dir_list[i+1][:-4]))
    plt.close()



