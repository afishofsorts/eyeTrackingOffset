import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

###################################################################################
# GENERATES PLOTS FOR REGIONAL DENSITY PEAKS AND MISCALIBRATION OF RANDOM SAMPLES #
###################################################################################

# grabbing directory names for each manually verified file
rootDir = 'tests\\randSample\\saved\\data\\corr'
dir_list = os.listdir(rootDir)

savedDir = 'tests\\randSample\\saved'

isOutlier = np.load('tests\\randSample\\saved\\data\\isOutlier.npy')
calList = pandas.read_excel('saved\\Primary Care Calibration\\Primary Care Eye-Tracking Calibration.xlsx').to_numpy()

for i in range(len(dir_list)):
    data = pandas.read_csv(rootDir + '\\' + str(dir_list[i])).to_numpy()[:, 1:]
    newData = cb.dataClean(data) # removes data  marked as invalid
    psoPeaks = pandas.read_csv(savedDir + '\\data\\peaks\\peaks' +  str(dir_list[i][4:-4]) + '.csv').to_numpy()[:, 1:]

    normOutInd = np.where(isOutlier[:, 0]==dir_list[i][4:-4])[0]
    MLOutInd = np.where(calList[:, 0]==dir_list[i][4:-4])[0]

    cb.plotEyeData(newData[:, 4:], title='PSO Density Peaks for ' + str(dir_list[i]), 
                   genCOM=psoPeaks[0, :], psoPeaks=psoPeaks[1:, :], save=False)
    if isOutlier[normOutInd, 1]=='True':
        plt.text(1000, 880, 'SPRD Miscalibrated: True', color='w')
    else:
        plt.text(1000, 880, 'SPRD Miscalibrated: False', color='w')
    if calList[MLOutInd, 5]=='Miscalibrated':
        plt.text(1000, 910, 'ML Miscalibrated: True', color='w')
    else:
        plt.text(1000, 910, 'ML Miscalibrated: False', color='w')

    plt.savefig(savedDir + '\\figs\\corr\\pso_peaks_' + str(dir_list[i][4:-4]))
    plt.close()