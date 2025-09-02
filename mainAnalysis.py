import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
import random as rand

##################################################################
# RANDOMLY SAMPLES 1 CORRECTED EYE TRACKING DATA SET AND PLOT IT #
##################################################################

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(cd) # grabs list of file names
isOutliers = np.load('saved\\corrData\\outlier_list.npy')

IDs = pandas.read_excel('saved\\Primary Care Calibration\\Primary Care Eye-Tracking Calibration.xlsx').to_numpy()[:, 0]
MLcal = pandas.read_excel('saved\\Primary Care Calibration\\Primary Care Eye-Tracking Calibration.xlsx').to_numpy()[:, 5]

randInd = rand.randint(0, len(isOutliers)-1)
dirInd = np.where(np.array(dir_list)==IDs[randInd]+'.csv')[0][0]

rootDir = 'saved\\corrData'

data = pandas.read_csv(rootDir + '\\data\\corr_' + str(dir_list[dirInd])).to_numpy()[:, 1:]
newData = cb.dataClean(data) # removes data  marked as invalid
psoPeaks = pandas.read_csv('saved\\corrData\\peaks\\peaks_' +  str(dir_list[dirInd])).to_numpy()[:, 1:]

cb.plotEyeData(newData[:, 4:], title='PSO Density Peaks for ' + str(dir_list[dirInd]), 
                genCOM=psoPeaks[0, :], psoPeaks=psoPeaks[1:, :], save=False)
if isOutliers[dirInd]=='True':
    plt.text(1000, 880, 'SPRD Miscalibrated: True', color='w')
else:
    plt.text(1000, 880, 'SPRD Miscalibrated: False', color='w')
if MLcal[randInd]=='Miscalibrated':
    plt.text(1000, 910, 'ML Miscalibrated: True', color='w')
else:
    plt.text(1000, 910, 'ML Miscalibrated: False', color='w')

plt.show()