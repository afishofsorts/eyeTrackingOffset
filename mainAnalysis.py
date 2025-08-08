import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

############################################################
# GENERATES PLOTS FOR MISCALIBRATED REGIONAL DENSITY PEAKS #
############################################################

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(cd) # grabs list of file names
isOutliers = np.load('saved\\corrData\\outlier_list.npy')

IDs = pandas.read_excel('saved\\Primary Care Calibration\\Primary Care Eye-Tracking Calibration.xlsx').to_numpy()[:, 0]
MLcal = pandas.read_excel('saved\\Primary Care Calibration\\Primary Care Eye-Tracking Calibration.xlsx').to_numpy()[:, 5]

print(IDs)
print(isOutliers)
for i in range(50):
    dirInd = np.where(np.array(dir_list)==IDs[i]+'.csv')[0]
    print(dirInd)
    if MLcal[i] == 'Miscalibrated' or isOutliers[dirInd]==1:
        print(i)
        # DATA IS CURRENTLY SHRUNK, WILL HAVE INCORRECT MISCALIBRATION MARKINGS
        # data = np.load()
        # cb.plotEyeData()