import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

##############################################
# GENERATES PLOTS FOR REGIONAL DENSITY PEAKS #
##############################################

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

savedDir = 'tests\\regPso\\saved'
psoPeaks = np.load(savedDir + '\\pso_peaks.npy')

k=0 # indexing for non-miscalibrated files
for i in range(len(dir_list)-1):
    if inds[i]==1: # only performs correction for data marked as Good
        data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
        newData = cb.dataClean(data) # removes data  marked as invalid

        cb.plotEyeData(newData[:, 4:], 'pso_peaks_' + str(dir_list[i+1][:-4]), title='PSO Density Peaks for ' + str(dir_list[i+1]),
                       dir=savedDir + '\\figs', genCOM=psoPeaks[0, :, k], psoPeaks=psoPeaks[1:, :, k])
        k=k+1
