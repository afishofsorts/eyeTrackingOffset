import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

##############################################
#  #
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

savedDir = 'tests\\rearviewNudge\\saved'
psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy')
rvNumber = np.load(savedDir + '\\data\\rv_number.npy')
RVSPdifs = np.load(savedDir + '\\data\\RVSP_difs.npy')
RVRDdifs = np.load(savedDir + '\\data\\RVSP_difs.npy')

k=0 # indexing for errDist
for i in range(len(dir_list)-1):
    if inds[i]==1: # only performs correction for data marked as Good
        data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
        newData = cb.dataClean(data) # removes data  marked as invalid

        cb.plotEyeData(newData[:, 4:], genCOM=psoPeaks[0, :, k], psoPeaks=psoPeaks[2:4, :, k], save=False, 
                       title='PSO Density Peaks for ' + str(dir_list[i+1][:-4]))

        plt.text(1250, 1000, rvNumber[k], color='w')
        if RVSPdifs[k, 0] < 500:
            plt.text(1250, 970, 'RVSP Short x', color='w')
        if RVSPdifs[k, 0] > 500 and RVSPdifs[k, 1]<275:
            plt.text(1250, 940, 'RVSP Short y Long x', color='w')
        rvData = cb.dataClean(newData, cb.rv)
        if cb.rvCheck(psoPeaks[4, :, k], psoPeaks[3, :, k], psoPeaks[2, :, k], rvData[:, 4:]):
            plt.text(1250, 910, 'RVSP Correction Applicable', color='w')

        plt.savefig('tests\\rearviewNudge\\saved\\figs\\rvStats\\pso_peaks_' + str(dir_list[i+1][:-4]), bbox_inches='tight', pad_inches=0.5)
        plt.close()
        k=k+1
