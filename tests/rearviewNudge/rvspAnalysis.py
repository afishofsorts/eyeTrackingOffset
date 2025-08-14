import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

###################################################################
# INVESTIGATES DISTANCES BETWEEN PEAKS FOR SHRINKING CHECK        #
###################################################################

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

# booloan array for which data sets have outlier sprd vectors, but only for the ones that made it past ML algo
isOutlier = np.load('tests\\rearviewNudge\\saved\\data\\is_outlier.npy') 
psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy')

rvNumber = np.zeros(int(sum(inds)))
RVSPdifs = np.zeros((int(sum(inds)), 2))
RVRDdifs = np.zeros((int(sum(inds)), 2))
k=0 
for i in range(len(dir_list)-1):
    if inds[i]==1:
        data = pandas.read_csv(mvp+ '\\' + str(dir_list[i+1])).to_numpy()
        validData = cb.dataClean(data)

        rvNumber[k] = cb.nCheck(psoPeaks[3, :, k], validData[:, 4:], 50)
        RVSPdifs[k, :] = psoPeaks[3, :, k] - psoPeaks[2, :, k]
        RVRDdifs[k, :] = psoPeaks[3, :, k] - psoPeaks[4, :, k]

        k = k + 1

calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()
k=0
for i in range(len(dir_list)-1):
    if inds[i]==1:
        origData = cb.dataClean(pandas.read_csv(mvp+ '\\' + str(dir_list[i+1])).to_numpy())
        corrData = pandas.read_csv('tests\\rearviewNudge\\saved\\data\\rvspCorr\\' + str(dir_list[i+1])).to_numpy()[:, 1:]
        corrVec = corrData[0, :] - origData[0, 4:]
        psoPeaks[:, :, k] = cb.translate(corrVec, psoPeaks[:, :, k])
        cb.plotEyeData(corrData, psoPeaks=psoPeaks[2:4, :, k], standards=True, filename='', save=False, 
                        title='PSO Density Peaks for ' + str(dir_list[i+1][:-4]))

        if isOutlier[k] or calList[i, 1]=='Miscalibrated':
            plt.text(1000, 880, 'Marked as Miscalibrated: True', color='w')
        else:
            plt.text(1000, 880, 'Marked as Miscalibrated: False', color='w')
        if RVSPdifs[k, 0] < 550:
            plt.text(1000, 970, 'RVSP Short x', color='w')
        if RVSPdifs[k, 0] > 550 and RVSPdifs[k, 1]<275:
            plt.text(1000, 940, 'RVSP Short y Long x', color='w')
        rvData = cb.dataClean(origData, cb.rv)
        if cb.rvCheck(psoPeaks[4, :, k], psoPeaks[3, :, k], psoPeaks[2, :, k], rvData[:, 4:]):
            plt.text(1000, 910, 'RVSP Correction Applicable', color='w')
        plt.text(1000, 850, 'Number of rv points < 25 px: ' + str(cb.nCheck(psoPeaks[3, :, k], rvData[:, 4:], 50)), color='w')

        plt.savefig('tests\\rearviewNudge\\saved\\figs\\rvspCorr\\rv_sp_corr_' + str(dir_list[i+1][:-4]))
        plt.close()

        cb.plotEyeData(corrData, filename='' + str(dir_list[i+1][:-4]), save=False)
        if isOutlier[k] or calList[i, 1]=='Miscalibrated':
            plt.text(1250, 900, 'Marked as Miscalibrated: True', color='w')
        else:
            plt.text(1250, 900, 'Marked as Miscalibrated: False', color='w')
        
        plt.savefig('tests\\rearviewNudge\\saved\\figs\\sanityCheck\\corr_' + str(dir_list[i+1][:-4]))
        plt.close()
        k=k+1