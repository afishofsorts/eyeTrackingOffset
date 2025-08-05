import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)
# tags for each participant's data
calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()

# marks which data sets are Good with a 1
inds = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    if calList[i, 1]=='Good' or calList[i, 1]=='Offset':
        inds[i] = 1

isOutlier = np.load('tests\\rearviewNudge\\saved\\data\\is_outlier.npy')
psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy')

calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()
k=0
for i in range(len(dir_list)-2):
    if inds[i]==1:
        origData = cb.dataClean(pandas.read_csv(mvp+ '\\' + str(dir_list[i+1])).to_numpy())[:, 4:]
        corrData = pandas.read_csv('tests\\rearviewNudge\\saved\\data\\rvspCorr\\' + str(dir_list[i+1])).to_numpy()[:, 1:]
        corrVec = corrData[0, :] - origData[0, :]
        psoPeaks[:, :, k] = cb.translate(corrVec, psoPeaks[:, :, k])
        cb.plotEyeData(corrData, psoPeaks=psoPeaks[2:4, :, k], standards=True, filename='', save=False, 
                        title='PSO Density Peaks for ' + str(dir_list[i+1][:-4]))

<<<<<<< HEAD
        if isOutlier[k]:
=======
        if isOutlier[i] or calList[i, 1]=='Miscalibrated':
>>>>>>> a4386ae35a2157fa4012a6fc80abccfb23d57197
            plt.text(1250, 900, 'Marked as Miscalibrated: True', color='w')
        else:
            plt.text(1250, 900, 'Marked as Miscalibrated: False', color='w')

        plt.savefig('tests\\rearviewNudge\\saved\\figs\\rvspCorr\\rv_sp_corr_' + str(dir_list[i+1][:-4]))
        plt.close()

        cb.plotEyeData(corrData, filename='' + str(dir_list[i+1][:-4]), save=False)
<<<<<<< HEAD
        if isOutlier[k]:
=======
        if isOutlier[k] or calList[i, 1]=='Miscalibrated':
>>>>>>> a4386ae35a2157fa4012a6fc80abccfb23d57197
            plt.text(1250, 900, 'Marked as Miscalibrated: True', color='w')
        else:
            plt.text(1250, 900, 'Marked as Miscalibrated: False', color='w')
        
        plt.savefig('tests\\rearviewNudge\\saved\\figs\\sanityCheck\\corr_' + str(dir_list[i+1][:-4]))
        plt.close()
        k=k+1