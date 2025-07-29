import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
import dcor

###############################################################
#  MEASURES THE COVARIANCE OF ELIGIBLE REARVIEW DENSITY PEAKS #
###############################################################

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

psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy')
RVSPdifs = np.load('tests\\rearviewNudge\\saved\\data\\RVSP_difs.npy')
RVRDdifs = np.load('tests\\rearviewNudge\\saved\\data\\RVRD_difs.npy')

k=0
l = 0
for i in range(len(dir_list)-1):
    if inds[i]==1: # only performs correction for data marked as Good
        data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
        newData = cb.dataClean(data) # removes data  marked as invalid
        if not cb.rvCheck(psoPeaks[4, :, k], psoPeaks[3, :, k], psoPeaks[2, :, k], newData[:, 4:]):
            RVSPdifs = np.delete(RVSPdifs, k-l, 0)
            RVRDdifs = np.delete(RVRDdifs, k-l, 0)
            l = l + 1
        k = k + 1


RVSPmean = np.array([np.mean(RVSPdifs[:, 0]), np.mean(RVSPdifs[:, 1])])

norm = np.random.multivariate_normal(RVSPmean, np.cov(RVSPdifs.T), size=len(RVSPdifs[:, 0]))
edist = dcor.energy_distance(RVRDdifs, 
                             norm) 

print(edist) # very high for this subset of RVRD data compared to entire RVRD data

np.save('saved\\pso_RVSP_covar', np.cov(RVSPdifs.T))
np.save('saved\\pso_RVRD_covar', np.cov(RVRDdifs.T))
