import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
from commandBank import commandBank as cb
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt

###############################################################################################
# GENERATES REARVIEW/SPEEDOMETER DENSITY PEAKS AND WHETHER REARVIEW CORRECTION WILL BE USED   #
###############################################################################################

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

psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy')[3, :, :]

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

savedDir = 'tests\\rearviewNudge\\saved'
np.save(savedDir + '\\data\\rv_number', rvNumber)
np.save(savedDir + '\\data\\RVSP_difs', RVSPdifs)
np.save(savedDir + '\\data\\RVRD_difs', RVRDdifs)

plt.title('Speedometer to Rearview Vector')
plt.ylabel('y (pixels)'); plt.xlabel('x (pixels)')
plt.plot(RVSPdifs[:, 0], RVSPdifs[:, 1], 'o')
plt.savefig(savedDir + 'figs\\RVSP_difs.png')
plt.close()

plt.title('Road to Rearview Vector')
plt.ylabel('y (pixels)'); plt.xlabel('x (pixels)')
plt.plot(RVRDdifs[:, 0], RVRDdifs[:, 1], 'o')
plt.savefig(savedDir + '\\figs\\RVRD_difs')
