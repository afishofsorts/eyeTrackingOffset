import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

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

manualMiscal = np.array(['E034', 'E038', 'E059', 'E061', 'E064', 'E075', 'E042', 'E047', 'E088', 'E105', 'E109'])

psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy')

n = int(sum(inds)-len(manualMiscal))
RVSPdifs = np.zeros((n, 2))
RVRDdifs = np.zeros((n, 2))
SPRDdifs = np.zeros((n, 2))
# calculated difference vectors
k=0 
for i in range(len(dir_list)-1):
    misInds = np.where(manualMiscal == dir_list[i+1][:-4])[0]
    if inds[i]==1 and len(misInds)==0:
        data = pandas.read_csv(mvp+ '\\' + str(dir_list[i+1])).to_numpy()
        validData = cb.dataClean(data)
        RVSPdifs[k, :] = psoPeaks[3, :, k] - psoPeaks[2, :, k]
        RVRDdifs[k, :] = psoPeaks[3, :, k] - psoPeaks[4, :, k]
        SPRDdifs[k, :] = psoPeaks[4, :, k] - psoPeaks[2, :, k]

        k = k + 1

# saves covariances
np.save('saved\\pso_RVSP_covar', np.cov(RVSPdifs.T))
np.save('saved\\pso_RVRD_covar', np.cov(RVRDdifs.T))
np.save('saved\\pso_SPRD_covar', np.cov(SPRDdifs.T))

print(np.sqrt(np.diag(np.cov(RVSPdifs.T))))
print(np.sqrt(np.diag(np.cov(RVRDdifs.T))))
print(np.sqrt(2/3*np.diag(np.cov(SPRDdifs.T))))
