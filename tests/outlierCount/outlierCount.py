import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

##################################################################################
# ESTIMATES THE AMOUNT OF DATA TO BE ERRONEOUSLY MARKED AS MISCALIBRATED OR GOOD #
##################################################################################

# grabbing directory names for each manually verified file
mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)
# tags for each participant's data
calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()

manualMiscal = np.array(['E034', 'E038', 'E059', 'E061', 'E064', 'E075', 'E042', 'E047', 'E088', 'E105', 'E109'])

# marks which data sets were makred Miscalibrated by ML algorithm with a 1
inds = np.zeros(len(dir_list)-1)
manualInds = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    if calList[i, 1]=='Miscalibrated':
        inds[i] = 1
    if any(dir_list[i+1][:-4] == manualMiscal):
        manualInds[i] = 1

isOutliers = np.load('tests\\rearviewNudge\\saved\\data\\is_outlier.npy') # boolean array for speedometer to road vector outliers

orOut = np.zeros(len(dir_list)-1)
k=0
for i in range(len(inds)):
    if inds[i] == 1:
        orOut[i]=1
    else:
        if isOutliers[k] == 1:
            orOut[i]=1
        k = k + 1

# -1 is false negative, 0 is true positive or true negative, and 1 is false positive
print(manualInds - orOut)

FPs = np.where(manualInds - orOut == 1)[0]
FNs = np.where(manualInds - orOut == -1)[0]
Ns = np.where(orOut == 1)[0]
Ps = np.where(orOut == 0)[0]
print('Percentage of False Positives: ' + str(len(FPs)/len(inds))) 
print('Percentage of False Negatives: ' + str(len(FNs)/len(inds))) 
print('Percentage of True Positives: ' + str((len(Ps)-len(FPs))/len(inds))) 
print('Percentage of True Negatives: ' + str((len(Ns)-len(FNs))/len(inds))) 
print('Percentage Marked as Miscal: ' + str(sum(orOut)/len(inds))) # amount of data to be marked as miscalibrated by either ML or vector outlier
print(1-sum(manualInds)/len(inds))


