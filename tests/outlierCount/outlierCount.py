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

manualMiscal = np.array(['E038', 'E048', 'E059', 'E061', 'E062', 'E064', 'E068', 'E075', 'E088', 'E104', 'E109', 'E114'])

# marks which data sets were makred Miscalibrated by ML algorithm with a 1
inds = np.zeros(len(dir_list)-1)
manualInds = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    if calList[i, 1]=='Miscalibrated':
        inds[i] = 1
    if any(dir_list[i+1][:-4] == manualMiscal):
        manualInds[i] = 1

isOutliers = np.load('tests\\outlierCount\\saved\\peak_outliers.npy') # boolean array for speedometer to road vector outliers

andOut = np.zeros(len(dir_list)-1)
orOut = np.zeros(len(dir_list)-1)
for i in range(len(inds)):
    if isOutliers[i] == 1 and inds[i] == 1:
        andOut[i] = 1
    if isOutliers[i] == 1 or inds[i] == 1:
        orOut[i] = 1

# -1 is false negative, 0 is true positive or true negative, and 1 is false positive
print(manualInds - orOut) 
print(manualInds - isOutliers)
print(manualInds - inds)

# checking if outlier check changes much about existing ML check
print(orOut-andOut)

print(1/len(inds)) # percentage of false positives and false negatives since only recorded 1 for both
print(sum(orOut)/len(inds)) # amount of data to be marked as miscalibrated by either ML or vector outlier

# so around 1.5 percent of good data will be tossed and 1.5 percent bad will make it in



