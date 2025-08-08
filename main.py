from commandBank import commandBank as cb
import os
import pandas
import numpy as np

###############################################
# GENERIC SKELETON FOR CORRECTING OFFSET DATA #
###############################################

# file directory for offset data
rootDir = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(rootDir) # grabs list of file names

isOutlier = np.empty(len(dir_list))
for i in range(len(dir_list)):
    data = pandas.read_csv(rootDir + '\\' + str(dir_list[i])).to_numpy()
    
    data[:, 4:], isOutlier[i], peaks = cb.SPRVCorr(data) # corrects offset data using rearview and speedometer density peaks

    df = pandas.DataFrame(data)
    df.to_csv('saved\\corrData\\data\\corr_' + str(dir_list[i]))

    df = pandas.DataFrame(peaks)
    df.to_csv('saved\\corrData\\peaks\\peaks_' + str(dir_list[i]))

np.save('saved\\corrData\\outlier_list', isOutlier)