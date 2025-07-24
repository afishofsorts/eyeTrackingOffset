from commandBank import commandBank as cb
import os
import pandas
import numpy as np

###############################################
# GENERIC SKELETON FOR CORRECTING OFFSET DATA #
###############################################

rootDir = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(rootDir)

isOutlier = np.empty(len(dir_list))
for i in range(len(dir_list)):
    data = pandas.read_csv(rootDir + '\\' + str(dir_list[i])).to_numpy()
    
    data[:, 4:], isOutlier[i] = cb.SPCorr(data)
    filename = str(dir_list[i])

    df = pandas.DataFrame(data)
    df.to_csv('saved\\corrData\\corr_' + filename)


np.save('saved\\corrData\\outlier_list', isOutlier)
    