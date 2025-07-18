from commandBank import commandBank as cb
import os
import pandas
import numpy as np

###############################################
# GENERIC SKELETON FOR CORRECTING OFFSET DATA #
###############################################

rootDir = ''
dir_list = os.listdir(dir)

isOutlier = np.empty(len(dir_list)-1)
for i in range(len(dir_list)-1):
    data = pandas.read_csv(rootDir + '\\' + str(dir_list[i+1])).to_numpy()
    
    corrData, isOutlier[i] = cb.SPCorr(data)
    filename = ''
    np.save('saved\\corrData\\corr_' + filename, corrData)

np.save('saved\\corrData\\outlier_list', isOutlier)
    