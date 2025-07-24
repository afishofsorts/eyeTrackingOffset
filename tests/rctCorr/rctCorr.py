import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

###############################################################################
# CORRECTS AND PLOTS RCT EYE TRACKING DATA USING SPEEDOMETER DENSITY PEAK     #
###############################################################################

# grabbing directory names for each manually verified file
allSeries = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(allSeries)

rctListSK = pandas.read_excel('saved\\Primary Care Calibration\\List of RCT IDs.xlsx').to_numpy()[:, 2]
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have.xlsx').to_numpy()[:, 0]

isOutlier = np.empty(len(rctListTS))
for i in range(len(rctListTS)):
    ind = dir_list.index(rctListTS[i] + '.csv')
    data = pandas.read_csv(allSeries + '\\' + str(dir_list[ind])).to_numpy()
    
    data[:, 4:], isOutlier[i] = cb.SPCorr(data)
    filename = str(dir_list[ind])
    print(filename)
    print(i)

    df = pandas.DataFrame(data)
    df.to_csv('tests\\rctCorr\\saved\\data\\' + filename)


np.save('tests\\rctCorr\\saved\\data\\outlier_list', isOutlier)






