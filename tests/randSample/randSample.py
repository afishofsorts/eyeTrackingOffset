import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
from commandBank import commandBank as cb
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import random as rand

#############################################################################
# SAMPLES AND CORRECTS RANDOM DATA SETS USING SPEEDOMETER AND REARVIEW      #
#############################################################################

# grabbing directory names for each manually verified file
cd = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(cd)

isOutlier = np.empty(30)
for i in range(2):
    k = rand.randrange(0, len(dir_list))
    data = pandas.read_csv(cd+ '\\' + str(dir_list[k])).to_numpy()
    validData = cb.dataClean(data)

    cb.plotEyeData(validData[:, 4:], save=False)
    
    plt.savefig('tests\\randSample\\saved\\figs\\uncorr\\' + str(dir_list[k][:-4]))
    plt.close()

    data[:, 4:], isOutlier[i], peaks = cb.SPRVCorr(data)

    df = pandas.DataFrame(data)
    df.to_csv('tests\\randSample\\saved\\data\\corr\\corr' + str(dir_list[i]))

    df = pandas.DataFrame(peaks)
    df.to_csv('tests\\randSample\\saved\\data\\peaks\\peaks' + str(dir_list[i]))