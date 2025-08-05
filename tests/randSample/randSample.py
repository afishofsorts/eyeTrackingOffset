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

# grabbing directory names for each manually verified file
cd = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(cd)


for i in range(100):
    k = rand.randrange(0, len(dir_list))
    data = pandas.read_csv(cd+ '\\' + str(dir_list[k])).to_numpy()[:, :-2]
    validData = cb.dataClean(data)

    corrData, isOutlier = cb.SPRVCorr(validData)
    cb.plotEyeData(corrData, save=False)
    if isOutlier:
        plt.text(1250, 900, 'Marked as Miscalibrated: True', color='w')
    else:
        plt.text(1250, 900, 'Marked as Miscalibrated: False', color='w')
    
    plt.savefig('tests\\randSample\\saved\\' + str(dir_list[k][:-4]))
    plt.close()