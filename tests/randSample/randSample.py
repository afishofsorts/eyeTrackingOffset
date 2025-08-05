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

    cb.plotEyeData(validData[:, 4:], save=False)
    
    plt.savefig('tests\\randSample\\saved\\uncorr\\' + str(dir_list[k][:-4]))
    plt.close()