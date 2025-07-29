import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
dir_list = os.listdir(cd)

isDupe = np.zeros(len(dir_list))
for i in range(len(dir_list)-1):
    dataLen1 = pandas.read_csv(cd + '\\' + str(dir_list[i])).to_numpy()[-1, 0]
    dataLen2 = pandas.read_csv(cd + '\\' + str(dir_list[i+1])).to_numpy()[-1, 0]
    if dataLen1 == dataLen2:
        isDupe[i] = 1

print(sum(isDupe))






        

