import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have.xlsx').to_numpy()[:, 0]

dataLen = np.zeros(len(rctListTS)-1)
for i in range(len(dataLen)):
    dataLen[i] = len(pandas.read_csv(cd + '\\' + str(rctListTS[i]) + '.csv').to_numpy()[:, 0])

isDupe = np.zeros(len(dataLen))
for i in range(len(dataLen)):
    simLen = np.where(dataLen == dataLen[i])[0]
    print(simLen)
    data1 = pandas.read_csv(cd + '\\' + str(rctListTS[simLen[0]]) + '.csv').to_numpy()
    for j in range(len(simLen)-1):
        data2 = pandas.read_csv(cd + '\\' + str(rctListTS[simLen[j+1]]) + '.csv').to_numpy()  
        print(data1[:, 4:]-data2[:, 4:])
        if sum(data1[:, 4] - data2[:, 4])==0:
            isDupe[i] = 1

print(isDupe)






        

