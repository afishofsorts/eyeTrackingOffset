import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

cd = 'tests\\rctCorr'
dir_list = os.listdir(cd + '\\saved\\data')

dataLen = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    data = pandas.read_csv(cd + '\\saved\\data\\' + str(dir_list[i])).to_numpy()
    dataLen[i] = data[-1, 0]

manualShrink = np.array([0, 1, 2, 3, 4, 9, 13, 16, 17, 18 , 26, 29, 40, 41, 43, 72, 85, 97])
dupeCol = ['N/A']*(len(dir_list)-1)
isShrunk = np.zeros(len(dir_list)-1)
k = 0
for i in range(len(dir_list)-1):
    dupeList = np.where(dataLen == dataLen[i])[0]
    if len(dupeList) > 1:
        dupeCol[i] = ''
        for j in range(len(dupeList)-1):
            if j==0:
                dupeCol[i] = str(dir_list[dupeList[j+1]][:-4])
            else:
                dupeCol[i] = dupeCol[i] + ', ' + str(dir_list[dupeList[j+1]][:-4])
    if i == manualShrink[k]:
        isShrunk[i] = 1
        k = k + 1

df1 = pandas.DataFrame({'Duplicates':dupeCol})
df2 = pandas.DataFrame({'Shunk?':isShrunk})
origXL = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have.xlsx')

origXL['Duplicates'] = df1
origXL['Shrunk?'] = df2

origXL.to_excel('RCT IDs that we have.xlsx')