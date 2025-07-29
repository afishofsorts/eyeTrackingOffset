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
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have.xlsx').to_numpy()[:, 0]

dataLen = np.zeros(len(rctListTS))
for i in range(len(rctListTS)-1):
    data = pandas.read_csv(cd + '\\saved\\data\\' + str(rctListTS[i]) + '.csv').to_numpy()
    dataLen[i] = data[-1, 0]

manualShrink = np.array([0, 1, 2, 3, 4, 9, 13, 16, 17, 18 , 26, 29, 40, 41, 43, 72, 85, 97])
dupeCol = ['N/A']*(len(rctListTS))
justData = ['N/A']*len(rctListTS)
isShrunk = np.zeros(len(rctListTS))
k = 0
for i in range(len(rctListTS)):
    dupeList = np.where(dataLen == dataLen[i])[0]
    if len(dupeList) > 1:
        dupeCol[i] = ''
        l = 0
        for j in range(len(dupeList)):
            if dupeList[j] != i:
                if l==0:
                    dupeCol[i] = str(rctListTS[dupeList[j]])
                    if dupeCol[i] != rctListTS[i]:
                        justData[i] = str(rctListTS[dupeList[j]])
                else:
                    dupeCol[i] = dupeCol[i] + ', ' + str(rctListTS[dupeList[j]])
                    if str(rctListTS[dupeList[j]]) != rctListTS[i]:
                        justData[i] = justData[i] + ', ' + str(rctListTS[dupeList[j]])
                l = l + 1
    if i == manualShrink[k]:
        isShrunk[i] = 1
        k = k + 1

idDupeListTS = np.zeros(len(rctListTS))
for i in range(len(rctListTS)):
    dupeListTS = np.where(rctListTS == rctListTS[i])[0]
    idDupeListTS[i] = len(dupeListTS)-1

df1 = pandas.DataFrame({'Duplicate Data':dupeCol})
df2 = pandas.DataFrame({'Duplicate IDs':idDupeListTS})
df3 = pandas.DataFrame({'Duplicate Data, Not ID':justData})
df4 = pandas.DataFrame({'Shunk?':isShrunk})
origXL = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have.xlsx')

origXL['Duplicate Data'] = df1
origXL['Duplicate IDs'] = df2
origXL['Duplicate Data, Not ID'] = df3
origXL['Shrunk?'] = df4


origXL.to_excel('RCT IDs that we have.xlsx', index=False)