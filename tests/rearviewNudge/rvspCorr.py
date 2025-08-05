import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
from commandBank import commandBank as cb
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt

# grabbing directory names for each manually verified file
mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)
# tags for each participant's data
calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()

# marks which data sets are Good with a 1
inds = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    if calList[i, 1]=='Good' or calList[i, 1]=='Offset':
        inds[i] = 1

isOutlier = np.zeros(int(sum(inds)))
k=0
for i in range(int(sum(inds))):
    if inds[i] == 1:
        data = pandas.read_csv(mvp+ '\\' + str(dir_list[i+1])).to_numpy()
        validData = cb.dataClean(data)

        corrData, isOutlier[k] = cb.SPRVCorr(validData)

        df = pandas.DataFrame(corrData)
        df.to_csv('tests\\rearviewNudge\\saved\\data\\rvspCorr\\' + str(dir_list[i+1]))
        k=k+1

np.save('tests\\rearviewNudge\\saved\\data\\is_outlier.npy', isOutlier)