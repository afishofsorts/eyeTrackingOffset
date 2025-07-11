from scipy import stats
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb
import numpy as np
import os
import pandas

# loads regional COM anchor points
goldStats = np.load('saved\\gold_stats.npy')
silverStats = np.load('saved\\silver_stats.npy')

# grabbing directory names for each manually verified file
mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)
# tags for each participant's data
calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()

# marks which data sets are Good with a 1
inds = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    if calList[i, 1]=='Offset':
        inds[i] = 1

# combining general COM and regional COMs
avgComs = np.append([goldStats[0]], silverStats[0], axis=0)
vars = np.append([goldStats[1]], silverStats[1], axis=0)


outDists = np.load('saved\\outlier_dists.npy')
k=0
for i in range(len(dir_list)-1):
    data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
    newData = cb.dataClean(data) # removes data  marked as invalid
    comWeights, coms = cb.regComs(data, goldStats[0]) # computes regional COMs
    
    flags = cb.flag(coms[1:, :], avgComs[1:, :], outDists[1:])
    if sum(flags) > 1 or calList[i, 1]=='Miscalibrated':
        plt.text(1500, 700, str(flags))
        cb.plotComComp(goldStats, silverStats, coms, newData, 'tests\\flagTest\\saved\\' + str(calList[i, 1]) + '_' + str(dir_list[i+1][:-4]))