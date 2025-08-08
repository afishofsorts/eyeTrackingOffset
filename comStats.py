import numpy as np
import pandas
import os
from commandBank import commandBank as cb

################################################################
# GENERATES THE AVERAGE CENTER OF MASS FROM GOOD MVP DATA      #
################################################################

# grabbing directory names for each manually verified file
mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)
# tags for each participant's data
calList = pandas.read_excel(mvp + '\\Calibration List.xlsx').to_numpy()

# marks which data sets are good with a 1
inds = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    if calList[i, 1]=='Good':
        inds[i] = 1

genCOM = np.zeros(shape=(2, int(sum(inds)))) # array initialization
weights = np.zeros((int(sum(inds))))
k=0
for i in range(len(dir_list)-1):
    if inds[i]==1: # only performs correction for data marked as Good
        data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
        newData = cb.dataClean(data) # removes data marked as invalid
        genCOM[:, k] = cb.comGen(newData[:, 4:])
        weights[k] = len(newData[:, 4])
        k = k+1

avgCom = cb.comGen(genCOM.T, weights) # finds weighted average of general COMs
print(avgCom)
comCovar = np.cov(genCOM, fweights=weights)

np.save('saved\\com_stats.npy', [avgCom, np.diag(comCovar)])
np.save('saved\\com_raw.npy', genCOM)
