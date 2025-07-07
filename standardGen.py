import numpy as np
import pandas
import os
from commandBank import commandBank as cb

##############################################
# GENERATES THE STANDARD ANCHOR COORDINATES  #
##############################################

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

goldPt = cb.comGen(genCOM.T, weights) # finds weighted average of general COMs
print(goldPt)
goldCovar = np.cov(genCOM, fweights=weights)

np.save('saved\\gold_stats.npy', [goldPt, np.diag(goldCovar)])
np.save('saved\\gold_raw.npy', genCOM)

regCOMS = np.zeros(shape=(6, 2, int(sum(inds)))) # array initialization
regWeights = np.zeros(shape=(6, int(sum(inds))))
k=0
for i in range(len(dir_list)-1):
    if inds[i]==1: 
        data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
        newData = cb.dataClean(data)
        regWeights[:, k], regCOMS[:, :, k] = cb.regComs(data, goldPt) # computes regional COMs
        k = k+1

silverPts = np.zeros((5, 2))
silverVars = np.zeros((5, 2))
for i in range(5):
    silverPts[i] = cb.comGen(regCOMS[i+1, :, :].T, regWeights[i+1]) # finds weighted average of each regional COM
    silverVars[i, :] = np.diag(np.cov(regCOMS[i+1, :, :], fweights=regWeights[i+1]))

print(silverVars)
np.save('saved\\silver_stats.npy', [silverPts, silverVars])
np.save('saved\\silver_raw.npy', regCOMS[1:, :, :])
cb.plotComs(regCOMS, 'Regional COMs for Good Data', 'saved\\figures\\reg_coms.jpg')