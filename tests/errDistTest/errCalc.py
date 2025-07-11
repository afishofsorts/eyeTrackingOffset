import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

##############################################
# TESTS CORRECTION ALGORITHM ERROR DATA      #
##############################################

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
    if calList[i, 1]=='Good':
        inds[i] = 1

# combining general COM and regional COMs
avgComs = np.append([goldStats[0]], silverStats[0], axis=0)
vars = np.append([goldStats[1]], silverStats[1], axis=0)


errDist = np.zeros(int(sum(inds))) # array initialization
k=0 # indexing for errDist
for i in range(len(dir_list)-1):
    if inds[i]==1: # only performs correction for data marked as Good
        data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
        newData = cb.dataClean(data) # removes data  marked as invalid
        comWeights, coms = cb.regComs(data, goldStats[0]) # computes regional COMs

        difs = cb.relVectors(coms[1, :], coms[2:, :])
        err = cb.meanComCorr(avgComs[1:, :], difs, vars[1:]) - coms[1, :] # since original data is already considered Good, any correction is the error
        errDist[k] = np.sqrt(err @ err)
        plt.rcParams["figure.figsize"] = [10.00, 5.0]
        img = plt.imread("saved\\Primary Care Calibration\\Calibration Shape.jpg")
        fig, ax = plt.subplots()
        im = ax.imshow(img, extent=[0, 1920, 0, 1080])
        plt.plot(newData[:, 4], newData[:, 5], 'o', markersize=2)
        plt.plot(goldStats[0][0], goldStats[0][1], '*', color='y', markersize=10)
        plt.plot(silverStats[0][:, 0], silverStats[0][:, 1], 'D', color='grey', markersize=10)
        plt.plot(coms[0, 0], coms[0, 1], 'o', color='r')
        plt.plot(coms[1:, 0], coms[1:, 1], 'o', color='orange')
        plt.title('COMS for ' + str(dir_list[i+1][:-4]) + ' Compared To Standards')
        plt.text(50, 100, 'Correction Vector: [' + str(round(err[0], 2)) + ', ' + str(round(err[1], 2)) + ']', fontsize='small', color='w')
        plt.text(50, 50, 'Correction Dist: ' + str(round(errDist[k], 2)), fontsize='small', color='w')
        plt.xlabel('x position (pixels)'); plt.ylabel('y position (pixels)')
        plt.xlim(0, 1920); plt.ylim(0, 1080)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

        # Put a legend to the right of the current axis
        ax.legend(['Eye Tracking Data', 'General COM Standard', 'Regional COM Standards', 'General COM', 'Regional COMS'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('tests\\errDistTest\\saved\\good_corr_' + str(dir_list[i+1][:-4]))
        plt.close()
        k = k + 1

np.save('tests\\errDistTest\\saved\\err_data.npy', errDist)

        

