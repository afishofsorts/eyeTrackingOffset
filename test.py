import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

##############################################
# GENERATES PLOTS FOR REGIONAL DENSITY PEAKS #
##############################################

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

psoPeaks = np.load('tests\\regPso\\saved\\pso_peaks.npy')

k=0 # indexing for errDist
for i in range(3):
    if inds[i]==1: # only performs correction for data marked as Good
        data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
        newData = cb.dataClean(data) # removes data  marked as invalid
        comWeights, psoPeaks[:, :, k] = cb.regPso(newData) # computes regional COMs
        plt.rcParams["figure.figsize"] = [20.00, 10.0]
        img = plt.imread("saved\\Primary Care Calibration\\Exemplar Scene Cropped.jpg")
        fig, ax = plt.subplots()
        im = ax.imshow(img, extent=[0, 1920, 0, 1080])
        plt.plot(newData[:, 4], newData[:, 5], 'o', mec = 'black', mew='0.8', markersize=4, color='fuchsia')
        # plt.plot(psoPeaks[0, 0, k], psoPeaks[0, 1, k], '*', mec='black', color='gold', markersize=25)
        plt.plot(psoPeaks[2, 0, k], psoPeaks[2, 1, k], 'd', color='limegreen', mec='black', markersize=13)
        plt.plot(psoPeaks[4, 0, k], psoPeaks[4, 1, k], 'd', color='limegreen', mec='black', markersize=13)
        plt.xlim(0, 1920); plt.ylim(0, 1080)
        ax.set_axis_off()


        # Put a legend to the right of the current axis
        # ax.legend(['Eye Tracking Data', 'General COM', 'PSO Density Peak'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('pso_' + str(dir_list[i+1][:-4]),bbox_inches='tight', pad_inches=0)
        plt.close()
        k=k+1
