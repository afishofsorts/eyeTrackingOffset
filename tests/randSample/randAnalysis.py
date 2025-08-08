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
rootDir = 'tests\\randSample\\saved\\data\\corr'
dir_list = os.listdir(rootDir)

savedDir = 'tests\\randSample\\saved'

for i in range(len(dir_list)):
    data = pandas.read_csv(rootDir + '\\' + str(dir_list[i])).to_numpy()[:, 1:]
    newData = cb.dataClean(data) # removes data  marked as invalid
    psoPeaks = pandas.read_csv(savedDir + '\\data\\peaks\\peaks' +  str(dir_list[i][4:-4]) + '.csv').to_numpy()[:, 1:]

    cb.plotEyeData(newData[:, 4:], 'pso_peaks_' + str(dir_list[i][4:-4]), title='PSO Density Peaks for ' + str(dir_list[i]),
                    dir=savedDir + '\\figs\\corr', genCOM=psoPeaks[0, :], psoPeaks=psoPeaks[1:, :])