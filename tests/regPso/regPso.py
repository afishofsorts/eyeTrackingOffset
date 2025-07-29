import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

############################################################################
# ESTIMATES DENSITY PEAKS FOR EACH REGION OF EYE TRACKING DATA USING PSO   #
############################################################################

# grabbing directory names for each manually verified file
mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)

psoPeaks = np.zeros((6, 2, len(dir_list)-1)) # array initialization

for i in range(len(dir_list)-1):
    data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
    newData = cb.dataClean(data) # removes data  marked as invalid
    psoWeights, psoPeaks[:, :, i] = cb.regPso(newData) # computes regional COMs

np.save('tests\\regPso\\saved\\pso_peaks.npy', psoPeaks)

