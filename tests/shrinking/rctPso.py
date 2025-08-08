import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

################################################################################
# ESTIMATES DENSITY PEAKS FOR EACH REGION OF RCT EYE TRACKING DATA USING PSO   #
################################################################################

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have_updated.xlsx').to_numpy()[:, 0]

psoPeaks = np.zeros((6, 2, len(rctListTS))) # array initialization

for i in range(len(rctListTS)):
    data = pandas.read_csv(cd + '\\' + str(rctListTS[i]) + '.csv').to_numpy()
    newData = cb.dataClean(data) # removes data  marked as invalid
    psoWeights, psoPeaks[:, :, i] = cb.regPso(newData) # computes regional COMs

np.save('tests\\shrinking\\saved\\rct_peaks.npy', psoPeaks)

