import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

###################################################################
# INVESTIGATES DISTANCES BETWEEN PEAKS FOR SHRINKING CHECK        #
###################################################################

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have_updated.xlsx').to_numpy()[:, 0]

savedDir = 'tests\\shrinking\\saved'
psoPeaks = np.load(savedDir + '\\rct_peaks.npy')

sprd = psoPeaks[4, :, :] - psoPeaks[2, :, :]
sprv = psoPeaks[3, :, :] - psoPeaks[2, :, :]
rdrv = psoPeaks[4, :, :] - psoPeaks[3, :, :]

Rsprd = np.sqrt((sprd.T)**2 @ np.ones(2))
Rsprv = np.sqrt((sprv.T)**2 @ np.ones(2))
Rrdrv = np.sqrt((rdrv.T)**2 @ np.ones(2))

plt.hist(Rsprd+Rsprv+Rrdrv)
plt.show()
