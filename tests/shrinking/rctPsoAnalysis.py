import numpy as np
import pandas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

##################################################
# GENERATES PLOTS FOR RCT REGIONAL DENSITY PEAKS #
##################################################

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have_updated.xlsx').to_numpy()[:, 0]

savedDir = 'tests\\shrinking\\saved'
psoPeaks = np.load(savedDir + '\\rct_peaks.npy')

for i in range(len(rctListTS)):
    data = pandas.read_csv(cd + '\\' + str(rctListTS[i]) + '.csv').to_numpy()
    newData = cb.dataClean(data) # removes data marked as invalid

    cb.plotEyeData(newData[:, 4:], 'rct_peaks_' + str(rctListTS[i]), title='PSO Density Peaks for ' + str(rctListTS[i]),
                    dir=savedDir + '\\figs', genCOM=psoPeaks[0, :, i], psoPeaks=psoPeaks[1:, :, i])
