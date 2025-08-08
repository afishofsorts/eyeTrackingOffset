import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

###########################################################################
# INVESTIGATES DATA SPREAD TO INVESTIGATE DIFFERENCES WITH SHRUNK DATA    #
###########################################################################

mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)

for i in range(len(dir_list)-1):
    data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
    newData = cb.dataClean(data)

    bins=np.arange(-300, 1500, 100)
    n = plt.hist(newData[:, 5], bins=bins)
    plt.title(sum(n[0][:4]))
    plt.savefig('tests\\shrinking\\saved\\hists\\mvp\\' + str(dir_list[i+1][:-4]) + '_y')
    plt.close()

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have_updated.xlsx').to_numpy()[:, 0]

for i in range(len(rctListTS)):
    data = pandas.read_csv(cd + '\\' + str(rctListTS[i]) + '.csv').to_numpy()
    newData = cb.dataClean(data)

    bins=np.arange(-300, 1500, 100)
    n = plt.hist(newData[:, 5], bins=bins)

    plt.title(sum(n[0][:4]))
    plt.savefig('tests\\shrinking\\saved\\hists\\rct\\' + str(rctListTS[i]) + '_y')
    plt.close()

