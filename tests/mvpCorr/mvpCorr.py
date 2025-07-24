import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

###############################################################################
# CORRECTS AND PLOTS MVP EYE TRACKING DATA USING SPEEDOMETER DENSITY PEAK     #
###############################################################################

# grabbing directory names for each manually verified file
mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)

isOutlier = np.zeros(len(dir_list)-1)
for i in range(len(dir_list)-1):
    data = pandas.read_csv(mvp + '\\' + str(dir_list[i+1])).to_numpy()
    
    corrCoords, isOutlier[i] = cb.SPCorr(data)
    df = pandas.DataFrame(corrCoords)
    df.to_csv('tests\\mvpCorr\\saved\\data\\' + str(dir_list[i+1]))
    
np.save('tests\\mvpCorr\\saved\\data\\isOutlier.npy', isOutlier)


