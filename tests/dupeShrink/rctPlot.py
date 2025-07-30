import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

cd = 'saved\\Primary Care Calibration\\Time-Series Data'
rctListTS = pandas.read_excel('saved\\Primary Care Calibration\\RCT IDs that we have.xlsx').to_numpy()[:, 0]

for i in range(len(rctListTS)):
    data = pandas.read_csv(cd + '\\' + str(rctListTS[i]) + '.csv').to_numpy()
    cb.plotEyeData(data, filename=rctListTS[i], dir=cd + '\\')








        

