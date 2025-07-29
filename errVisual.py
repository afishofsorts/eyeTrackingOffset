import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from commandBank import commandBank as cb

t = np.linspace(0, 2*np.pi, 1000)
x = 42*np.cos(t) + cb.spCenter[0]-10; y = 42*np.sin(t) + cb.spCenter[1]-10

plt.rcParams["figure.figsize"] = [20.00, 10.0]
img = plt.imread("saved\\Primary Care Calibration\\Exemplar Scene Cropped.jpg")
fig, ax = plt.subplots()
im = ax.imshow(img, extent=[0, 1920, 0, 1080])
plt.plot(1550, 660, 'o', color='r')


plt.plot(cb.spCenter[0]-10, cb.spCenter[1]-10, 'o', mec='black', markersize=7, color='fuchsia')
plt.plot(x, y, color='r')
plt.xlim(0, 1920); plt.ylim(0, 1080)
ax.set_axis_off()
plt.savefig('err_visual',bbox_inches='tight', pad_inches=0)
