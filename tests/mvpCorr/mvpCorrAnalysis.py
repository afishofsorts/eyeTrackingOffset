import numpy as np
import matplotlib.pyplot as plt
import os
import pandas

mvp = 'saved\\Primary Care Calibration\\Manually Validated Participants'
dir_list = os.listdir(mvp)

isOutlier = np.load('tests\\mvpCorr\\saved\\data\\isOutlier.npy')
for i in range(len(dir_list)-1):
    corrCoords = pandas.read_csv('tests\\mvpCorr\\saved\\data\\' + str(dir_list[i+1])).to_numpy()[:, 1:]

    plt.rcParams["figure.figsize"] = [20.00, 10.0]
    img = plt.imread("saved\\Primary Care Calibration\\Exemplar Scene Cropped.jpg")
    fig, ax = plt.subplots()
    im = ax.imshow(img, extent=[0, 1920, 0, 1080])
    plt.plot(corrCoords[:, 0], corrCoords[:, 1], 'o', mec = 'black', mew='0.5', markersize=4, color='fuchsia')
    plt.plot(920, 312, 'o', mec='black', color='gold', markersize=8)
    plt.xlim(0, 1920); plt.ylim(0, 1080)

    plt.title('PSO Speedometer Comparison for ' + str(dir_list[i+1][:-4]))
    plt.xlabel('x position (pixels)'); plt.ylabel('y position (pixels)')
    plt.text(1350, 1000, 'isOutlier = ' + str(isOutlier[i]), color='w', size=20)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])

    # Put a legend to the right of the current axis
    ax.legend(['PSO SP', 'REF'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('tests\\mvpCorr\\saved\\figs\\pso_corr_' + str(dir_list[i+1][:-4]), bbox_inches='tight', pad_inches=0.5)
    plt.close()