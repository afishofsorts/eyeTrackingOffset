import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import random
from sko.PSO import PSO
import sys
from scipy import stats

##############################################
# LIST OF INTERNAL COMMANDS FOR THIS PROJECT #
##############################################

spCenter = np.array([920, 330]) # coordinates of speedometer center
rvCenter = np.array([1550, 660])

# xy pixel boundaries for each pertinent region of the screen
sm = np.array([[0, 550], [200, 600]]) # side mirror
speedom = np.array([[820, 1000], [200, 400]]) # speedometer
rv = np.array([[1300, 1900], [570, 800]]) # rearview
road = np.array([[700, 1100], [400, 650]])
panel = np.array([[1240, 1640], [0, 270]]) # bottom right console

# weighted average general center of mass and standard deviations for good data sets
comAvgs, comVars = np.load('saved\\com_stats.npy')

# shifts region boundary the same amount as general COM centering to the origin and adds 2 standard deviations of padding
def regConv(region, comAvgs):
    # INPUTS:
    # region:       2x2 array with x bounds in first row and y bounds in second, first column is minimums
    # comAvgs:      2D array
    # OUTPUTS:
    # newReg:       2x2 array

    newReg = region - (np.ones(2)[None].T @ comAvgs[None]).T
    return newReg

# converting all regions to be relative to the average general COM centered at the origin
newSm = regConv(sm, comAvgs)
newSpeedom = regConv(speedom, comAvgs)
newRv = regConv(rv, comAvgs)
newRoad = regConv(road, comAvgs)
newPanel = regConv(panel, comAvgs)

# returns data points that are within a given region and have been marked as valid
def dataClean(data, reg=np.array([[0, 0], [0, 0]])):
    # INPUTS:
    # data:         Array with rows of 6D data points
    # reg:          2x2 array with x bounds in first row and y bounds in second, first column is minimums
    # OUTPUTS:
    # validData:    Array with rows of 6D data points that are marked as valid and within reg

    if not (reg==np.array([[0, 0], [0, 0]])).all(): # checks if any region boundary has been input
        for l in range(len(data[:, 0])): # if so, marks data outside this region as invalid
            if data[l, 4]<reg[0, 0] or data[l, 4]>reg[0, 1] or data[l, 5]<reg[1, 0] or data[l, 5]>reg[1, 1]:
                data[l, 3] = 0

    validData = np.zeros(shape=(int(sum(data[:, 3])), 6)) # initializing array discluding data marked as invalid

    j = 0
    for k in range(len(data[:, 0])):
        if data[k, 3] == 1:
            validData[j, :] = data[k, :]
            j= j + 1
    return validData

# produces COM for 2D data set, weighted only if weight array is provided
def comGen(data, weights=np.array([])):
    # INPUTS:
    # data:         Array with rows of 6D dimensional data points
    # weights:      1D array of weights for each data point
    # OUTPUTS:
    #               2D array of the data's COM

    if len(weights)==0:
        return np.array([stat.fmean(data[:, 0]), stat.fmean(data[:, 1])])
    else: # only weights if weights are given
        return np.array([stat.fmean(data[:, 0], weights), stat.fmean(data[:, 1], weights)]) 

# density function for PSO
def boxCircFunc(p, data, r=100):
    # INPUTS:
    # p:            Center of 100 pixel ball used to calculate density
    # data:         Array with rows of 6D data points
    # OUTPUTS:
    #               Negative of the density

    n = len(data[:, 0])
    dispMatr = np.ones(shape=(n))[None].T @ p[None]
    newData = data-dispMatr
    inds = np.where(np.logical_and(abs(newData[:, 0])<50, abs(newData[:, 1])<100))[0] # first partitions data within 100 by 100 box centered at p
    l = len(inds)
    if l<10:
        return 99999999 # don't want to incentivize minimization of r over maximization of n
    else:
        sum = 0
        p = 0
        for i in range(l):
            r2 = newData[inds[i], :] @ newData[inds[i], :]
            if r2 < r**2: # only considers data within 50 pixels by radius
                sum = sum + r2
                p = p + 1
        if p == 0:
            return 99999999 # avoiding divide by zero in fitness function
        else:
            return sum/p**2

# Performs particle swarm optimization density peak estimate within some subset of data
def psoL2(region, data, r=100):
    # INPUTS:
    # region:       2x2 array with x bounds in first row and y bounds in second, 
    #               first column is minimums and second is maximums
    # data:         Array with rows of 6D data points
    # OUTPUTS:
    #               Numerical estimate of the local minimum

    psoFunc = lambda p: boxCircFunc(p, data[:, 4:], r) # function definition
    seed = random.randrange(0, sys.maxsize); random.seed(seed)
    pso = PSO(func=psoFunc, n_dim=2, pop=40, max_iter=30, lb=[region[0, 0], region[1, 0]], ub=[region[0, 1], region[1, 1]], w=0.7, c1=0.5, c2=0.5) # performs PSO fitting over omegas
    pso.run()

    return pso.gbest_x

# Performs pso optimization in 5 relevant regions of eye tracking data
def regPso(data, globalReg=np.array([[0, 0], [0, 0]]), isStandard=False):
    # INPUTS:
    # data:         Array with rows of 6D data points
    # OUTPUTS:
    #               List of two arrays: number of data points for each region and the estimated density peak in those regions
    
    validData = dataClean(data, globalReg) # does no region cleaning if globalReg is the default 0 matrix

    n = len(validData[:, 4])
    genCOM = comGen(validData[:, 4:])

    dispMatr = np.ones(len(validData[:, 0]))[None].T @ genCOM[None]
    validData[:, 4:] = validData[:, 4:] - dispMatr # centers all eye tracking points

    # partitions the data into relevant regions
    smData = dataClean(validData, newSm) 
    validData[:, 3] = np.ones(n) # because data cleaning changes validity array, this resets it
    speedomData = dataClean(validData, newSpeedom)
    validData[:, 3] = np.ones(n)
    rvData = dataClean(validData, newRv)
    validData[:, 3] = np.ones(n)
    roadData = dataClean(validData, newRoad)
    validData[:, 3] = np.ones(n)
    panelData = dataClean(validData, newPanel)

    # calculates regional Peaks
    smPeak = psoL2(newSm, smData)
    speedomPeak = psoL2(newSpeedom, speedomData)
    rvPeak = psoL2(newRv, rvData, 50)
    roadPeak = psoL2(newRoad, roadData)
    panelPeak = psoL2(newPanel, panelData)

    m1 = len(smData[:, 4]); m2 = len(speedomData[:, 4])
    m3 = len(rvData[:, 4]); m4 = len(roadData[:, 4]); m5 = len(panelData[:, 4])
    
    return np.array([n, m1, m2, m3, m4, m5]), np.array([genCOM, smPeak+genCOM, speedomPeak+genCOM, rvPeak+genCOM, roadPeak+genCOM, panelPeak+genCOM])

# given a reference point, returns the differences between that and a list of points
def relVectors(ref, rest):
    # INPUTS:
    # ref:          2D reference point
    # rest:         Array with rows of 2D points to be compared to ref
    # OUTPUTS:
    # difs:         Array with rows of 2D vectors from ref to each point in rest
    
    difs = np.zeros((len(rest)+1, 2)) # reserves the first vector as 0 for distance relative to itself
    for i in range(len(rest)):
        difs[i+1, :] = rest[i, :]-ref
    return difs

# Trnslationally shifts 3 dimensional data set
def translate(vec, data):
    # INPUTS:
    # vec:          1x2 array for translation 
    # data:         Array with rows of 2D data points
    # OUTPUTS:
    #               Array with rows of 2D translated data points
    
    return data + vec[None].repeat(len(data[:, 0]), 0)

# Checks if a 2 dimensional vector's probability of being drawn from a multivariate normal distribution is less than some benchmark p
def isOutlier(x, mean, covar, p=0.05):
    # INPUTS:
    # x:        1x2 array of sample point
    # mean:     1x2 array of multivariate normal mean
    # covar:    2x2 array of multivariate normal covariance
    # p:        Probability cutoff for outliers
    # OUTPUTS:
    #           Boolean value indicating if the sample point had a less than p chance of being drawn
    
    d2 = (x-mean) @ np.linalg.inv(covar) @ (x-mean).T # square of Mahalanobis distance
    if 1-stats.chi2.cdf(d2, 2) > p: # probability of being within Mahalonibis distance is given by deg 2 chi squared
        return False
    if 1-stats.chi2.cdf(d2, 2) <= p:
        return True

covar = np.load('saved\\pso_SPRD_covar.npy') # measured covariance of difference between speedometer and road density peaks
mean = np.load('saved\\pso_SPRD_mean.npy') # measured mean of difference between speedometer and road density peaks

# Corrects eye tracking data offset using pso optimized density peak for the speedometer
def SPCorr(data):
    # INPUTS:
    # data:         Array with rows of 2D data points
    # OUTPUTS:
    #               List of two arrays: array series of 2 dimensional correct data and boolean value indicating if
    #               the road to speedometer vector if an outlier
    
    validData = dataClean(data) # does no region cleaning is globalReg is the default 0 matrix

    n = len(validData[:, 4])
    genCOM = comGen(validData[:, 4:])

    dispMatr = np.ones(len(validData[:, 0]))[None].T @ genCOM[None]
    validData[:, 4:] = validData[:, 4:] - dispMatr # centers all eye tracking points

    speedomData = dataClean(validData, newSpeedom)
    validData[:, 3] = np.ones(n)
    roadData = dataClean(validData, newRoad)

    # calculates regional Peaks
    speedomPeak = psoL2(newSpeedom, speedomData)
    roadPeak = psoL2(newRoad, roadData)

    corr = spCenter - speedomPeak - genCOM
    
    return translate(corr, data[:, 4:]), isOutlier(roadPeak-speedomPeak, mean, covar)

# Calculates the most likely position a series translationally offset normal distribution measurements
# were in originally
def meanPeakCorr(avgPeaks, difs, vars):
    # INPUTS:
    # avgPeaks: Array with rows of 2D density peak reference points used to corrcect to
    # difs:     2D array of differences between the measured density peaks, not avgPeaks.
    #           Zero entry indicates the density peak every other peak is relative to
    # vars:     x and y variances of each density peak
    # OUTPUTS:
    #           2D array of most likely position of zero entry in difs

    return np.diag((avgPeaks-difs).T @ (1/vars)/sum(1/vars))

# Finds the number of points with a box of width 2*w centered on 2D point p
def nCheck(p, data, w):
    # INPUTS:
    # p:        2D array for the center of box
    # data:     Array series of 6 dimensional data points
    # w:        Scalar for half the width of the box
    # OUTPUTS:
    #           Number of points within box centered on p

    n = len(data[:, 0])
    dispMatr = np.ones(shape=(n))[None].T @ p[None] 
    newData = data-dispMatr # centers p at origin
    inds = np.where(np.logical_and(abs(newData[:, 0])<w, abs(newData[:, 1])<w))[0] # grabs data within w of either component of p
    return len(inds)

# Checks if rearview density peak is eligible to be used for offset correction
def rvCheck(rdPeak, rvPeak, speedomPeak, data):
    # INPUTS:
    # rdPeak:       2D array of road density peak
    # rvPeak:       2D array of rearview density peak
    # speedomPeak:  2D array of speedometer density peak
    # OUTPUTS:
    #               Boolean value

    RVRD = rvPeak - rdPeak
    RVSP = rvPeak - speedomPeak
    rdCheck = (475<RVRD[0]<700 and 75<RVRD[1]<185) # checks if road to rearview vector is excessive
    speedomCheck = (550<RVSP[0]<750 and 250<RVSP[1]<400)# checks if speedometer to rearview vector is excessive
    isN = nCheck(rvPeak, data, 50)>25 # checks if the number of data points within 100 pixel box is greater than 30
    return rdCheck and speedomCheck and isN

# loads upper limit of variances for speedometer and rearview density peaks
peakVars = np.array([np.diag(np.load('saved\\pso_SPRD_covar.npy')), np.diag(np.load('saved\\pso_RVRD_covar.npy'))])

# Corrects eye tracking data offset using pso optimized density peak for the speedometer
def SPRVCorr(data):
    # INPUTS:
    # data:         Array with rows of 2D data points
    # OUTPUTS:
    #               List of two arrays: array series of 2 dimensional correct data and boolean value indicating if
    #               the road to speedometer vector if an outlier
    
    validData = dataClean(data) # does no region cleaning if globalReg is the default 0 matrix

    n = len(validData[:, 4])
    genCOM = comGen(validData[:, 4:])

    dispMatr = np.ones(len(validData[:, 0]))[None].T @ genCOM[None]
    validData[:, 4:] = validData[:, 4:] - dispMatr # centers all eye tracking points

    speedomData = dataClean(validData, newSpeedom)
    validData[:, 3] = np.ones(n)
    roadData = dataClean(validData, newRoad)
    validData[:, 3] = np.ones(n)
    rvData = dataClean(validData, newRv)

    # calculates regional Peaks
    speedomPeak = psoL2(newSpeedom, speedomData)
    roadPeak = psoL2(newRoad, roadData)
    rvPeak = psoL2(newRv, rvData)

    # if rearview density peak isn't an outlier or lacks enough points, corrects based off of both it and speedometer
    if rvCheck(roadPeak, rvPeak, speedomPeak, rvData[:, 4:]): 
        corrX1 = meanPeakCorr(np.array([spCenter, rvCenter]), np.array([[0, 0], rvPeak-speedomPeak]), peakVars)
        return translate(corrX1-speedomPeak-genCOM, data[:, 4:]), isOutlier(roadPeak-speedomPeak, mean, covar)
    else: # otherwise corrects just off of speedometer
        corr = spCenter - speedomPeak - genCOM
        return translate(corr, data[:, 4:]), isOutlier(roadPeak-speedomPeak, mean, covar)

# Plots eye tracking data and other relevant points
def plotEyeData(data, filename='image', genCOM=np.array([0, 0]), psoPeaks=np.array([0, 0]), dir='',title='', save=True, standards=False):
    # INPUTS:
    # data:         Array with rows of 2D data points
    # filename:     String for name of saved image
    # genCOM:       2D array of center of mass of data, if zero not plotted
    # psoPeaks:     Array with rows of 2D density peaks, if zero not plotted
    # dir:          String of directory to save file to
    # title:        String of title to be drawn on plot
    # save:         Boolean indicating if the image should be saved
    # standards:    Boolean indicating if speedometer and rearview reference points should be plotted

    plt.rcParams["figure.figsize"] = [19.00, 9.0]
    img = plt.imread("saved\\Primary Care Calibration\\Exemplar Scene (No Border).jpg")
    fig, ax = plt.subplots()
    im = ax.imshow(img, extent=[0, 1920, 0, 1080])
    plt.plot(data[:, 0], data[:, 1], 'o', mec = 'black', mew='0.8', markersize=4, color='fuchsia', label='Eye Tracking Data')
    plt.xlim(0, 1920); plt.ylim(0, 1080)
    plt.title(title)
    plt.xlabel('x position (pixels)'); plt.ylabel('y position (pixels)')

    if not np.all(genCOM==np.array([0, 0])):
        plt.plot(genCOM[0], genCOM[1], '*', mec='black', color='gold', markersize=25, label='COM')
    if not np.all(psoPeaks==np.array([0, 0])):
        plt.plot(psoPeaks[0, 0], psoPeaks[0, 1], 'd', color='limegreen', mec='black', markersize=13, label='Density Peaks')
        for i in range(len(psoPeaks[:, 0])-1):
            plt.plot(psoPeaks[i+1, 0], psoPeaks[i+1, 1], 'd', color='limegreen', mec='black', markersize=13)
    if standards:
        plt.plot(spCenter[0], spCenter[1], 'o', color='gold', markersize=6, mec = 'black', mew='1.0', label='Standards')
        plt.plot(rvCenter[0], rvCenter[1], 'o', color='gold', markersize=6, mec = 'black', mew='1.0')
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)

    if save:
        plt.savefig(dir + '\\' + filename, bbox_inches='tight', pad_inches=0.5)
        plt.close()
