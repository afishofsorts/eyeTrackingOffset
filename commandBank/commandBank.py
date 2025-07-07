import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

##############################################
# LIST OF INTERNAL COMMANDS FOR THIS PROJECT #
##############################################

# xy pixel boundaries for each pertinent region of the screen
sm = np.array([[0, 550], [200, 600]]) # side mirror
speedom = np.array([[820, 1000], [200, 400]]) # speedometer
rv = np.array([[1250, 1900], [540, 800]]) # rearview
road = np.array([[600, 1050], [400, 550]])
panel = np.array([[1240, 1640], [0, 270]]) # bottom right console

# weighted average general center of mass and standard deviations for good data sets
comAvgs, comVars = np.load('saved\\gold_stats.npy')
comSDs = np.sqrt(comVars)

# shifts region boundary the same amount as general COM centering to the origin and adds 2 standard deviations of padding
def regConv(region, comAvgs, comSDs):
    newReg = region - (np.ones(2)[None].T @ comAvgs[None]).T + 2*(np.array([-1, 1])[None].T @ comSDs[None]).T
    return newReg

# converting all regions to be relative to the average general COM centered at the origin
newSm = regConv(sm, comAvgs, comSDs)
newSpeedom = regConv(speedom, comAvgs, comSDs)
newRv = regConv(rv, comAvgs, comSDs)
newRoad = regConv(road, comAvgs, comSDs)
newPanel = regConv(panel, comAvgs, comSDs)

# returns data points that are within a given region and have been marked as valid
def dataClean(data, reg=np.array([[0, 0], [0, 0]])):
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

# produces COM for 2D data set, weighted only if 1D weight array is provided
def comGen(data, weights=np.array([])):
    if len(weights)==0:
        return np.array([stat.fmean(data[:, 0]), stat.fmean(data[:, 1])])
    else: 
        return np.array([stat.fmean(data[:, 0], weights), stat.fmean(data[:, 1], weights)])

# returns all anchor region COMs given a set of 2D eye tracking data
def regComs(data, goldCOM, globalReg=np.array([[0, 0], [0, 0]])):
    validData = dataClean(data, globalReg) # does no region cleaning is globalReg is the default 0 matrix

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

    # calculates regional COMs
    smCOM = comGen(smData[:, 4:])
    speedomCOM = comGen(speedomData[:, 4:])
    rvCOM = comGen(rvData[:, 4:])
    roadCOM = comGen(roadData[:, 4:])
    panelCOM = comGen(panelData[:, 4:])

    m1 = len(smData[:, 4]); m2 = len(speedomData[:, 4])
    m3 = len(rvData[:, 4]); m4 = len(roadData[:, 4]); m5 = len(panelData[:, 4])
    
    # shifts the regional COMs back relative to the general COM anchor point
    return np.array([n, m1, m2, m3, m4, m5]), np.array([genCOM, smCOM+goldCOM, speedomCOM+goldCOM, rvCOM+goldCOM, roadCOM+goldCOM, panelCOM+goldCOM])

# plots a series of regional COMs
def plotComs(coms, title='Regional Center of Masses', dir='None'):
    plt.xlim(0, 1920); plt.ylim(0, 1080)
    for i in range(6):
        plt.plot(coms[i, 0, :], coms[i, 1, :], 'o', markersize=2)
    plt.title(title)
    plt.xlabel('x position (pixels)'); plt.ylabel('y position (pixels)')
    plt.legend(['General', 'Side Mirror', 'Speedometer', 'Rearview', 'Road', 'Console'])
    if dir=='None':
        plt.show()
    else: 
        plt.savefig(dir)

# given a reference point, returns the differences between that and a list of vectors
def relVectors(ref, rest):
    difs = np.zeros((len(rest)+1, 2)) # reserves the first vector as 0 for distance relative to itself
    for i in range(len(rest)):
        difs[i+1, :] = rest[i, :]-ref
    return difs

# computes the correction vector given standard anchor COMs, the difference vectors of a single data set's COMs, and the anchor variances
def meanComCorr(avgComs, difs, vars):
    return np.diag((avgComs-difs).T @ (1/vars)/sum(1/vars))