'''
Created by Andrew Pepper 7/11/2019
'''
import numpy as np
import scipy.optimize as opt
import sys



G = 6.67e-8 # CGS



# returns an array of indexes which lie in the 2d planar subset of the 
# simulation domain. 
# 'p' -> a point in the plane
# 'n' -> the normal of the plane
def slice2dFrom3d(centers, widths, p, n):
    n = n/np.linalg.norm(n)
    inds = []
    for i, (x, y, z, wx, wy, wz) in enumerate(zip(centers[0], centers[1], centers[2], widths[0], widths[1], widths[2])):
        relPos = np.asarray([x - p[0], y - p[1], z - p[2]])
        '''
        maxDist = pow(3, 0.5)*max(widths[0][i], 
                                  widths[1][i], 
                                  widths[2][i])
        '''
        cornerDist = [-np.inner([ wx,  wy,  wz], n),
                      -np.inner([ wx,  wy, -wz], n),
                      -np.inner([ wx, -wy,  wz], n),
                      -np.inner([ wx, -wy, -wz], n),
                      -np.inner([-wx,  wy,  wz], n),
                      -np.inner([-wx,  wy, -wz], n), 
                      -np.inner([-wx, -wy,  wz], n),
                      -np.inner([-wx, -wy, -wz], n)]
        for cd in cornerDist:
            if abs( abs(np.inner(relPos, n))/cd - 1 ) < 1024*sys.float_info.epsilon:
                inds.append(i)
                break
        
    return np.asarray(inds)



def getCOM3d(centers, Etots, KEs, IEs, masses):
    comGuess = getCOM3d_old(centers, masses)
    GPE = np.asarray(Etots) - np.asarray(KEs) - np.asarray(IEs)
    GPE_ord = np.argsort(GPE)
    bestInd = []
    # 16 is an qualitative, empirical choice
    numBestInds = 16
    for i in GPE_ord:
        distToCOM = np.linalg.norm([centers[0, i] - comGuess[0],
                                    centers[1, i] - comGuess[1], 
                                    centers[2, i] - comGuess[2]])
        if  distToCOM < 7e8: # 7e8 ~ R_Earth
            bestInd.append(i)
            if len(bestInd) >= numBestInds:
                break

    com = [0, 0, 0]
    for i in bestInd:
        com[0] += centers[0, i]/numBestInds
        com[1] += centers[1, i]/numBestInds
        com[2] += centers[2, i]/numBestInds

    return com, bestInd



def getCOM3d_grav(centers, masses):
    Us = np.zeros(np.asarray(masses).shape)
    for i, u in enumerate(Us):
        for x, y, z, m in zip(np.asarray(centers[0]), np.asarray(centers[1]), np.asarray(centers[2]), np.asarray(masses)):
            r = np.linalg.norm([x, y, z])
            u -= G*m/r

    print Us
    minUarg = np.argmin(Us)
    return [ centers[0][minUarg], 
             centers[1][minUarg], 
             centers[2][minUarg] ]



def getCOM3d_old(centers, masses):
    com = [0, 0, 0]

    for x, y, z, m in zip(np.asarray(centers[0]), np.asarray(centers[1]), np.asarray(centers[2]), np.asarray(masses)):
        com[0] += m * x
        com[1] += m * y
        com[2] += m * z

    M = 0.0
    for m in masses:
        M += m

    com[0] /= M
    com[1] /= M
    com[2] /= M

    return com


def getRads3d(centers, com=[0, 0, 0]):
    rads = []

    for x, y, z in zip(np.asarray(centers[0]), np.asarray(centers[1]), np.asarray(centers[2])):
        rads.append(np.linalg.norm([x - com[0], y - com[1], z - com[2]]))

    return np.asarray(rads)



def getL3d(centers, vels, masses, com):
    Ls = []

    for x, y, z, vx, vy, vz, m in zip(centers[0], centers[1], centers[2], vels[0], vels[1], vels[2], masses):
        r = [x - com[0], y - com[1], z - com[2]]
        Ls.append(np.cross(r, [vx, vy, vz])*m)
    
    return Ls
