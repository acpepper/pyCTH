import matplotlib as mpl
mpl.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from matplotlib.colors import LogNorm
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

fmtr = ScalarFormatter()
fmtr.set_powerlimits((2, 2))

import DataOutBinReader as dor
import HcthReader as hcr

from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)



G = 6.67e-8 # CGS


'''
def getFractionalLs(centers, vels, com, masses, escpdInds, diskInds):
    c_of_escpd_mass = [np.asarray(centers[0])[escpdInds],
                       np.asarray(centers[1])[escpdInds],
                       np.asarray(centers[2])[escpdInds]]
    v_of_escpd_mass = [np.asarray(vels[0])[escpdInds],
                       np.asarray(vels[1])[escpdInds],
                       np.asarray(vels[2])[escpdInds]]
    escpdL = np.asarray( pps.getL3d(c_of_escpd_mass, 
                                    v_of_escpd_mass, 
                                    np.asarray(masses)[escpdInds],
                                    com) )
    if len(escpdL) > 0:
        L_esc = np.linalg.norm([escpdL[:, 0].sum(), 
                                escpdL[:, 1].sum(), 
                                escpdL[:, 2].sum()])
        print "escpdL.z.sum() = {}".format((escpdL[:, 2]).sum())
        print "norm(escpdL.sum()) = {}".format(L_esc)
    else:
        L_esc = 0

    c_of_disk_mass = [np.asarray(centers[0])[diskInds], 
                      np.asarray(centers[1])[diskInds], 
                      np.asarray(centers[2])[diskInds]]
    v_of_disk_mass = [np.asarray(vels[0])[diskInds], 
                      np.asarray(vels[1])[diskInds], 
                      np.asarray(vels[2])[diskInds]]
    diskL = np.asarray( pps.getL3d(c_of_disk_mass, 
                                   v_of_disk_mass, 
                                   np.asarray(masses)[diskInds],
                                   com) )
    if len(diskL) > 0:
        L_D = np.linalg.norm( [diskL[:, 0].sum(), 
                               diskL[:, 1].sum(), 
                               diskL[:, 2].sum()] )
        print "diskL.z.sum() = {}".format(diskL[:, 2].sum())
        print "norm(diskL.sum()) = {}".format(L_D)
    else:
        L_D = 0
    
    totalL = np.asarray( pps.getL3d(centers, 
                                    [vels[0], 
                                     vels[1],
                                     vels[2]],
                                    np.asarray(masses),
                                    com) )
    L_tot = np.linalg.norm( [totalL[:, 0].sum(), 
                             totalL[:, 1].sum(), 
                             totalL[:, 2].sum()] )
    print "totalL.z.sum() = {}".format(totalL[:, 2].sum())
    print "norm(totalL.sum()) = {}".format(L_tot)

    return L_esc, L_D, L_tot
'''


def findPlanet(dobrDat, ti=0):
    masses = np.asarray(dobrDat.M1[ti]) + np.asarray(dobrDat.M2[ti])
    com, comInds = dobrDat.getCOM3d(ti)
    rads = dobrDat.getRads3d(0, com=com)
    
    # Here we sort the indexes by distance from the center of mass
    # The arrays which are in this order are the following:
    # - mSum (prepended with a 0)
    # - mGrad
    radInds = rads.argsort()
    
    # Calculate the integrated mass and mass gradient:
    # mSum and mGrad, respectively
    mSum = [0]
    mGrad = []
    lastRad = 0
    for j in radInds:
        mSum.append( mSum[-1] + masses[j] )        
        mGrad.append( (mSum[-1] - mSum[-2])/(rads[j] - lastRad) )
        lastRad = rads[j]

    # Make an initial guess of planet radius
    # We use the radius containing half the integrated mass
    mGradMean = (mGrad[-1] - mGrad[0]) / (rads[radInds][-1] - rads[radInds][0])
    R_P_ind = radInds[0]
    mTotal = mSum[-1]
    for j, dmdr in enumerate(mGrad):
        if mSum[j + 1] > 0.5*mTotal:
            R_P_ind = j
            break

    # For each cell , solve for 'massless_a' (the orbital radius the cell would
    # have after relaxation divided by the mass of the planet) using 
    # sqrt(G * M_P * a_eq) = [specific angular momentum]
    # NOTE: we onle store the cell indexes and 'massless_a' for those cells 
    #       with non-zero mass
    massless_a = []
    massless_a_inds = []
    ams = np.zeros(radInds.shape)
    for j in radInds:
        if masses[j] == 0:
            continue
        pos = [dobrDat.centers[ti][0][j] - com[0], 
               dobrDat.centers[ti][1][j] - com[1], 
               dobrDat.centers[ti][2][j] - com[2]]
        vel = [dobrDat.VX[ti][j], 
               dobrDat.VY[ti][j],
               dobrDat.VZ[ti][j]]
        am = np.linalg.norm(np.cross(pos, vel)) # specific angular momentum
        ams[j] = am
        massless_a.append(pow(am, 2) / G) # [orbital radius] * [M_P]
        massless_a_inds.append(j)

    # Iteratively calculate the planet mass and radius by comparing the
    # relaxed radii of each cell to the planet's radius and excluding
    # the cells which fall into the planet and/or are energetically unbound
    print "Begining planet mass and radius calculation"
    diskInds = []
    escpdInds = []
    M_P = mSum[R_P_ind + 1]
    R_P = rads[radInds][R_P_ind]
    while True:
        diskInds = []
        escpdInds = []
        nextM_P = 0
        for j, ma in zip(massless_a_inds, massless_a):
            # If the orbit of the cell is greater than the planets radius or 
            # it is energetically unbound then we exclude the cell from the 
            # next planetary mass calculation
            if np.linalg.norm([dobrDat.VX[ti][j],
                               dobrDat.VY[ti][j], 
                               dobrDat.VZ[ti][j]]) > pow(2*G*M_P/rads[j], 0.5):
                escpdInds.append(j)
            elif rads[j] > R_P and ma > R_P * M_P:
                diskInds.append(j)
            else:
                nextM_P += masses[j]

        print "Last M_P = {}, next M_P = {}".format(M_P, nextM_P)
        
        # If the the change in planet mass is proportionally small, 
        # exit the itteration
        if abs( nextM_P/M_P ) < 1 + pow(2.0, -6):
            M_P = nextM_P
            break

        # recalculate the radius
        M_P = nextM_P
        for j, M in enumerate(mSum[1:]):
            if M > M_P:
                R_P_ind = j - 1
                R_P = rads[radInds][R_P_ind]
                break

    print "Done"

    # Calculate the predicted lunar mass via 
    # Eq 1 of Canup, Barr, and Crawford 2013
    L_esc, L_D, L_tot = dobrDat.getFractionalLs(escpdInds, 
                                                diskInds,
                                                com=com)
    M_esc = masses[escpdInds].sum()
    M_D = masses[diskInds].sum()
    a_R = 2.9*R_P
    M_L = (1.9*(L_D/M_D/pow(G*M_P*a_R, 0.5)) - 1.1 - 1.9*(M_esc/M_D))*M_D
    print "M_L ~ {}".format(M_L)

    return M_P, R_P, diskInds, escpdInds



def plotIntMass(sortedRads, EarthRadInd, mSum, time, saveDir):
    # plot the planet radius and the integrated mass
    fig, ax = plt.subplots()
    plt.semilogy(np.asarray(sortedRads)/1e5, np.asarray(mSum[1:])*1e-3, c='k')
    plt.axvline(x=sortedRads[EarthRadInd]/1e5, ls='--', c='r', label='R_P')
    plt.xlim(0, 1e5)
    plt.ylim(1e21, 8e24)
    plt.legend(loc="lower right")
    plt.title("Integrated mass at t = {:.2f} hrs".format(time/3600))
    plt.xlabel("Radial distance (km)")
    plt.ylabel("Intagrated mass (kg)")
    ax.xaxis.set_major_formatter(fmtr)
    plt.tight_layout()
    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"intM_t{:.2f}.png".format(time/3600))
    plt.close()


def plotSlice(dobrDat, diskInds, escpdInds, R_P, saveDir, ti=0):
    com, comInds = dobrDat.getCOM3d(ti)

    # Get a slice though the xy-plane
    p = [0, 0, 0]
    n = [0, 0, 1]
    slcInds = dobrDat.slice2dFrom3d(p, n, ti)
    
    # Make a pretty plot of the slice
    fig, ax = plt.subplots()

    # first triangulate the location of points in the slice
    x = np.asarray(np.asarray(dobrDat.centers[ti][0])[slcInds])
    y = np.asarray(np.asarray(dobrDat.centers[ti][1])[slcInds])
    triang = mpl.tri.Triangulation(x/1e5, y/1e5)

    # plot density
    z1 = np.asarray(np.asarray(dobrDat.DENS[ti])[slcInds])*1e3
    tpc = ax.tripcolor(triang, z1, shading='flat', vmin=1e-12, vmax=3e4, norm=LogNorm(), cmap=parula_map)
    clb = fig.colorbar(tpc)
    clb.set_label(r"Density (kg/m^3)")

    # This array distinguishes between escaped cells, cells in the disk, and 
    # cells in the planet
    z2 = np.zeros(z1.shape)
    for k, j in enumerate(slcInds):
        if j in diskInds:
            z2[k] = -2
        elif j in escpdInds:
            z2[k] = -1

    # Horizontal hatches -> disk material
    # Vertical hatches -> escaped material
    tcf = ax.tricontourf(triang, z2, 1, hatches=['-----', '|||'], alpha=0.0)

    # Plot the location of the center of mass and the radius of the planet
    circle = plt.Circle((com[0]/1e5, com[1]/1e5), R_P/1e5, color='r', fill=False)
    ax.add_artist(circle)
    ax.scatter(np.asarray(dobrDat.centers[ti][0])[comInds]/1e5, np.asarray(dobrDat.centers[ti][1])[comInds]/1e5, s=11, marker='.', color='r')
    ax.scatter([com[0]/1e5], [com[1]/1e5], s=11, marker='x', color='g')

    ax.set_aspect('equal')
    ax.set_xlim(-8e4, 8e4)
    ax.set_ylim(-8e4, 8e4)
    ax.set_title("Equitorial plane at t = {:.2f} hrs".format(dobrDat.times[ti]/3600))
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.xaxis.set_major_formatter(fmtr)
    ax.yaxis.set_major_formatter(fmtr)
    plt.tight_layout()
    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"slcDens_t{:.2f}.png".format(dobrDat.times[ti]/3600))
    plt.close()



def plotAvels(dobrDat, diskInds, escpdInds, M_P, R_P, saveDir, ti=0):
    com, comInds = dobrDat.getCOM3d(ti)

    # Get a slice though the xy-plane
    p = [0, 0, 0]
    n = [0, 0, 1]
    slcInds = dobrDat.slice2dFrom3d(p, n, ti)

    rads = dobrDat.getRads3d(ti, com=com)[slcInds]
    
    # For each cell , solve for 'massless_a' (the orbital radius the cell would
    # have after relaxation divided by the mass of the planet) using 
    # sqrt(G * M_P * a_eq) = [specific angular momentum]
    massless_a = np.zeros(slcInds.shape)
    ams = np.zeros(slcInds.shape)
    for j, k in enumerate(slcInds):
        if dobrDat.M1[ti][k] == 0 and dobrDat.M2[ti][k] == 0:
            continue
        pos = [dobrDat.centers[ti][0][k] - com[0], 
               dobrDat.centers[ti][1][k] - com[1], 
               dobrDat.centers[ti][2][k] - com[2]]
        vel = [dobrDat.VX[ti][k], dobrDat.VY[ti][k], dobrDat.VZ[ti][k]]
        am = np.linalg.norm(np.cross(pos, vel)) # specific angular momentum
        ams[j] = am
        massless_a[j] = pow(am, 2)/G # [orbital radius] * [M_P]

    # Make an angular velocity scatter plot and over-lay keplerian profile
    avs = np.zeros(slcInds.shape)
    colors = []
    for j, k in enumerate(slcInds):
        # calculate angular velocity
        # avs[k] = G*G*M_P*M_P/pow(ams[j], 3)
        avs[j] = ams[j]/pow(rads[j], 2)
        if k in diskInds:
            colors.append((0, 0, 1, 0.5))
        elif k in escpdInds:
            colors.append((1, 0, 0, 0.5))
        else:
            colors.append((0, 0, 0, 0.5))
        
    fig, ax = plt.subplots()
    plt.scatter(np.asarray(rads)/1e5, avs, s=1, c=colors)
    rs = np.linspace(0, 2e10, 300)
    plt.plot(rs/1e5, pow( G*M_P / pow(rs, 3), 0.5 ), c='b')
    plt.axhline(y=0.000581776417, ls='--', c='r', label="3 hr rotation period")
    plt.ylim(0, 0.0008)
    plt.xlim(0, 1e5)
    plt.legend(loc="upper right")
    plt.title("Angular velocities at t = {:.2f} hrs".format(dobrDat.times[ti]/3600))
    plt.xlabel("Radial distance (km)")
    plt.ylabel("Angular velocity (rad/s)")
    ax.xaxis.set_major_formatter(fmtr)
    ax.yaxis.set_major_formatter(fmtr)
    plt.tight_layout()
    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"angVel_t{:.2f}.png".format(dobrDat.times[ti]/3600))
    plt.close()

    
def plotOrbitRad(dobrDat, diskInds, escpdInds, M_P, R_P, saveDir, ti=0):
    masses = np.asarray(dobrDat.M1[ti]) + np.asarray(dobrDat.M2[ti])
    com, comInds = dobrDat.getCOM3d(ti)
    
    # Get a slice though the xy-plane
    p = [0, 0, 0]
    n = [0, 0, 1]
    slcInds = dobrDat.slice2dFrom3d(p, n, ti)
    
    rads = dobrDat.getRads3d(ti, com=com)[slcInds]
    
    # For each cell , solve for 'massless_a' (the orbital radius the cell would
    # have after relaxation divided by the mass of the planet) using 
    # sqrt(G * M_P * a_eq) = [specific angular momentum]
    massless_a = np.zeros(slcInds.shape)
    ams = np.zeros(slcInds.shape)
    for j, k in enumerate(slcInds):
        if masses[k] == 0:
            continue
        pos = [dobrDat.centers[ti][0][k] - com[0], 
               dobrDat.centers[ti][1][k] - com[1], 
               dobrDat.centers[ti][2][k] - com[2]]
        vel = [dobrDat.VX[ti][k], dobrDat.VY[ti][k], dobrDat.VZ[ti][k]]
        am = np.linalg.norm(np.cross(pos, vel)) # specific angular momentum
        ams[j] = am
        massless_a[j] = pow(am, 2)/G # [orbital radius] * [M_P]

    # Make an angular velocity scatter plot and over-lay keplerian profile
    avs = np.zeros(slcInds.shape)
    colors = []
    for j, k in enumerate(slcInds):
        # calculate angular velocity
        # avs[k] = G*G*M_P*M_P/pow(ams[j], 3)
        avs[j] = ams[j]/pow(rads[j], 2)
        if k in diskInds:
            colors.append((0, 0, 1, 0.5))
        elif k in escpdInds:
            colors.append((1, 0, 0, 0.5))
        else:
            colors.append((0, 0, 0, 0.5))
        
    fig, ax = plt.subplots()
    plt.scatter(np.asarray(rads)/1e5, np.asarray(massless_a)/M_P/1e5, s=1, c=colors, label=None)
    line = plt.axhline(y=R_P/1e5, ls='--', c='r', label="R_P")
    plt.axvline(x=R_P/1e5, ls='--', c='r', label=None)
    plt.ylim(0, 1e5)
    plt.xlim(0, 1e5)
    plt.xlabel("Radial distance (km)")
    plt.ylabel(r"$a_{eq}$ (km)")
    plt.legend(loc="upper center")
    plt.title("Relaxed orbital radii at t = {:.2f} hrs".format(dobrDat.times[ti]/3600))
    ax.xaxis.set_major_formatter(fmtr)
    ax.yaxis.set_major_formatter(fmtr)
    plt.tight_layout()
    fig.savefig(saveDir+"orbRad_t{:.2f}.png".format(dobrDat.times[ti]/3600))
    plt.close()



def plotEnergies(impactDir, impactName, dobrFname, saveDir):
    # get energies from hcth file (histDat)
    histDat = hcr.HcthReader(impactDir+impactName+'/hcth')
    
    Eken = histDat.mcube[0][0] + histDat.mcube[0][1]
    Eint = histDat.mcube[-4][0] + histDat.mcube[-4][1]
    Egra = []
    
    # Integrate the gravitational potential energy from the
    # data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, impactDir+impactName)
    print "Number of dumps to analyze: {}".format(len(cycs))
    dobrTimes = []
    for cyc in cycs:
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, impactDir+impactName)
        print "time of this data dump: {}".format(dobrDat.times[0])

        dobrTimes.append(dobrDat.times[0])
        Egra.append( 1e-7*( np.asarray(dobrDat.SGU)
                            *( np.asarray(dobrDat.M1)
                               + np.asarray(dobrDat.M2) ) ).sum())

    print dobrTimes
        
    Eken_spl = interpolate.splrep(histDat.times, Eken)
    Eint_spl = interpolate.splrep(histDat.times, Eint)
    Egra_spl = interpolate.splrep(dobrTimes, Egra)

    finalT = min((histDat.times[-1], dobrTimes[-1]))
    numTs = min((len(histDat.times), len(dobrTimes)))
    newTs = np.linspace(0, finalT,
                        numTs,
                        endpoint=True)
        
    newEken = interpolate.splev(newTs, Eken_spl)
    newEint = interpolate.splev(newTs, Eint_spl)
    newEgra = interpolate.splev(newTs, Egra_spl)

    normedZeroE = newEgra[-1]

    colors = parula_map(np.linspace(0, 1, 5))

    plt.plot(newTs, newEgra - normedZeroE, label='GPE', c='k')
    plt.fill_between(newTs,
                     newEgra - normedZeroE,
                     newEgra + newEint - normedZeroE,
                     color=colors[1])
    plt.plot(newTs, newEgra + newEint - normedZeroE, label='IE', c='k')
    plt.fill_between(newTs,
                     newEgra + newEint - normedZeroE,
                     newEgra + newEint + newEken - normedZeroE,
                     color=colors[3])
    plt.plot(newTs, newEgra + newEint + newEken - normedZeroE, label='KE', c='k')


    
    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    plt.savefig(saveDir+"energyBudget.png")
