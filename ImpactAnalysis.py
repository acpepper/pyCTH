import matplotlib as mpl
mpl.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from matplotlib.colors import LogNorm
import numpy as np
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys

fmtr = ScalarFormatter()
fmtr.set_powerlimits((2, 2))

import DataOutBinReader as dor
import HcthReader as hcr

from matplotlib.colors import LinearSegmentedColormap



G = 6.674e-8 # CGS

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



'''
def findPlanet(dobrDat, ti=0):
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
    masses = np.asarray(dobrDat.M1[ti]) + np.asarray(dobrDat.M2[ti])
    for j in radInds:
        mSum.append( mSum[-1] + masses[j] )        
        mGrad.append( (mSum[-1] - mSum[-2])/(rads[j] - lastRad) )
        lastRad = rads[j]

    # Make an initial guess of planet radius.
    # We use the radius containing half the integrated mass
    mGradMean = (mGrad[-1] - mGrad[0]) / (rads[radInds][-1] - rads[radInds][0])
    R_P_ind = radInds[0]
    mTotal = mSum[-1]
    for j, dmdr in enumerate(mGrad):
        if mSum[j + 1] > 0.5*mTotal:
            R_P_ind = j
            break

    # For each cell, solve for 'massless_a' (the orbital radius the cell would
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
        massless_a.append(pow(am, 2) / G) # [orbital radius]*M_P
        massless_a_inds.append(j)

    # Iteratively calculate the planet mass and radius by comparing the
    # relaxed radii of each cell to the planet's radius and excluding
    # the cells which fall into the planet and/or are energetically unbound
    print("Begining planet mass and radius calculation")
    diskInds = []
    escpInds = []
    M_P = mSum[R_P_ind + 1]
    R_P = rads[radInds][R_P_ind]
    while True:
        diskInds = []
        escpInds = []
        nextM_P = 0
        for j, ma in zip(massless_a_inds, massless_a):
            # If the orbit of the cell is greater than the planets radius or 
            # it is energetically unbound then we exclude the cell from the 
            # next planetary mass calculation
            if np.linalg.norm([dobrDat.VX[ti][j],
                               dobrDat.VY[ti][j], 
                               dobrDat.VZ[ti][j]]) > pow(2*G*M_P/rads[j], 0.5):
                escpInds.append(j)
            elif rads[j] > R_P and ma > R_P * M_P:
                diskInds.append(j)
            else:
                nextM_P += masses[j]

        print("Last M_P = {}, next M_P = {}".format(M_P, nextM_P))
        
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

    print("Done")

    # Calculate the predicted lunar mass via 
    # Eq 1 of Canup, Barr, and Crawford 2013
    L_esc, L_D, L_tot = dobrDat.getFractionalLs(escpInds, 
                                                diskInds,
                                                com=com)
    M_esc = masses[escpInds].sum()
    M_D = masses[diskInds].sum()
    a_R = 2.9*R_P
    M_L = (1.9*(L_D/M_D/pow(G*M_P*a_R, 0.5)) - 1.1 - 1.9*(M_esc/M_D))*M_D
    print("M_L ~ {}".format(M_L))

    return M_P, R_P, diskInds, escpInds
'''


def plotIntMass(sortedRads, EarthRadInd, mSum, time, saveDir):
    # plot the planet radius and the integrated mass
    fig, ax = plt.subplots()
    plt.semilogy(np.asarray(sortedRads)/1e5, np.asarray(mSum[1:])*1e-3, c='k')
    plt.axvline(x=sortedRads[EarthRadInd]/1e5, ls='--', c='r', label='R_P')
    plt.xlim(0, 1e5)
    plt.ylim(1e21, 8e24)
    fig.legend(loc="lower right")
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


def plotSlice(dobrDat, diskInds, escpInds, R_P, saveDir, scal=("DENS", 1e3), ti=0, **kwargs):
    # get the center of mass
    try:
        com = kwargs["com"]
    except KeyError:
        com = dobrDat.getCOM3d(ti)[0]
    # Get a slice though the xy-plane
    try:
        slcInds = kwargs["slcInds"]
    except KeyError:
        p = [0, 0, 0]
        n = [0, 0, 1]
        slcInds = dobrDat.slice2dFrom3d(p, n, ti)
    
    # Make a pretty plot of the slice
    fig, ax = plt.subplots()

    # first triangulate the location of points in the slice
    x = np.asarray(dobrDat.centers[ti][0])[slcInds]
    y = np.asarray(dobrDat.centers[ti][1])[slcInds]
    triang = mpl.tri.Triangulation(x/1e5, y/1e5)

    # plot scalar field
    z1 = np.asarray(getattr(dobrDat, scal[0])[ti])[slcInds]*scal[1]
    if scal[0] == "DENS":
        tpc = ax.tripcolor(triang, z1,
                           shading='flat',
                           vmin=1e1,
                           vmax=1e4,
                           norm=LogNorm(),
                           cmap=parula_map)
        clb = fig.colorbar(tpc)
        clb.set_label(r"Density (kg/m^3)")
    elif scal[0] == "P":
        tpc = ax.tripcolor(triang, z1,
                           shading='flat',
                           vmin=1e8,
                           vmax=2e11,
                           norm=LogNorm(),
                           cmap=parula_map)
        clb = fig.colorbar(tpc)
        clb.set_label(r"Pressure (kg/m/s^2)")
    elif scal[0] == "T":
        tpc = ax.tripcolor(triang, z1,
                           shading='flat',
                           vmin=1e2,
                           vmax=1e4,
                           norm=LogNorm(),
                           cmap=parula_map)
        clb = fig.colorbar(tpc)
        clb.set_label(r"Temperature (K)")

    # This array identifies disk cells
    z2 = np.zeros(z1.shape)
    for k, j in enumerate(slcInds):
        if j in diskInds:
            z2[k] = -1
    # Horizontal hatches -> disk material
    hcf1 = ax.tricontourf(triang, z2, 0, hatches=['--'], alpha=0.0)
    
    # This array identifies escaped cells
    z3 = np.zeros(z1.shape)
    for k, j in enumerate(slcInds):
        if j in escpInds:
            z3[k] = -1
    # Vertical hatches -> escaped material
    hcf2 = ax.tricontourf(triang, z3, 0, hatches=['||'], alpha=0.0)
        
    # Plot the location of the center of mass and the radius of the planet
    circ = plt.Circle((com[0]/1e5, com[1]/1e5), R_P/1e5, color='g', fill=False)
    ax.add_artist(circ)
    cntr = ax.scatter([com[0]/1e5], [com[1]/1e5], s=11, marker='x', color='g')

    ax.set_aspect('equal')
    ax.set_xlim(com[0]/1e5 - 8e4, com[1]/1e5 + 8e4)
    ax.set_ylim(com[0]/1e5 - 8e4, com[1]/1e5 + 8e4)
    ax.set_title("Equatorial plane at t = {:.2f} hrs".format(dobrDat.times[ti]/3600))
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.xaxis.set_major_formatter(fmtr)
    ax.yaxis.set_major_formatter(fmtr)

    # make proxy artists for legend entries
    diskLgnd = mpatches.FancyBboxPatch((0, 0), 1, 1, fc='none', hatch='--')
    escpLgnd = mpatches.FancyBboxPatch((0, 0), 1, 1, fc='none', hatch='||')
    circLgnd = plt.scatter([], [], facecolors='none', edgecolors='g')
    fig.legend((diskLgnd, escpLgnd, cntr, circLgnd), ("Disk material", "Escaped material", "Center of Mass", "Radius"), loc="upper right")    
    
    plt.tight_layout()

    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"slc{}_t{:.2f}.png".format(scal[0], dobrDat.times[ti]/3600))
    plt.close()



def plotSliceMixing(dobrDat, saveDir, ti=0, **kwargs):
    # get the center of mass
    try:
        com = kwargs["com"]
    except KeyError:
        com = dobrDat.getCOM3d(ti)[0]
    # Get a slice though the xy-plane
    try:
        slcInds = kwargs["slcInds"]
    except KeyError:
        p = [0, 0, 0]
        n = [0, 0, 1]
        slcInds = dobrDat.slice2dFrom3d(p, n, ti)
    
    fig, axs = plt.subplots(1, 2, sharey=True)

    # first triangulate the location of cells in the slice
    x = np.asarray(dobrDat.centers[ti][0])[slcInds]
    y = np.asarray(dobrDat.centers[ti][1])[slcInds]
    triang = mpl.tri.Triangulation(x/1e5, y/1e5)
    
    # for each point in the slice, find the mixing of material 1
    z1 = np.asarray(dobrDat.M1ID[ti])[slcInds]
    tpc1 = axs[0].tripcolor(triang, z1, shading='flat', vmin=0, vmax=1, cmap=parula_map)

    axs[0].set_aspect('equal')
    axs[0].set_xlim(com[0]/1e5 - 8e4, com[1]/1e5 + 8e4)
    axs[0].set_ylim(com[0]/1e5 - 8e4, com[1]/1e5 + 8e4)
    axs[0].set_xlabel("X (km)")
    axs[0].set_ylabel("Y (km)")
    axs[0].xaxis.set_major_formatter(fmtr)
    axs[0].yaxis.set_major_formatter(fmtr)
    axs[0].set_title("Mantle")

    # for each point in the slice, find the mixing of material 2    
    z2 = np.asarray(dobrDat.M2ID[ti])[slcInds]
    tpc2 = axs[1].tripcolor(triang, z2, shading='flat', vmin=0, vmax=1, cmap=parula_map)
    
    # The color bar is the same for both materials, the choice to reference it
    # to 'tpc2' is arbitrary
    clb = fig.colorbar(tpc2,
                       ax=axs,
                       orientation="horizontal",
                       fraction=0.1,
                       anchor=(0.0, 3.0),
                       panchor=(0.0, 1.0),
                       label="Percentage of Impactor Material")

    axs[1].set_aspect('equal')
    axs[1].set_xlim(com[0]/1e5 - 8e4, com[1]/1e5 + 8e4)
    axs[1].set_ylim(com[0]/1e5 - 8e4, com[1]/1e5 + 8e4)
    axs[1].set_xlabel("X (km)")
    axs[1].xaxis.set_major_formatter(fmtr)
    axs[1].yaxis.set_major_formatter(fmtr)
    axs[1].set_title("Core")

    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"slcMixing_t{:.2f}.png".format(dobrDat.times[ti]/3600),
                bbox_inches='tight')
    plt.close()
    


def plotProfMixing(dobrDat, saveDir, ti=0, **kwargs):
    # get the center of mass
    try:
        com = kwargs["com"]
    except KeyError:
        com = dobrDat.getCOM3d(ti)[0]
        
    # arange the cells by density
    densInds = np.asarray(dobrDat.DENS[ti]).argsort()
    masses = np.asarray(dobrDat.M1[ti]) + np.asarray(dobrDat.M2[ti])
    maxMass = max(masses)
    
    m1MixInds = []
    m1Sizes = []
    m1Colors = []
    m2MixInds = []
    m2Sizes = []
    m2Colors = []
    for i in densInds:
        # If cell is all core don't append to 'm1MixInds'
        if dobrDat.VOLM1[ti][i] < 2048*sys.float_info.epsilon:
            m2MixInds.append(i)
            m2Sizes.append(200*dobrDat.M2[ti][i]/maxMass)
            m2Colors.append(dobrDat.TM2[ti][i]/8.6173e-5)
            continue
        # If cell is all mantle don't append to 'm2MixInds'
        if dobrDat.VOLM2[ti][i] < 2048*sys.float_info.epsilon:
            m1MixInds.append(i)
            m1Sizes.append(200*dobrDat.M1[ti][i]/maxMass)
            m1Colors.append(dobrDat.TM1[ti][i]/8.6173e-5)
            continue
        m1MixInds.append(i)
        m1Sizes.append(200*dobrDat.M1[ti][i]/maxMass)
        m1Colors.append(dobrDat.TM1[ti][i]/8.6173e-5)
        m2MixInds.append(i)
        m2Sizes.append(200*dobrDat.M2[ti][i]/maxMass)
        m2Colors.append(dobrDat.TM2[ti][i]/8.6173e-5)
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    sctr1 = axs[0].scatter(np.asarray(dobrDat.DENS[ti])[m1MixInds],
                           np.asarray(dobrDat.M1ID[ti])[m1MixInds] - 1,
                           label="Mantle",
                           s=m1Sizes,
                           c=m1Colors,
                           cmap=parula_map,
                           vmin=3e2,
                           vmax=1e4,
                           alpha=0.3)
    sctr2 = axs[1].scatter(np.asarray(dobrDat.DENS[ti])[m2MixInds],
                           np.asarray(dobrDat.M2ID[ti])[m2MixInds] - 1,
                           label="Core",
                           s=m2Sizes,
                           c=m2Colors,
                           cmap=parula_map,
                           vmin=3e2,
                           vmax=1e4,
                           alpha=0.3)
    clb = plt.colorbar(sctr2,
                       ax=axs,
                       orientation="horizontal",
                       fraction=0.1,
                       anchor=(0.0, 3.0),
                       panchor=(0.0, 1.0),
                       label="Temperature (K)")
    clb.set_alpha(1)
    clb.draw_all()

    axs[0].set_title("Mantle Mixing")
    axs[1].set_title("Core Mixing")
    
    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"prfMixing_t{:.2f}.png".format(dobrDat.times[ti]/3600),
                bbox_inches='tight')
    plt.close()



def plotEnergyScatter(dobrDat, diskInds, escpInds, M_P, R_P, saveDir, ti=0, **kwargs):
    try:
        com = kwargs["com"]
    except KeyError:
        com = dobrDat.getCOM3d(ti)[0]
        
    # Get a slice though the xy-plane
    try:
        slcInds = kwargs["slcInds"]
    except KeyError:
        p = [0, 0, 0]
        n = [0, 0, 1]
        slcInds = dobrDat.slice2dFrom3d(p, n, ti)

    masses = (np.asarray(dobrDat.M1[ti]) + np.asarray(dobrDat.M2[ti]))[slcInds]
    rads = dobrDat.getRads3d(ti, com=com)[slcInds]
    radInds = rads.argsort()

    KEs = np.asarray(dobrDat.KE[ti])[slcInds]*1e-4
    GUs = np.asarray(dobrDat.SGU[ti])[slcInds]*1e-4
    Sum = KEs + GUs

    fig, ax = plt.subplots()
    smSctr = plt.scatter(rads/1e5, Sum, s=1)
    guSctr = plt.scatter(rads/1e5, GUs, s=1)
    keSctr = plt.scatter(rads/1e5, KEs, s=1)

    Rs = np.linspace(R_P, 1e10, 200)
    E_escp = G*M_P/Rs
    escpLn = plt.plot(Rs*1e-5, E_escp*1e-4, c='r', ls='--', label='Escaped')

    ax.set_xlim(0, 1e5)
    ax.set_ylim(-5e7, 5e7)
    ax.axvline(x=R_P/1e5, ls='--', label="R_P")

    # make proxy artist for legend entries
    handles, labels = ax.get_legend_handles_labels()
    fig.legend((keSctr, guSctr, smSctr, handles[0]), ('Kenetic', 'Gavitational', 'Total', labels[0]), loc="upper right")
    
    plt.title("Cell Energies at t = {:.2f} hrs".format(dobrDat.times[ti]/3600))
    plt.xlabel("Radial distance (km)")
    plt.ylabel("Specific Energy (J/kg)")
    ax.xaxis.set_major_formatter(fmtr)
    ax.yaxis.set_major_formatter(fmtr)
    plt.tight_layout()
    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"energyScatter_t{:.2f}.png".format(dobrDat.times[ti]/3600))
    plt.close()



def plotAvels(dobrDat, diskInds, escpInds, M_P, R_P, saveDir, ti=0, **kwargs):
    # get center of mass
    try:
        com = kwargs["com"]
    except KeyError:
        com = dobrDat.getCOM3d(ti)[0]
        
    # Get a slice though the xy-plane
    try:
        slcInds = kwargs["slcInds"]
    except KeyError:
        p = [0, 0, 0]
        n = [0, 0, 1]
        slcInds = dobrDat.slice2dFrom3d(p, n, ti)

    rads = dobrDat.getRads3d(ti, com=com)[slcInds]
    masses = (np.asarray(dobrDat.M1[ti]) + np.asarray(dobrDat.M2[ti]))[slcInds]
    
    # For each cell calculate the specific angular momentum, 'am'
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

    # Make an angular velocity scatter plot and over-lay keplerian profile
    avs = np.zeros(slcInds.shape)
    colors = []
    for j, k in enumerate(slcInds):
        # calculate angular velocity
        # avs[k] = G*G*M_P*M_P/pow(ams[j], 3)
        avs[j] = ams[j]/pow(rads[j], 2)
        if k in diskInds:
            colors.append((0, 0, 1, 0.5))
        elif k in escpInds:
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
    fig.legend(loc="upper right")
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


    
'''
def plotOrbitRad(dobrDat, diskInds, escpInds, M_P, R_P, saveDir, ti=0):
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
        elif k in escpInds:
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
'''



def plotEnergyTotal_dyn(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the energy from the data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    erthEken = np.zeros(numCycs)
    diskEken = np.zeros(numCycs)
    escpEken = np.zeros(numCycs)
    erthEint = np.zeros(numCycs)
    diskEint = np.zeros(numCycs)
    escpEint = np.zeros(numCycs)
    erthEgra = np.zeros(numCycs)
    diskEgra = np.zeros(numCycs)
    escpEgra = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600
        M_P, R_P, erthInds, diskInds, escpInds = dobrDat.findPlanet(0)

        erthEken[i] = 1e-7*( np.asarray(dobrDat.KE[0])[erthInds]
                             *( np.asarray(dobrDat.M1[0])[erthInds]
                                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskEken[i] = 1e-7*( np.asarray(dobrDat.KE[0])[diskInds]
                             *( np.asarray(dobrDat.M1[0])[diskInds]
                                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpEken[i] = 1e-7*( np.asarray(dobrDat.KE[0])[escpInds]
                             *( np.asarray(dobrDat.M1[0])[escpInds]
                                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()
        
        erthEint[i] = 1e-7*( np.asarray(dobrDat.IE[0])[erthInds]
                             *( np.asarray(dobrDat.M1[0])[erthInds]
                                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskEint[i] = 1e-7*( np.asarray(dobrDat.IE[0])[diskInds]
                             *( np.asarray(dobrDat.M1[0])[diskInds]
                                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpEint[i] = 1e-7*( np.asarray(dobrDat.IE[0])[escpInds]
                             *( np.asarray(dobrDat.M1[0])[escpInds]
                                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()
        
        erthEgra[i] = 1e-7*( np.asarray(dobrDat.SGU[0])[erthInds]
                             *( np.asarray(dobrDat.M1[0])[erthInds]
                                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskEgra[i] = 1e-7*( np.asarray(dobrDat.SGU[0])[diskInds]
                             *( np.asarray(dobrDat.M1[0])[diskInds]
                                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpEgra[i] = 1e-7*( np.asarray(dobrDat.SGU[0])[escpInds]
                             *( np.asarray(dobrDat.M1[0])[escpInds]
                                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()
        
    colors = parula_map(np.linspace(0, 1, 7))

    '''
    dobrTs = dobrTs[:2]
    erthEken = erthEken[:2]
    diskEken = diskEken[:2]
    escpEken = escpEken[:2]
    erthEint = erthEint[:2]
    diskEint = diskEint[:2]
    escpEint = escpEint[:2]
    erthEgra = erthEgra[:2]
    diskEgra = diskEgra[:2]
    escpEgra = escpEgra[:2]
    '''
    normedZeroEs = ( (erthEgra + diskEgra + escpEgra).min()
                     * np.ones(len(dobrTs)) )
    Emax = ( ( erthEgra + diskEgra + escpEgra )
             + ( erthEken + diskEken + escpEken)
             + ( erthEint + diskEint + escpEint ) ).max()

    fig, ax_E = plt.subplots()
    

    # plot energy minimum and gravitational potential energies
    ax_E.fill_between(dobrTs,
                      normedZeroEs,
                      erthEgra + diskEgra + escpEgra,
                      color=colors[0])
    ax_E.fill_between(dobrTs,
                      erthEgra + diskEgra + escpEgra,
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken ),
                      color=colors[1])
    # plot the boundaries in black
    ax_E.plot(dobrTs, normedZeroEs, c='k', lw=1)
    ax_E.plot(dobrTs, erthEgra + diskEgra + escpEgra, c='k', lw=1)


    # plot kinetic energies
    ax_E.fill_between(dobrTs,
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken ),
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken),
                      color=colors[2])
    ax_E.fill_between(dobrTs,
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken),
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken + escpEken),
                      color=colors[3])
    ax_E.fill_between(dobrTs,
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken + escpEken),
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken + escpEken)
                      + ( erthEint ),
                      color=colors[4])
    # plot the boundaries in black
    ax_E.plot(dobrTs,
              ( erthEgra + diskEgra + escpEgra )
              + ( erthEken ),
              c='k',
              lw=1)
    ax_E.plot(dobrTs,
              ( erthEgra + diskEgra + escpEgra )
              + ( erthEken + diskEken),
              c='k',
              lw=1)
    ax_E.plot(dobrTs,
              ( erthEgra + diskEgra + escpEgra )
              + ( erthEken + diskEken + escpEken),
              c='k',
              lw=1)


    # plot internal energies
    ax_E.fill_between(dobrTs,
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken + escpEken)
                      + ( erthEint ),
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken + escpEken)
                      + ( erthEint + diskEint ),
                      color=colors[5])
    ax_E.fill_between(dobrTs,
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken + escpEken)
                      + ( erthEint + diskEint ),
                      ( erthEgra + diskEgra + escpEgra )
                      + ( erthEken + diskEken + escpEken)
                      + ( erthEint + diskEint + escpEint ),
                      color=colors[6])
    # plot the boundaries in black
    ax_E.plot(dobrTs,
              ( erthEgra + diskEgra + escpEgra )
              + ( erthEken + diskEken + escpEken)
              + ( erthEint ),
              c='k',
              lw=1)
    ax_E.plot(dobrTs,
              ( erthEgra + diskEgra + escpEgra )
              + ( erthEken + diskEken + escpEken)
              + ( erthEint + diskEint ),
              c='k',
              lw=1)
    ax_E.plot(dobrTs,
              ( erthEgra + diskEgra + escpEgra )
              + ( erthEken + diskEken + escpEken)
              + ( erthEint + diskEint + escpEint ),
              c='k',
              lw=1)

    
    # make second y axis to display percentage
    ax_perc = ax_E.twinx()
    ax_perc.set_ylim(0, 1)
    
    ax_E.set_ylim(normedZeroEs[0], Emax)
    ax_E.set_xlim(dobrTs[0], dobrTs[-1])

    ax_E.set_xlabel("Time (hr)")
    ax_E.set_ylabel("Energy (J)")
    ax_perc.set_ylabel("Fraction of maximum energy")

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes,
               ("GU",
                "Earth KE", "Disk KE", "Escaped KE",
                "Earth IE", "Disk IE", "Escaped IE"))
    
    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    plt.savefig(saveDir+"energyTotal_dyn.png")

    return dobrTs, (erthEgra, erthEken, erthEint), (diskEgra, diskEken, diskEint), (escpEgra, escpEken, escpEint)



def plotEnergyTotal_mat(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the energy from the data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    EkenMat1 = np.zeros(numCycs)
    EkenMat2 = np.zeros(numCycs)
    EintMat1 = np.zeros(numCycs)
    EintMat2 = np.zeros(numCycs)
    EgraMat1 = np.zeros(numCycs)
    EgraMat2 = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600

        EkenMat1[i] = 1e-7*( np.asarray(dobrDat.KE[0])
                             * np.asarray(dobrDat.M1[0]) ).sum()
        EkenMat2[i] = 1e-7*( np.asarray(dobrDat.KE[0])
                             * np.asarray(dobrDat.M2[0]) ).sum()

        EintMat1[i] = 1e-7*( np.asarray(dobrDat.IE[0])
                             * np.asarray(dobrDat.M1[0]) ).sum()
        EintMat2[i] = 1e-7*( np.asarray(dobrDat.IE[0])
                             * np.asarray(dobrDat.M2[0]) ).sum()

        EgraMat1[i] = 1e-7*( np.asarray(dobrDat.SGU[0])
                             * np.asarray(dobrDat.M1[0]) ).sum()
        EgraMat2[i] = 1e-7*( np.asarray(dobrDat.SGU[0])
                             * np.asarray(dobrDat.M2[0]) ).sum()
        
    colors = parula_map(np.linspace(0, 1, 5))

    '''
    dobrTs = dobrTs[:2]
    EkenMat1 = EkenMat1[:2]
    EkenMat2 = EkenMat2[:2]
    EintMat1 = EintMat1[:2]
    EintMat2 = EintMat2[:2]
    EgraMat1 = EgraMat1[:2]
    EgraMat2 = EgraMat2[:2]
    '''
    normedZeroEs = ( (EgraMat1 + EgraMat2).min()
                     *np.ones(len(dobrTs)) )
    Emax = ( ( EgraMat1 + EgraMat2 )
             + ( EkenMat1 + EkenMat2 )
             + ( EintMat1 + EintMat2 ) ).max()

    fig, ax_E = plt.subplots()
    
    # plot energy minimum and gravitational potential energies
    ax_E.fill_between(dobrTs,
                      normedZeroEs,
                      EgraMat1 + EgraMat2,
                      color=colors[0])
    # plot the boundaries in black
    ax_E.plot(dobrTs, normedZeroEs, c='k', lw=1)
    ax_E.plot(dobrTs, EgraMat1 + EgraMat2, c='k', lw=1)

    
    # plot kinetic energies
    ax_E.fill_between(dobrTs,
                      (   EgraMat1 + EgraMat2 ),
                      (   EgraMat1 + EgraMat2 )
                      + ( EkenMat1 ),
                      color=colors[1])
    ax_E.fill_between(dobrTs,
                      (   EgraMat1 + EgraMat2 )
                      + ( EkenMat1 ),
                      (   EgraMat1 + EgraMat2 )
                      + ( EkenMat1 + EkenMat2 ),
                     color=colors[2])
    # plot the boundaries in black
    ax_E.plot(dobrTs,
              (   EgraMat1 + EgraMat2 )
              + ( EkenMat1 ),
              c='k',
              lw=1)
    ax_E.plot(dobrTs,
              (   EgraMat1 + EgraMat2 )
              + ( EkenMat1 + EkenMat2 ),
              c='k',
              lw=1)

    # plot internal energies
    ax_E.fill_between(dobrTs,
                      (   EgraMat1 + EgraMat2 )
                      + ( EkenMat1 + EkenMat2 ),
                      (   EgraMat1 + EgraMat2 )
                      + ( EkenMat1 + EkenMat2 )
                      + ( EintMat1 ),
                     color=colors[3])
    ax_E.fill_between(dobrTs,
                      (   EgraMat1 + EgraMat2 )
                      + ( EkenMat1 + EkenMat2 )
                      + ( EintMat1 ),
                      (   EgraMat1 + EgraMat2 )
                      + ( EkenMat1 + EkenMat2 )
                      + ( EintMat1 + EintMat2 ),
                     color=colors[4])
    # plot the boundaries in black
    ax_E.plot(dobrTs,
              (   EgraMat1 + EgraMat2 )
              + ( EkenMat1 + EkenMat2 )
              + ( EintMat1 ),
              c='k',
              lw=1)
    ax_E.plot(dobrTs,
              (   EgraMat1 + EgraMat2 )
              + ( EkenMat1 + EkenMat2 )
              + ( EintMat1 + EintMat2 ),
              c='k',
              lw=1)

    
    # make second y axis to display percentage
    ax_perc = ax_E.twinx()
    ax_perc.set_ylim(0, 1)
    
    ax_E.set_ylim(normedZeroEs[0], Emax)
    ax_E.set_xlim(dobrTs[0], dobrTs[-1])

    ax_E.set_xlabel("Time (hr)")
    ax_E.set_ylabel("Energy (J)")
    ax_perc.set_ylabel("Fraction of maximum energy")
    
    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes,
               ("GU",
                "Mantle KE", "Core KE",
                "Mantle IE", "Core IE"))
    
    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    plt.savefig(saveDir+"energyTotal_mat.png")

    return dobrTs, (EgraMat1, EgraMat2), (EkenMat1, EkenMat2), (EintMat1, EintMat2)



def plotEnergy_dyn(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the energy from the data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    erthEken = np.zeros(numCycs)
    diskEken = np.zeros(numCycs)
    escpEken = np.zeros(numCycs)
    erthEint = np.zeros(numCycs)
    diskEint = np.zeros(numCycs)
    escpEint = np.zeros(numCycs)
    erthEgra = np.zeros(numCycs)
    diskEgra = np.zeros(numCycs)
    escpEgra = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600
        M_P, R_P, erthInds, diskInds, escpInds = dobrDat.findPlanet(0)

        erthEken[i] = 1e-7*( np.asarray(dobrDat.KE[0])[erthInds]
                             *( np.asarray(dobrDat.M1[0])[erthInds]
                                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskEken[i] = 1e-7*( np.asarray(dobrDat.KE[0])[diskInds]
                             *( np.asarray(dobrDat.M1[0])[diskInds]
                                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpEken[i] = 1e-7*( np.asarray(dobrDat.KE[0])[escpInds]
                             *( np.asarray(dobrDat.M1[0])[escpInds]
                                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()
        
        erthEint[i] = 1e-7*( np.asarray(dobrDat.IE[0])[erthInds]
                             *( np.asarray(dobrDat.M1[0])[erthInds]
                                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskEint[i] = 1e-7*( np.asarray(dobrDat.IE[0])[diskInds]
                             *( np.asarray(dobrDat.M1[0])[diskInds]
                                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpEint[i] = 1e-7*( np.asarray(dobrDat.IE[0])[escpInds]
                             *( np.asarray(dobrDat.M1[0])[escpInds]
                                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()
        
        erthEgra[i] = 1e-7*( np.asarray(dobrDat.SGU[0])[erthInds]
                             *( np.asarray(dobrDat.M1[0])[erthInds]
                                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskEgra[i] = 1e-7*( np.asarray(dobrDat.SGU[0])[diskInds]
                             *( np.asarray(dobrDat.M1[0])[diskInds]
                                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpEgra[i] = 1e-7*( np.asarray(dobrDat.SGU[0])[escpInds]
                             *( np.asarray(dobrDat.M1[0])[escpInds]
                                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()
        
    colors = parula_map(np.linspace(0, 1, 7))

    '''
    dobrTs = dobrTs[:4]
    erthEken = erthEken[:4]
    diskEken = diskEken[:4]
    escpEken = escpEken[:4]
    erthEint = erthEint[:4]
    diskEint = diskEint[:4]
    escpEint = escpEint[:4]
    erthEgra = erthEgra[:4]
    diskEgra = diskEgra[:4]
    escpEgra = escpEgra[:4]
    '''
    normedZeroEs = ( (erthEgra + diskEgra + escpEgra).max()
                     *np.ones(len(dobrTs)) )
    Etot = ( erthEken + diskEken + escpEken - normedZeroEs
             + erthEint + diskEint + escpEint
             + erthEgra + diskEgra + escpEgra )
    
    fig, ax_E = plt.subplots()

    # Gravitational potential energy
    ax_E.plot(dobrTs,
              abs( ( erthEgra + diskEgra + escpEgra - normedZeroEs )
                   / Etot),
              c=colors[0])

    # Kinetic energies
    ax_E.plot(dobrTs, abs(erthEken/Etot), c=colors[1])
    ax_E.plot(dobrTs, abs(diskEken/Etot), c=colors[2])
    ax_E.plot(dobrTs, abs(escpEken/Etot), c=colors[3])

    # Internal energies
    ax_E.plot(dobrTs, abs(erthEint/Etot), c=colors[4])
    ax_E.plot(dobrTs, abs(diskEint/Etot), c=colors[5])
    ax_E.plot(dobrTs, abs(escpEint/Etot), c=colors[6])

    # ax_E.set_ylim([normedZeroEs[0], Emax])
    ax_E.set_xlim([dobrTs[0], dobrTs[-1]])

    ax_E.set_xlabel("Time (hr)")
    ax_E.set_ylabel("Fraction of total energy")

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes,
               ("GU",
                "Earth KE", "Disk KE", "Escaped KE",
                "Earth IE", "Disk IE", "Escaped IE"))
    
    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    plt.savefig(saveDir+"energy_dyn.png")

    return dobrTs, (erthEgra, erthEken, erthEint), (diskEgra, diskEken, diskEint), (escpEgra, escpEken, escpEint)



def plotEnergy_mat(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the energy from the data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    EkenMat1 = np.zeros(numCycs)
    EkenMat2 = np.zeros(numCycs)
    EintMat1 = np.zeros(numCycs)
    EintMat2 = np.zeros(numCycs)
    EgraMat1 = np.zeros(numCycs)
    EgraMat2 = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600

        EkenMat1[i] = 1e-7*( np.asarray(dobrDat.KE[0])
                             * np.asarray(dobrDat.M1[0]) ).sum()
        EkenMat2[i] = 1e-7*( np.asarray(dobrDat.KE[0])
                             * np.asarray(dobrDat.M2[0]) ).sum()

        EintMat1[i] = 1e-7*( np.asarray(dobrDat.IE[0])
                             * np.asarray(dobrDat.M1[0]) ).sum()
        EintMat2[i] = 1e-7*( np.asarray(dobrDat.IE[0])
                             * np.asarray(dobrDat.M2[0]) ).sum()

        EgraMat1[i] = 1e-7*( np.asarray(dobrDat.SGU[0])
                             * np.asarray(dobrDat.M1[0]) ).sum()
        EgraMat2[i] = 1e-7*( np.asarray(dobrDat.SGU[0])
                             * np.asarray(dobrDat.M2[0]) ).sum()
        
    colors = parula_map(np.linspace(0, 1, 5))
    '''
    dobrTs = dobrTs[:2]
    EkenMat1 = EkenMat1[:2]
    EkenMat2 = EkenMat2[:2]
    EintMat1 = EintMat1[:2]
    EintMat2 = EintMat2[:2]
    EgraMat1 = EgraMat1[:2]
    EgraMat2 = EgraMat2[:2]
    '''
    normedZeroEs = ( (EgraMat1 + EgraMat2).min()
                     *np.ones(len(dobrTs)) )
    Etot = ( ( EgraMat1 + EgraMat2  - normedZeroEs )
             + ( EkenMat1 + EkenMat2 )
             + ( EintMat1 + EintMat2 ) ).max()
    
    fig, ax_E = plt.subplots()

    # plot the fractional energy
    # gravitational potential energy
    ax_E.plot(dobrTs,
              abs((EgraMat1 + EgraMat2 - normedZeroEs)/Etot),
              c=colors[0])

    # kinetic energy
    ax_E.plot(dobrTs, abs(EkenMat1/Etot), c=colors[1])
    ax_E.plot(dobrTs, abs(EkenMat2/Etot), c=colors[2])

    # Internal energies
    ax_E.plot(dobrTs, abs(EintMat1/Etot), c=colors[3])
    ax_E.plot(dobrTs, abs(EintMat2/Etot), c=colors[4])
    
    ax_E.set_xlim([dobrTs[0], dobrTs[-1]])

    ax_E.set_xlabel("Time (hr)")
    ax_E.set_ylabel("Fraction of total energy")

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes,
               ("GU",
                "Mantle KE", "Core KE",
                "Mantle IE", "Core IE"))
    
    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    plt.savefig(saveDir+"energy_mat.png")

    return dobrTs, (EgraMat1, EgraMat2), (EkenMat1, EkenMat2), (EintMat1, EintMat2)



def plotEnergy(dataDir, saveDir, dobrFname="binDat", **kwargs):
    # determine which categories we're dividing the energy into
    # mat = material
    # dyn = dynamical (e.g. disk, escp, ... etc.)
    # all = both material and dynamical
    try:
        divCat = kwargs["divCat"]
        if divCat not in ["mat", "dyn", "all"]:
            raise RuntimeError("'{}' not valid value for divCat".format(divCat))
    except KeyError:
        divCat = "custom"

    # determine which value to use for Etot
    # init = normalize to total initial energy
    # gmin = normalize to the total energy when GU is minimized
    # adpt = normalize to the total energy at each data dump
    try:
        EtotVal = kwargs
        if EtotVal not in ["init", "gmin", "adpt"]:
            raise RuntimeError("'{}' not valid value for EtotVal".format(EtotVal))
    except KeyError:
        EtotVal = "adpt"

    # determine which energy values we'll be using;
    # to accomplish this we will reference the following nested list:
    compList =  [ [ ["erthEkenMat1", "erthEkenMat2"],
                    ["diskEkenMat1", "diskEkenMat2"],
                    ["escpEkenMat1", "escpEkenMat2"] ],
                  [ ["erthEintMat1", "erthEintMat2"],
                    ["diskEintMat1", "diskEintMat2"],
                    ["escpEintMat1", "escpEintMat2"] ],
                  [ ["erthEkenMat1", "erthEkenMat2"],
                    ["diskEkenMat1", "diskEkenMat2"],
                    ["escpEkenMat1", "escpEkenMat2"] ] ]

    # the user may provide a tuple of Boolean values which describes which
    # of the categories above will be included. For example, the following
    # tuple:
    #
    # (True, (True, (True, True), False), True) 
    #
    # indicates that
    # - the total kinetic energy will be plotted as one line
    # - the internal energy of the earth will be plotted as one line
    # - the internal energy of Mat 1 in the disk will be plotted as one line
    # - the internal energy of Mat 2 in the disk will be plotted as one line
    # - the internal energy of escaped material will not be plotted at all
    # - the gravitational potential energy will be plotted as one line
    try:
        if divCat == "custom":
            Ecomps = kwargs["Ecomps"]
        else:
            raise RuntimeError("'divCat' and 'Ecomps' cannot be assigned at the same time")
    except KeyError:
        Ecomps = ( ( (True, True), (True, True), (True, True) ),
                   ( (True, True), (True, True), (True, True) ),
                   ( True ) )

    # gather the energy components that we'll be using
    
    if Ecomps:
        print("One line for {}".format(compList))
    for i, eType in enumerate(Ecomps):
        if eType:
            print("One line for {}".format(compList[i]))
            break
        for j, dynType in eType:
            if dynType:
                print("One line for {}".format(compList[i][j]))
                break
            for k, matType in dynType:
                if matType:
                    print("One line for {}".format(compList[i][j][k]))
                    break
        
    # find the number of data dumps
    #dobrDat = dor.DataOutBinReader()
    #cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)

        
def plotAngMomTotal_dyn(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the Angular momentum from the
    # data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    erthAM = np.zeros(numCycs)
    diskAM = np.zeros(numCycs)
    escpAM = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600
        M_P, R_P, erthInds, diskInds, escpInds = dobrDat.findPlanet(0)

        erthAM[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] )[erthInds], 2)
                + np.power(np.asarray(dobrDat.LY[0])[erthInds], 2)
                + np.power(np.asarray(dobrDat.LZ[0])[erthInds], 2), 0.5 )
            * ( np.asarray(dobrDat.M1[0])[erthInds]
                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskAM[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] )[diskInds], 2)
                + np.power(np.asarray(dobrDat.LY[0])[diskInds], 2)
                + np.power(np.asarray(dobrDat.LZ[0])[diskInds], 2), 0.5 )
            * ( np.asarray(dobrDat.M1[0])[diskInds]
                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpAM[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] )[escpInds], 2)
                + np.power(np.asarray(dobrDat.LY[0])[escpInds], 2)
                + np.power(np.asarray(dobrDat.LZ[0])[escpInds], 2), 0.5 )
            * ( np.asarray(dobrDat.M1[0])[escpInds]
                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()

    colors = parula_map(np.linspace(0, 1, 3))
    
    fig, ax_AM = plt.subplots()

    '''
    dobrTs = dobrTs[:2]
    erthAM = erthAM[:2]
    diskAM = diskAM[:2]
    escpAM = escpAM[:2]
    '''

    # earth energy
    ax_AM.fill_between(dobrTs,
                          0,
                          erthAM,
                          color=colors[0])
    # disk energy
    ax_AM.fill_between(dobrTs,
                          erthAM,
                          erthAM + diskAM,
                          color=colors[1])
    # escaped energy
    ax_AM.fill_between(dobrTs,
                          erthAM + diskAM,
                          erthAM + diskAM + escpAM,
                          color=colors[2])
    # plot the boundaries in black
    ax_AM.plot(dobrTs, erthAM, c='k', lw=1)
    ax_AM.plot(dobrTs, erthAM + diskAM, c='k', lw=1)
    ax_AM.plot(dobrTs, erthAM + diskAM + escpAM, c='k', lw=1)

    # make second y axis to display percentage
    ax_perc = ax_AM.twinx()
    ax_perc.set_ylim(0, 1)

    ax_AM.set_ylim(erthAM.min(), (erthAM + diskAM + escpAM).max())
    ax_AM.set_xlim([0, dobrTs[-1]])

    ax_AM.set_xlabel("Time (hr)")
    ax_AM.set_ylabel("Angular Momentum (kg m^2/s)")
    ax_perc.set_ylabel("Percentage of maximum angular momentum")

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes, ("Earth", "Disk", "Escaped"))
    
    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"angMomTotal_dyn.png")

    return dobrTs, erthAM, diskAM, escpAM



def plotAngMomTotal_mat(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the Angular momentum from the
    # data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    AMmat1 = np.zeros(numCycs)
    AMmat2 = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600

        AMmat1[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] ), 2)
                + np.power(np.asarray(dobrDat.LY[0]), 2)
                + np.power(np.asarray(dobrDat.LZ[0]), 2), 0.5 )
            * np.asarray(dobrDat.M1[0]) ).sum()
        AMmat2[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] ), 2)
                + np.power(np.asarray(dobrDat.LY[0]), 2)
                + np.power(np.asarray(dobrDat.LZ[0]), 2), 0.5 )
            * np.asarray(dobrDat.M2[0]) ).sum()

    colors = parula_map(np.linspace(0, 1, 2))
    
    fig, ax_AM = plt.subplots()

    '''
    dobrTs = dobrTs[:2]
    AMmat1 = AMmat1[:2]
    AMmat2 = AMmat2[:2]
    '''
    ax_AM.fill_between(dobrTs,
                       0,
                       AMmat1,
                       color=colors[0])
    ax_AM.fill_between(dobrTs,
                       AMmat1,
                       AMmat1 + AMmat2,
                       color=colors[1])
    # plot the boundaries in black
    ax_AM.plot(dobrTs, AMmat1, c='k', lw=1)
    ax_AM.plot(dobrTs, AMmat1 + AMmat2, c='k', lw=1)
    
    # make second y axis to display percentage
    ax_perc = ax_AM.twinx()
    ax_perc.set_ylim(0, 1)

    ax_AM.set_ylim(AMmat1.min(), (AMmat1 + AMmat2).max())
    ax_AM.set_xlim(0, dobrTs[-1])

    ax_AM.set_xlabel("Time (hr)")
    ax_AM.set_ylabel("Angular Momentum (kg m^2/s)")
    ax_perc.set_ylabel("Percentage of maximum angular momentum")
        
    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes, ("Mantle", "Core"))
    
    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"angMomTotal_mat.png")

    return dobrTs, AMmat1, AMmat2


def plotAngMom_dyn(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the energy from the data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    erthAM = np.zeros(numCycs)
    diskAM = np.zeros(numCycs)
    escpAM = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600
        M_P, R_P, erthInds, diskInds, escpInds = dobrDat.findPlanet(0)

        erthAM[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] )[erthInds], 2)
                + np.power(np.asarray(dobrDat.LY[0])[erthInds], 2)
                + np.power(np.asarray(dobrDat.LZ[0])[erthInds], 2), 0.5 )
            * ( np.asarray(dobrDat.M1[0])[erthInds]
                + np.asarray(dobrDat.M2[0])[erthInds] ) ).sum()
        diskAM[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] )[diskInds], 2)
                + np.power(np.asarray(dobrDat.LY[0])[diskInds], 2)
                + np.power(np.asarray(dobrDat.LZ[0])[diskInds], 2), 0.5 )
            * ( np.asarray(dobrDat.M1[0])[diskInds]
                + np.asarray(dobrDat.M2[0])[diskInds] ) ).sum()
        escpAM[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] )[escpInds], 2)
                + np.power(np.asarray(dobrDat.LY[0])[escpInds], 2)
                + np.power(np.asarray(dobrDat.LZ[0])[escpInds], 2), 0.5 )
            * ( np.asarray(dobrDat.M1[0])[escpInds]
                + np.asarray(dobrDat.M2[0])[escpInds] ) ).sum()

    colors = parula_map(np.linspace(0, 1, 3))
    '''
    dobrTs = dobrTs[:2]
    erthAM = erthAM[:2]
    diskAM = diskAM[:2]
    escpAM = escpAM[:2]
    '''
    AMtot = erthAM + diskAM + escpAM
        
    fig, ax_AM = plt.subplots()

    # plot the fractional angular momentum
    ax_AM.plot(dobrTs, erthAM/AMtot, c=colors[0])
    ax_AM.plot(dobrTs, diskAM/AMtot, c=colors[1])
    ax_AM.plot(dobrTs, escpAM/AMtot, c=colors[2])
    
    ax_AM.set_xlim(dobrTs[0], dobrTs[-1])

    ax_AM.set_xlabel("Time (hr)")
    ax_AM.set_ylabel("Fraction of total angular momentum")

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes, ("Earth", "Disk", "Escaped"))

    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    plt.savefig(saveDir+"angMom_dyn.png")

    return dobrTs, erthAM, diskAM, escpAM



def plotAngMom_mat(dataDir, saveDir, dobrFname="binDat"):
    # Integrate the energy from the data-out-binary-reader file (dobrDat)
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)
    AMmat1 = np.zeros(numCycs)
    AMmat2 = np.zeros(numCycs)
    print("Number of dumps to analyze: {}".format(len(cycs)))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print("Time of this data dump: {}".format(dobrDat.times[0]/3600))
        dobrTs[i] = dobrDat.times[0]/3600

        AMmat1[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] ), 2)
                + np.power(np.asarray(dobrDat.LY[0]), 2)
                + np.power(np.asarray(dobrDat.LZ[0]), 2), 0.5 )
            * np.asarray(dobrDat.M1[0]) ).sum()
        AMmat2[i] = 1e-7*(
            np.power(
                np.power(np.asarray( dobrDat.LX[0] ), 2)
                + np.power(np.asarray(dobrDat.LY[0]), 2)
                + np.power(np.asarray(dobrDat.LZ[0]), 2), 0.5 )
            * np.asarray(dobrDat.M2[0]) ).sum()

    colors = parula_map(np.linspace(0, 1, 2))
    '''
    dobrTs = dobrTs[:2]
    AMmat1 = AMmat1[:2]
    AMmat2 = AMmat2[:2]
    '''
    AMtot = AMmat1 + AMmat2
    
    fig, ax_AM = plt.subplots()

    # plot the fractional angular momentum 
    ax_AM.plot(dobrTs, AMmat1/AMtot, c=colors[0])
    ax_AM.plot(dobrTs, AMmat2/AMtot, c=colors[1])    
    
    ax_AM.set_xlim([dobrTs[0], dobrTs[-1]])

    ax_AM.set_xlabel("Time (hr)")
    ax_AM.set_ylabel("Fraction of total angular momentum")

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig.legend(boxes, ("Mantle", "Core"))

    # make sure saveDir has '/' before saving
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    plt.savefig(saveDir+"angMom_mat.png")

    return dobrTs, AMmat1, AMmat2
