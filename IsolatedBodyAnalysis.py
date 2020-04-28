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



def getCoreTemp(dobrDat,saveDir, ti=0, **kwargs):
    # get the center of mass
    com, comInds = dobrDat.getCOM3d(ti)
    
    # find the temperature of the COM
    comTemps = np.zeros(len(comInds))
    for i, comi in enumerate(comInds):
        comTemps[i] = dobrDat.TM2[ti][comi]
        
    avgTemp = 0
    avgTstd = 0
    
    return comTemps.mean(), (avgTemp, avgTstd)



def plotEnergySpec_dyn(dataDir, saveDir, dobrFname="binDat", **kwargs):
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)

    numCycs = 3
    
    # First collect the energies
    erthEken = np.zeros(numCycs)
    diskEken = np.zeros(numCycs)
    escpEken = np.zeros(numCycs)
    erthEint = np.zeros(numCycs)
    diskEint = np.zeros(numCycs)
    escpEint = np.zeros(numCycs)
    erthEgra = np.zeros(numCycs)
    diskEgra = np.zeros(numCycs)
    escpEgra = np.zeros(numCycs)
    print "Number of dumps to analyze: {}".format(len(cycs))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs[-3:]):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print "Time of this data dump: {}".format(dobrDat.times[0]/3600)
        dobrTs[i] = dobrDat.times[0]/3600
        # find the disk and escaped material
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

    # prepare to calcualte the FFT
    # NOTE: Since we are limited by the nyquist frequency (i.e. 2x the
    #       sampling frequency), we have to determine if the number of
    #       samples is even or odd and allocate the number of frequencies
    #       accordingly
    numFreq = numCycs if numCycs % 2 == 0 else (numCycs + 1)/2
    Fs = np.linspace(0.0, 0.5/dobrTs[1], numFreq)*3600

    # Compute the FFT
    erthEkenSpec = np.fft.rfft(erthEken)
    diskEkenSpec = np.fft.rfft(diskEken)
    escpEkenSpec = np.fft.rfft(escpEken)    
    erthEintSpec = np.fft.rfft(erthEint)
    diskEintSpec = np.fft.rfft(diskEint)
    escpEintSpec = np.fft.rfft(escpEint)    
    EgraSpec = np.fft.rfft(erthEgra + diskEgra + escpEgra)

    # func = get_inv_fourier(amp)

    colors = parula_map(np.linspace(0, 1, 7))
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # plot the energy evolution
    ax1.plot(dobrTs, erthEken, c=colors[0])
    ax1.plot(dobrTs, diskEken, c=colors[1])
    ax1.plot(dobrTs, escpEken, c=colors[2])
    ax1.plot(dobrTs, erthEint, c=colors[3])
    ax1.plot(dobrTs, diskEint, c=colors[4])
    ax1.plot(dobrTs, escpEint, c=colors[5])
    ax1.plot(dobrTs, erthEgra + diskEgra + escpEgra, c=colors[6])
    
    # plot the energy spectrum
    ax2.plot(Fs, np.abs(erthEkenSpec), c=colors[0])
    ax2.plot(Fs, np.abs(diskEkenSpec), c=colors[1])
    ax2.plot(Fs, np.abs(escpEkenSpec), c=colors[2])
    ax2.plot(Fs, np.abs(erthEintSpec), c=colors[3])
    ax2.plot(Fs, np.abs(diskEintSpec), c=colors[4])
    ax2.plot(Fs, np.abs(escpEintSpec), c=colors[5])
    ax2.plot(Fs, np.abs(EgraSpec), c=colors[6])

    # Make it look pretty
    ax1.set_title("Energy evolution")
    ax2.set_title("Energy spectrum")
    
    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel("Energy (J)")

    ax1.set_xlim(0, dobrTs[-1])
    
    ax2.set_xlabel("Frequency (rot/hr)")
    ax2.set_ylabel("Amplitude (J)")

    ax2.set_xlim(0, Fs[-1])

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    fig1.legend(boxes,
                ("GU",
                 "Earth KE", "Disk KE", "Escaped KE",
                 "Earth IE", "Disk IE", "Escaped IE"))
    fig2.legend(boxes,
                ("GU",
                 "Earth KE", "Disk KE", "Escaped KE",
                 "Earth IE", "Disk IE", "Escaped IE"))

    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig1.savefig(saveDir+"energyEvol_dyn.png")
    fig1.savefig(saveDir+"energySpec_dyn.png")



def plotEnergySpec_mat(dataDir, saveDir, dobrFname="binDat", **kwargs):
    dobrDat = dor.DataOutBinReader()
    cycs, numCycs = dobrDat.getCycles(dobrFname, dataDir)

    numCycs = 3
    
    # First collect the energies
    EkenMat1 = np.zeros(numCycs)
    EkenMat2 = np.zeros(numCycs)
    EintMat1 = np.zeros(numCycs)
    EintMat2 = np.zeros(numCycs)
    EgraMat1 = np.zeros(numCycs)
    EgraMat2 = np.zeros(numCycs)
    print "Number of dumps to analyze: {}".format(len(cycs))
    dobrTs = np.zeros(numCycs)
    for i, cyc in enumerate(cycs[-3:]):
        dobrDat = dor.DataOutBinReader()
        dobrDat.readSev(dobrFname, cyc, dataDir)
        print "Time of this data dump: {}".format(dobrDat.times[0]/3600)
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

    # prepare to calcualte the FFT
    # NOTE: Since we are limited by the nyquist frequency (i.e. 2x the
    #       sampling frequency), we have to determine if the number of
    #       samples is even or odd and allocate the number of frequencies
    #       accordingly
    numFreq = numCycs if numCycs % 2 == 0 else (numCycs + 1)/2
    Fs = np.linspace(0.0, 0.5/dobrTs[1], numFreq)*3600

    # Compute the FFT
    EkenMat1Spec = np.fft.rfft(EkenMat1)
    EkenMat2Spec = np.fft.rfft(EkenMat2)
    EintMat1Spec = np.fft.rfft(EintMat1)
    EintMat2Spec = np.fft.rfft(EintMat2)
    EgraSpec = np.fft.rfft(EgraMat1 + EgraMat2)

    colors = parula_map(np.linspace(0, 1, 5))
    
    fig, axs = plt.subplots(1, 2)

    # plot the energy evolution
    axs[0].plot(dobrTs, EkenMat1, c=colors[0])
    axs[0].plot(dobrTs, EkenMat2, c=colors[1])
    axs[0].plot(dobrTs, EintMat1, c=colors[2])
    axs[0].plot(dobrTs, EintMat2, c=colors[3])
    axs[0].plot(dobrTs, EgraMat1 + EgraMat2, c=colors[4])

    # plot the energy spectrum
    axs[1].plot(Fs, np.abs(EkenMat1Spec), c=colors[0])
    axs[1].plot(Fs, np.abs(EkenMat2Spec), c=colors[1])
    axs[1].plot(Fs, np.abs(EintMat1Spec), c=colors[2])
    axs[1].plot(Fs, np.abs(EintMat2Spec), c=colors[3])
    axs[1].plot(Fs, np.abs(EgraSpec), c=colors[4])

    # Make it look pretty
    axs[0].set_title("Energy evolution")
    axs[1].set_title("Energy spectrum")
    
    axs[0].set_xlabel("Time (hr)")
    axs[0].set_ylabel("Energy (J)")

    axs[0].set_xlim(0, dobrTs[-1])
    
    axs[1].set_xlabel("Frequency (rot/hr)")
    axs[1].set_ylabel("Amplitude (J)")

    axs[1].set_xlim(0, Fs[-1])

    # make proxy artists for legend entries
    boxes = []
    for c in colors:
        boxes.append(mpatches.FancyBboxPatch((0, 0), 1, 1, fc=c))

    plt.legend(boxes,
               ("GU",
                "Mantle KE", "Core KE",
                "Mantle IE", "Core IE"))

    # Make sure save directory is valid
    if saveDir[-1] != '/':
        saveDir = saveDir+'/'
    fig.savefig(saveDir+"energySpec_mat.png")
