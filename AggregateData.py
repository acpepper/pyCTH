'''
Functions to read and write aggregate data for post-processing.
AggData class definition; this class is used to generate and store the
aggregate data.

NOTE: the data is converted from CGS to SI units

List of Functions
-----------------
writeAggData()
readAggData()
'''
import numpy as np
try:
    import cpickle as pickle
except ImportError:
    import pickle
from . import DataOutBinReader as dobr
from . import HcthReader as hcr



def writeAggData(aggData, saveDir, pickleFname="aggData.pickle"):
    # make sure 'saveDir' is formatted properly
    if saveDir[-1] != '/':
        saveDir += '/'
        
    # Open the save file
    with open(saveDir+pickleFname, "wb") as saveFile:
        pickle.dump(aggData, saveFile)
        


def readAggData(saveDir, pickleFname="aggData.pickle"):
    # make sure 'saveDir' is formatted properly
    if saveDir[-1] != '/':
        saveDir += '/'
            
    # Open the save file
    with open(saveDir+pickleFname, "rb") as saveFile:
        aggData = pickle.load(saveFile)

    return aggData


class AggData:
    '''
    DOCUMENTATION FOR THE 'AggData' CLASS
    ==============================================
    
    Units: SI

    '''
    def __init__(self, **kwargs):
        self.genCompNames = ["KE", "IE", "GU"]
        self.matCompNames = []
        self.dynCompNames = ["none"]
        self.vecCompNames = []

        self.impactMode = True
        self.AMvectorMode = True

        self.cycles = []
        self.numDumps = 0
        
        self.numMats = 0
        self.numEs = 0
        
        self.Enames = []
        self.AMnames = []
        
        self.Es = []
        self.AMs = []
        self.Ts = []


        
    def generateAggData(self, dataDir, **kwargs):
        '''
        Function to aggregate the the data from DataOutBinary()
        
        NOTE: if AMvectorMode == True, the number of dimensions is assumed 
              to be 3
    
        Possible kwargs:
        ----------------
        saveDir    :   The directory to save to aggregated data
                       Default: dataDir
        dobrFname  :   The prefix used by spyplot when DataOutBinary() is 
                       called.
                       Default: "binDat"
        impactMode :   If 'True' then the energy data will be divided into 3 
                       additional categories: Earth, disk, and escaped
                       Default: True
        AMvectorMode : If 'True' the angular momentum data will be divided into
                       its vector components, else we simply store magnitude
                       Default: 'True'
        '''
        # make sure dataDir is formatted properly
        if dataDir[-1] != '/':
            dataDir += '/'

        # get IO variables
        try:
            dobrFname = kwargs["dobrFname"]
        except KeyError:
            dobrFname = "binDat"
        # get saveDir
        try:
            saveDir = kwargs["saveDir"]
        except KeyError:
            saveDir = dataDir
        # special behavior if we're analyzing an impact
        try:
            self.impactMode = kwargs["impactMode"]
        except KeyError:
            self.impactMode = self.impactMode
        # if AM components are desired:
        try:
            self.AMvectorMode = kwargs["AMvectorMode"]
        except KeyError:
            self.AMvectorMode = self.AMvectorMode
            
        # find the number of materials
        hcthDat = hcr.HcthReader(dataDir+"hcth")
        self.numMats = len(hcthDat.mcube[0])
    
        # find the number of energy components:
        # 3 general components: kinetic, internal, and gravitational potential
        # any number of material components
        # 3 dynamical components, if impact: Earth, disk, and escaped
        self.numEs = 3*self.numMats*(3 if self.impactMode else 1)

        # NOTE: there is a difference in 'self.genCompNames' and the local
        #       variable 'genCompNames' because the gravitational potential
        #       energy flag in 'self.genCompNames' is "GU", while the flag in
        #       'genCompNames' is "SGU". This is because 'genCompNames' refers
        #       to specific energy values
        genCompNames = ["KE", "IE", "SGU"]
        for i in range(self.numMats):
            self.matCompNames.append("M{}".format(i+1))
        if self.impactMode:
            self.dynCompNames = ["Earth", "disk", "escaped"]
        if self.AMvectorMode:
            self.vecCompNames = ["LX", "LY", "LZ"]
        else:
            self.vecCompNames = ["MAG"]

        # collect the names of every energy component to be stored.
        # NOTE: the local variable 'Enames' is different from 'self.Enames' for
        #       the name reason that 'genCompNames' is different from
        #       'self.genCompNames' above
        # general components
        Enames = []
        for i, genComp in enumerate(genCompNames):
            # material components
            for j, matComp in enumerate(self.matCompNames):
                if self.impactMode:
                    # dynamic components
                    for k, dynComp in enumerate(self.dynCompNames):
                        Enames.append([genComp, matComp, dynComp])
                        self.Enames.append([self.genCompNames[i],
                                            matComp,
                                            dynComp])
                else:
                    Enames.append([genComp, matComp])
                    self.Enames.append([self.genCompNames[i],
                                        matComp])

        # collect the names of every angular momentum component to be stored.
        # general components
        for i, vecComp in enumerate(self.vecCompNames):
            # material components
            for j, matComp in enumerate(self.matCompNames):
                if self.impactMode:
                    # dynamic components
                    for k, dynComp in enumerate(self.dynCompNames):
                        self.AMnames.append([vecComp, matComp, dynComp])
                else:
                    self.AMnames.append([vecComp, matComp])
    
        # find the number of DataOutBinary() dumps
        dobrDat = dobr.DataOutBinReader()
        self.cycles, self.numDumps = dobrDat.getCycles(dobrFname, dataDir)
        
        # initialize the energy array, angular momentum array, and time array
        # potential array (this will be needed to normalize the energy)
        self.Es = np.zeros((self.numEs, self.numDumps))
        self.AMs = np.zeros((self.numEs, self.numDumps))
        self.Ts = np.zeros(self.numDumps)
        for j, cyc in enumerate(self.cycles):
            # get the binary data
            dobrDat = dobr.DataOutBinReader()
            dobrDat.readSev(dobrFname, cyc, dataDir)
            
            # store the time of this data dump
            self.Ts[j] = dobrDat.times[0]/3600
            print("Time of this data dump: {} hr".format(self.Ts[j]))
            
            # if running in impactMode, we need an extra variable to
            # discriminate the cells based on their dynamical category
            if self.impactMode:
                # Find the dynamical energy components
                M_P, R_P, EInds, dInds, eInds = dobrDat.findPlanet(0)
                inds = {}
                inds[self.dynCompNames[0]] = EInds
                inds[self.dynCompNames[1]] = dInds
                inds[self.dynCompNames[2]] = eInds

            # calculate the energies
            for i, Ecomps in enumerate(Enames):
                if self.impactMode:
                    self.Es[i, j] = 1e-7*(
                        np.asarray(
                            getattr(dobrDat,
                                    Ecomps[0])[0]
                        )[ inds[Ecomps[2]] ]
                        * np.asarray(
                            getattr(dobrDat,
                                    Ecomps[1])[0]
                        )[ inds[Ecomps[2]] ]
                    ).sum()
                else:
                    self.Es[i, j] = 1e-7*(
                        np.asarray( getattr(dobrDat, Ecomps[0])[0] )
                        * np.asarray( getattr(dobrDat, Ecomps[1])[0] ) ).sum()

            # collect the specific angular momentum, if running in
            # AMvectorMode, the components of the specific angular momentum
            # are collected
            if self.AMvectorMode:
                specAM = {}
                specAM[self.vecCompNames[0]] = np.asarray(dobrDat.LX[0])
                specAM[self.vecCompNames[1]] = np.asarray(dobrDat.LY[0])
                specAM[self.vecCompNames[2]] = np.asarray(dobrDat.LZ[0])
            else:
                specAM = {"MAG" : np.power(
                    np.power(  np.asarray(dobrDat.LX[0]), 2)
                    + np.power(np.asarray(dobrDat.LY[0]), 2)
                    + np.power(np.asarray(dobrDat.LZ[0]), 2), 0.5 ) }

            # calculate the angular momenta                
            for i, AMcomps in enumerate(self.AMnames):
                if self.impactMode:
                    self.AMs[i, j] = 1e-7*(
                        specAM[ AMcomps[0] ][ inds[AMcomps[2]] ]
                        *np.asarray(getattr(dobrDat,
                                            AMcomps[1])[0])[ inds[AMcomps[2]] ]
                    ).sum()
                    
                else:
                    self.AMs[i, j] = 1e-7*(
                        specAM[ AMcomps[0] ]
                        * np.asarray( getattr(dobrDat, AMcomps[1]) ) ).sum()

        # Write to a binary file
        writeAggData(self, saveDir)
