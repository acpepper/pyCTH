'''
Created by Andrew Pepper, 7/11/2019
'''
import numpy as np
import os
import struct
import sys



class DataOutBinReader:
    '''
    DOCUMENTATION FOR THE 'DataOutBinReader' CLASS
    ==============================================
    This class is designed to be used to read 
    the output of CTH's 'DataOut()' when 
    'DataOutFormat(98)' is used. The data is 
    spread across multiple binary files in a 
    CSV-like format.

    EXAMPLE USAGE 1:
    ----------------
    import DataOutBinReader as bnr

    binDat = bnr.DataOutBinReader()
    base_fname = 'myBinary' # e.g. file: myBinary000127.7.dat
    binDat.readAll(base_fname, "/group/stewartgrp/acpepper/some_CTH_run_dir/")
    print(binDat.varNames)
    print(binDat.M1) # This module omits the '+' when specifying material vars
    print(binDat.M2)

    EXAMPLE USAGE 2:
    ----------------
    from DataOutBinReader import *

    binDat = DataOutBinReader()
    fname = 'myBinary000123.0.dat'
    binDat.readOne(fname) # Will only work with dumps by core 0
    print(binDat.varNames)
    print(binDat.T)

    LIST OF ATTRIBUTES:
    -------------------
    numDims      : How many dimensions are needed to describe the data points
    numVars      : The number of database variable  sampled at each timestep
    varNames     : An array of the names of the database variables.
                   NOTE: the variables are named to match CTH, with the only
                         difference being the omission of '+' symbols
    centers      : The positions of the data. A 3D list, the first argument 
                   selects the time, the second selects the dimension:
                   (0, 1, 2) = (x, y, z), and the last selects the cell
    widths       : The size of the cells. A 3D list, the first argument 
                   selects the time, the second selects the dimension:
                   (0, 1, 2) = (x, y, z), and the last selects the cell
    times        : List of times that have been sampled
    [DB VAR]     : The data that was sampled. A 2D list, the first argument 
                   selects the timestep, the second selects the cell that 
                   was sampled. 
                   NOTE: If a variable name would include a '+' the '+' will 
                         be omited (e.g. Material 1 mass 'M+1' becomes 'M1')
    _numDumps    : Number of data dumps performed throughout the simulation
    _numCores    : Number of cores used in the simulation
    _dumpCycs    : An array containg the cycle number at which each data
                   dump was performed
    '''



    def __init__(self):
        # File IO attributes
        self._numDumps = 0
        self._numCores = 0
        self._dumpCycs = []

        # Data attributes
        self.numDims = 0
        self.numVars = 0
        self.varNames = []
        self.centers = []
        self.widths = []
        self.times = []


    # Finds the number of dump cycles for a given file in 'runDir'
    # Also finds the total number of dumps recieved throughout the simulation
    #
    # This function assumes the user follows the default CTH naming convention:
    # [base name]000000.dat
    # [base name]001762.dat
    # ...
    # Or 
    # [base name]000000.0.dat
    # [base name]000000.1.dat
    # ...
    # [base name]000082.0.dat
    # [base name]000082.1.dat
    # ...
    def getCycles(self, fnameBase, runDir):
        fLen = len(fnameBase)
        dumpCycs = []
        numDumps = 0
        for file in os.listdir(runDir):
            if file[:fLen] == fnameBase:
                cycle = int(file[fLen:fLen+6])
                if cycle in dumpCycs:
                    continue
                else:
                    dumpCycs.append(cycle)
                    dumpCycs.sort()
                    numDumps += 1
    
        return dumpCycs, numDumps



    # This function assumes the user follows the default CTH naming convention:
    # [base name]000000.dat
    # [base name]001762.dat
    # ...
    # Or 
    # [base name]000000.0.dat
    # [base name]000000.1.dat
    # ...
    # [base name]000082.0.dat
    # [base name]000082.1.dat
    # ...
    def getNextCycle(self, fnameBase, lastCyc=0, runDir="."):
        fLen = len(fnameBase)

        if len(self._dumpCycs) == 0:
            for file in os.listdir(runDir):
                if file[:fLen] == fnameBase:
                    cycle = int(file[fLen:fLen+6])
                    if cycle in self._dumpCycs: 
                        continue
                    else:
                        self._dumpCycs.append(cycle)
                        self._dumpCycs.sort()
                        self._numDumps += 1
            
            return self._dumpCycs[0]
        
        for i in range(len(self._dumpCycs)):
            if self._dumpCycs[i] <= lastCyc:
                i += 1
            else:
                return self._dumpCycs[i]        

        raise IOError



    # This function assumes the user follows the default CTH naming convention:
    # [base name]001762.0.dat
    # [base name]001762.1.dat
    # ...
    def getNextCore(self, fnameBase, lastCore=0, runDir="."):
        fLen = len(fnameBase)

        if self._numCores == 0 or lastCore == -1:
            for file in os.listdir(runDir):
                if file[:fLen] == fnameBase:
                    # Make sure the data dump was not in serial mode
                    if len(file[fLen+7:-4]) == 0:
                        raise Exception("Data not dumped in default format")

                    core = int(file[fLen+7:-4])
                    if core > self._numCores:
                        self._numCores = core + 1
            
            return 0
        
        if lastCore < self._numCores - 1:
            return lastCore + 1
        
        raise IOError
        


    # This function assumes the user follows the default CTH naming convention:
    # [base name]000000.0.dat
    # [base name]000000.1.dat
    # ...
    # [base name]001762.0.dat
    # [base name]001762.1.dat
    # ...
    def getNextFname(self, lastFname, fnameBase, runDir="."):
        if len(lastFname) == 0:
            cycle = -1
            core = -1
        else:
            fLen = len(fnameBase)
            cycle = int(lastFname[fLen:fLen+6])
            core = int(lastFname[fLen+7:-4])

        while True:
            try:
                cycle = self.getNextCycle(fnameBase, cycle, runDir)
            except IOError:
                break
            while True:
                try:
                    core = self.getNextCore(fnameBase, core, runDir)
                except IOError:
                    break

                fname = fnameBase+"{:0>6}".format(cycle)+".{}.dat".format(core)
                return fname
                
        raise IOError



    # This function assumes the user has dumped the CTH data in parallel:
    # [base name]000000.0.dat
    # [base name]000000.1.dat
    # ...
    # [base name]001762.0.dat
    # [base name]001762.1.dat
    # ...
    def readAll(self, fnameBase, runDir="."):
        cycle = -1
        while True:
            try:
                cycle = self.getNextCycle(fnameBase, cycle, runDir)
                print("go to readSev!")
                self.readSev(fnameBase, cycle, runDir)
            except IOError:
                break
                


    # This function reads all the files in a single CTH data dump.
    # It also assumes the data was dumped in parallel:
    # [base name]000023.0.dat
    # [base name]000023.1.dat
    # ...
    def readSev(self, fnameBase, cycle, runDir="./"):
        # Make sure runDir is valid
        if runDir[-1] != '/':
            runDir = runDir+'/'

        # This will only be true if the function was called by the user
        if len(self._dumpCycs) == 0:
            self._dumpCycs = [cycle]

        core = -1
        while True:
            try:
                core = self.getNextCore(fnameBase, core, runDir)
                fname = fnameBase+"{:0>6}".format(cycle)+".{}.dat".format(core)
                print("fname = {}".format(fname))
                if core == 0:
                    self.readOne(runDir+fname)
                else:
                    self.readOneMore(runDir+fname, cycle)
            except IOError:
                break



    # This function reads a single CTH dump file i.e. it can be used if 
    # 'DataOut()' has been used in serial mode
    # NOTE: The data file MUST CONTAIN A HEADER. Data files dumped by cores 
    #       other than core 0 will not contain headers.
    # NOTE: This function will append the data contained in 'fname' to any
    #       exitsing data, this can be undesirable if you are trying to read
    #       the data files out of chronological order USE AT YOUR OWN RISK
    def readOne(self, fname):
        with open(fname, mode = 'rb') as dataIn:
            # First read the number of dimensions
            # (4 is the number of bytes in an int)
            data = dataIn.read(4)
            if self.numDims == 0:
                self.numDims = struct.unpack('i', data[:4])[0]
                print("numDims = {}".format(self.numDims))

            # The data at each dump is given its own row in the 'centers'
            # 'widths' and data arrays
            self.centers.append([])
            self.widths.append([])
            # Initialize the positions and sizes of the cells
            for i in range(self.numDims):
                self.centers[-1].append([])
                self.widths[-1].append([])

            # Now read the number of variables that were recorded
            # (4 is the number of bytes in an int)
            data = dataIn.read(4)
            if self.numVars == 0:
                self.numVars = struct.unpack('i', data[:4])[0]
                print("numVars = {}".format(self.numVars))

            # Now read the simulation time at which the data was dumped
            # (8 is the number of bytes in an double)
            data = dataIn.read(8)
            self.times.append(struct.unpack('d', data[:8])[0])

            # Next read the names of the variables ... 
            # this gets pretty complicated 
            for j in range(self.numVars):
                # So first we need to read the length of the name string
                # NOTE: the size is stored as an unsigned short
                data = dataIn.read(2)
                colNameLen = struct.unpack('H', data[:2])[0]

                # Now read, character by character, the variable name
                colName = ''
                for i in range(colNameLen):
                    data = dataIn.read(1)
                    char = struct.unpack('c', data[:1])[0]
                    # The '+' symbol is used to distingush material variables
                    # However it will cause problems if we don't remove it
                    if char != "+":
                        colName += char

                if not(colName in self.varNames):
                    self.varNames.append(colName)

                    # This adds a new attribute to the BinaryReader class
                    # DYNAMICALLY. The name of the attribute is stored in 
                    # 'colName' and it's value is initiallized to an empty list
                    # of lists
                    setattr(self, colName, [[]])
                else:
                    getattr(self, self.varNames[j]).append([])

            # Now read the actual data
            while True:
                # NOTE: if there are no more rows we're done reading
                #       so we break
                data = dataIn.read(8)
                if (data):
                    # First the location of this cell
                    # NOTE: The first position was already read when we checked
                    #       to make sure there were more rows
                    self.centers[-1][0].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.centers[-1][1].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.centers[-1][2].append(struct.unpack('d', data[:8])[0])
                    
                    # Next the size of the cell
                    data = dataIn.read(8)
                    self.widths[-1][0].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.widths[-1][1].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.widths[-1][2].append(struct.unpack('d', data[:8])[0])

                    for j in range(self.numVars):
                        data = dataIn.read(8)
                        try:
                            getattr(self, self.varNames[j])[-1].append(struct.unpack('d', data[:8])[0])
                        except struct.error:
                            break
                            
                else:
                    break



    def readOneMore(self, fname, cycle):
        cycInd = [i for i, c in enumerate(self._dumpCycs) if c == cycle][0]

        with open(fname, mode = 'rb') as dataIn:
            while True:
                # NOTE: if there are no more rows we're done reading
                #       so we break
                data = dataIn.read(8)
                if (data):
                    # First the location of this cell
                    # NOTE: The first position was already read when we checked
                    #       to make sure there were more rows
                    self.centers[cycInd][0].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.centers[cycInd][1].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.centers[cycInd][2].append(struct.unpack('d', data[:8])[0])
                    
                    # Next the size of the cell
                    data = dataIn.read(8)
                    self.widths[cycInd][0].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.widths[cycInd][1].append(struct.unpack('d', data[:8])[0])
                    data = dataIn.read(8)
                    self.widths[cycInd][2].append(struct.unpack('d', data[:8])[0])

                    for j in range(self.numVars):
                        data = dataIn.read(8)
                        try:
                            getattr(self, self.varNames[j])[cycInd].append(struct.unpack('d', data[:8])[0])
                        except struct.error:
                            break
                            
                else:
                    break



    # returns an array of indexes which lie in the 2d planar subset of the 
    # simulation domain. 
    # 'p' -> a point in the plane
    # 'n' -> the normal of the plane
    # NOTE: This function works best when 'p' lies on the boundary of CTH cells
    def slice2dFrom3d(self, p, n, ti=0):
        n = n/np.linalg.norm(n)
        inds = []
        for i, (x, y, z, wx, wy, wz) in enumerate(zip(self.centers[ti][0], self.centers[ti][1], self.centers[ti][2], self.widths[ti][0], self.widths[ti][1], self.widths[ti][2])):
            relPos = np.asarray([x - p[0], y - p[1], z - p[2]])
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



    def getGU(self):
        reqAttrs = ['SGU', 'M1', 'M2']
        for attr in reqAttrs:
            if not hasattr(self, attr):
                raise AttributeError("DataOutBinReader class has no attribute: {}".format(attr))

        return np.asarray(self.SGU[0])*( np.asarray(self.M1[0])
                                        + np.asarray(self.M2[0]) )



    def getCOM3d_mass(self, ti=0):
        reqAttrs = ['M1', 'M2']
        for attr in reqAttrs:
            if not hasattr(self, attr):
                raise AttributeError("DataOutBinReader class has no attribute: {}".format(attr))
        
        com = [0, 0, 0]
        masses = np.asarray(self.M1[ti]) + np.asarray(self.M2[ti])
        for x, y, z, m in zip(np.asarray(self.centers[ti][0]), np.asarray(self.centers[ti][1]), np.asarray(self.centers[ti][2]), np.asarray(masses)):
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



    def getCOM3d(self, ti=0):
        reqAttrs = ['SGU', 'M1', 'M2']
        for attr in reqAttrs:
            if not hasattr(self, attr):
                raise AttributeError("DataOutBinReader class has no attribute: {}".format(attr))

        comGuess = self.getCOM3d_mass()
        GU = self.getGU()
        GU_ord = np.argsort(GU)
        bestInd = []
        # 16 is an qualitative, empirical choice
        numBestInds = 16
        for i in GU_ord:
            if self.M1[ti][i] > 1/3*self.M2[ti][i]:
                continue
            
            distToCOM = np.linalg.norm([self.centers[ti][0][i] - comGuess[0],
                                        self.centers[ti][1][i] - comGuess[1], 
                                        self.centers[ti][2][i] - comGuess[2]])
            if  distToCOM < 7e8: # ~R_earth
                bestInd.append(i)
                if len(bestInd) >= numBestInds:
                    break

        com = [0, 0, 0]
        for i in bestInd:
            com[0] += self.centers[ti][0][i]/numBestInds
            com[1] += self.centers[ti][1][i]/numBestInds
            com[2] += self.centers[ti][2][i]/numBestInds

        return com, bestInd



    def getRads3d(self, ti=0, **kwargs):
        try:
            com = kwargs["com"]
        except KeyError:
            com = self.getCOM3d()[0]

        rads = []

        for x, y, z in zip(np.asarray(self.centers[ti][0]), np.asarray(self.centers[ti][1]), np.asarray(self.centers[ti][2])):
            rads.append(np.linalg.norm([x - com[0], y - com[1], z - com[2]]))

        return np.asarray(rads)



    def getL3d(self, ti=0, **kwargs):
        try:
            inds = kwargs["inds"]
        except KeyError:
            inds = np.arange(len(self.centers[ti][0]))
        try:
            useCom = kwargs["useCom"]
        except KeyError:
            useCom = False
        try:
            com = kwargs["com"]
        except KeyError:
            if useCom:
                com = self.getCOM3d()[0]
            else:
                com = [0, 0, 0]
            
        reqAttrs = ['M1', 'M2', 'VX', 'VY', 'VZ']
        for attr in reqAttrs:
            if not hasattr(self, attr):
                raise AttributeError("DataOutBinReader class has no attribute: {}".format(attr))
        
        masses = np.asarray(self.M1[ti])[inds] + np.asarray(self.M2[ti])[inds]
        Ls = []
        for x, y, z, vx, vy, vz, m in zip(np.asarray(self.centers)[ti,0,inds],
                                          np.asarray(self.centers)[ti,1,inds],
                                          np.asarray(self.centers)[ti,2,inds],
                                          np.asarray(self.VX)[ti, inds],
                                          np.asarray(self.VY)[ti, inds],
                                          np.asarray(self.VZ)[ti, inds],
                                          masses):
            if useCom:
                r = [x - com[0], y - com[1], z - com[2]]
                Ls.append(np.cross(r, [vx, vy, vz])*m)
            else:
                Ls.append(np.cross([x, y, z], [vx, vy, vz])*m)
                
        return Ls



    def findPlanet(self, ti=0, **kwargs):
        try:
            com = kwargs["com"]
        except KeyError:
            com = self.getCOM3d()[0]
        try:
            lunarD = kwargs["lunarD"]
        except KeyError:
            lunarD = 2.5

        reqAttrs = ['KE', 'M1', 'M2', 'VX', 'VY', 'VZ']
        for attr in reqAttrs:
            if not hasattr(self, attr):
                raise AttributeError("DataOutBinReader class has no attribute: {}".format(attr))
            
        rads = self.getRads3d(ti, com=com)
        masses = np.asarray(self.M1[ti]) + np.asarray(self.M2[ti])
        
        # Here we sort the indexes by distance from the center of mass
        # The arrays which are in this order are the following:
        radInds = rads.argsort()

        # we define R_P as the average radius of the 16 cells having the
        # greatest specific kenetic energy, which are no further that
        # ~3*R_earth from the center of mass
        upBnd_ind = np.searchsorted(rads[radInds], 2e9)
        keInds = np.asarray(self.KE[ti])[radInds][:upBnd_ind].argsort()
        R_P = rads[radInds][keInds][-8:].mean()

        # Enforce boundary conditions on R_P: 0.5*R_E <= R_P <= 2*R_E
        if R_P < 3e8:
            R_P = 3e8
        elif R_P > 1.2e9:
            R_P = 1.2e9

        # calculate an initial guess of the planet mass, M_P
        M_P = 0
        for k, j in enumerate(radInds):
            if k > keInds[-1]:
                break
            M_P += masses[j]
        
        # calculate the relaxed orbits and compare these to R_P.
        # For each cell, solve for 'aScaled' (the orbital radius the cell 
        # would have after relaxation multiplied by the mass of the planet)
        # using 
        # sqrt(G * M_P * a_eq) = [specific angular momentum]
        #              aScaled = a_eq*M_P
        aScaled = np.zeros(radInds.shape)
        omegas = np.zeros(radInds.shape)
        for j in radInds:
            pos = [self.centers[ti][0][j] - com[0], 
                   self.centers[ti][1][j] - com[1], 
                   self.centers[ti][2][j] - com[2]]
            vel = [self.VX[ti][j], 
                   self.VY[ti][j],
                   self.VZ[ti][j]]
            am = np.linalg.norm(np.cross(pos, vel)) # specific angular momentum
            omegas[j] = am/(rads[j]**2)
            aScaled[j] = pow(am, 2) / 6.674e-8 # [orbital radius]*M_P

        print("initial M_P = {} kg".format(M_P/1e3))
        
        # Now we itteratively update the planet mass estimate by screening the
        # cells with the aScaled array:
        # if aScaled/M_P <= R_P => M_P++
        # if aScaled/M_P > R_P => disk
        newM_P = 0
        while True:
            for j in radInds:
                # Ignore empty cells
                if masses[j] < 1024*sys.float_info.epsilon:
                    continue
                # Ignore escaped cells
                if self.KE[ti][j] - 6.674e-8*M_P/rads[j] > 0:
                    continue
                # Add mass if inside R_P
                if aScaled[j]/M_P <= R_P:
                    newM_P += masses[j]

            # Enforce boundary conditions on M_P: 0.5*M_E <= M_P <= 2*M_E
            if newM_P < 3e27:
                newM_P = 3e27
            elif newM_P > 1.2e28:
                newM_P = 1.2e28
                
            print("updated M_P = {} kg".format(newM_P/1e3))

            # If the the change in planet mass is proportionally small, 
            # exit the itteration
            if abs( newM_P/M_P  - 1 ) < pow(2.0, -6):
                M_P = newM_P
                break
            else:
                M_P = newM_P
                newM_P = 0

        # Find the escaped mass by comparing the cell's specific kinetic energy
        # to the specific gravitational potential energy of the planet
        pInds = []
        diskInds = []
        escpdInds = []
        for j in radInds:
            # Ignore empty cells
            if masses[j] < 1024*sys.float_info.epsilon:
                continue

            if self.KE[ti][j] - 6.674e-8*M_P/rads[j] > 0:
                escpdInds.append(j)
            elif aScaled[j]/M_P <= R_P:
                pInds.append(j)
            else:
                diskInds.append(j)

        L_esc, L_D, L_tot = self.getFractionalLs(escpdInds, 
                                                 diskInds,
                                                 com=com)
        print("R_P = {} km".format(R_P/1e5))
        M_esc = masses[escpdInds].sum()
        print("M_esc = {} kg".format(M_esc/1e3))
        M_D = masses[diskInds].sum()
        print("M_D = {} kg".format(M_D/1e3))
        M_D_frac = ( ( np.asarray(self.M1[ti])[diskInds]
                       *np.asarray(self.M1ID[ti])[diskInds] )
                     +( np.asarray(self.M2[ti])[diskInds]
                        *np.asarray(self.M2ID[ti])[diskInds] ) ).sum()
        print("fraction of M_D from target = {}%".format(M_D_frac/M_D*100))
        
        # We use an angular momentum-balance equation to calculate the
        # proportion of the disk material which accretes into a moon
        #
        # First we must find the average angular velocity of the planet
        omega_P = 0
        for j in pInds:
            omega_P += masses[j]*omegas[j]
        omega_P /= M_P
        print("omega_P = {} rev/hr".format(omega_P*3600/2/np.pi))

        # Next, we calculate the integrated mass.
        # We'll need this for the next calcualtion
        mSum = [masses[radInds][0]]
        for i in radInds[1:]:
            mSum.append( mSum[-1] + masses[i] )
            if mSum[-1] > M_P:
                M_enc = mSum[-1]
        
        # Now we must calculate the roche radius using fixed-point-itteration
        # roche_N+1 = 2.44*pow( 3*M_enc(roche_N)/4/np.pi/lunarD, 1./3 )
        # Our starting point is the roche radius of the 'M_P' we calcualted
        # earlier
        roche0 = 2.44*pow(3*M_P/4/np.pi/lunarD, 1./3)
        rocheInd = np.searchsorted(rads[radInds], roche0)
        rocheIndDiff = rocheInd
        roche1 = rads[radInds][rocheInd - 1]
        while abs(rocheIndDiff) > 1:
            M_enc = mSum[rocheInd - 1]
            roche0 = 2.44*pow(3*M_enc/4/np.pi/lunarD, 1./3)
            rocheIndDiff = np.searchsorted(rads[radInds], roche0) - rocheInd
            rocheInd += rocheIndDiff
            roche1 = rads[radInds][rocheInd - 1]
            
        # Define several constants
        A = pow(2*6.674e-8*M_enc*roche1, 0.5)
        B = R_P**2*M_D*omega_P
        M_L = (L_D - B)/(A - R_P**2*omega_P)
        print("M_L = {} kg".format(M_L/1e3))

        return M_P, R_P, pInds, diskInds, escpdInds



    def getFractionalLs(self, escpdInds, diskInds, ti=0, **kwargs):
        try:
            com = kwargs["com"]
        except KeyError:
            com = self.getCOM3d()[0]

        # First examine the escaped mass
        escpdL = np.asarray( self.getL3d(ti, inds=escpdInds) )
        if len(escpdL) > 0:
            L_esc = np.linalg.norm([escpdL[:, 0].sum(), 
                                    escpdL[:, 1].sum(), 
                                    escpdL[:, 2].sum()])
            print("L_esc = {} L_EM".format(L_esc/3.5e41))
        else:
            L_esc = 0

        # Next the disk mass
        diskL = np.asarray( self.getL3d(ti, inds=diskInds) )
        if len(diskL) > 0:
            L_D = np.linalg.norm( [diskL[:, 0].sum(), 
                                   diskL[:, 1].sum(), 
                                   diskL[:, 2].sum()] )
            print("L_D = {} L_EM".format(L_D/3.5e41))
        else:
            L_D = 0

        # Finally, the total mass
        # (it is assumed this is non-empty)
        totalL = np.asarray( self.getL3d(ti) )
        L_tot = np.linalg.norm( [totalL[:, 0].sum(), 
                                 totalL[:, 1].sum(), 
                                 totalL[:, 2].sum()] )
        print("L_tot = {} L_EM".format(L_tot/3.5e41))
        
        return L_esc, L_D, L_tot

