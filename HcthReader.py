'''
Created April 13th, 2018
@author: Erik Davies

Modified July 12, 2018
by Andrew Pepper
'''
import numpy as np
import struct



class HcthReader:
    '''
    DOCUMENTATION FOR THE 'HcthReader' CLASS
    ========================================
    This class is designed to read CTH's 
    'hcth' files, which are produced by the 
    'histt' block.

    EXAMPLE USAGE 1:
    ----------------
    import HcthReader as hcr

    fname = "/group/stewartgrp/acpepper/some_CTH_run_dir/hcth")
    histDat = hcr.HcthReader(fname)
    print histDat.varNames
    print histDat.gcube[3]

    EXAMPLE USAGE 2:
    ----------------
    from HcthReader import *

    fname = "/group/stewartgrp/acpepper/some_CTH_run_dir/hcth")
    histDat = hcr.HcthReader(fname)
    histDat.numTrcVars = 38
    histDat.numMatVars = 20
    histDat.numGlbVars = 20
    print histDat.getEtots()

    LIST OF ATTRIBUTES:
    -------------------
    varNames : The names of ALL variable in the inputfile
    times    : A 1D array with the times at which the hcth file was written to
    tcube    : A 3D array for tracer particle data. The 0th dimension is the
               number of variables associated with each tracer particle. The 
               1st dimension is the number of tracers. Finally, the 2nd 
               dimension is the number of timesteps where the hcth file has 
               been written to (i.e. the time axis of the data).
    mcube    : A 3D array for the material data. The 0th dimension is the 
               number of material variables. The 1st dimension is the number 
               of materials. The 2nd dimension is the number of timesteps where
               the hcth file has been written to.
    gcube    : A 2D array for the global data. The 0th dimension is the number
               of global variables. The 1st dimension is the number of 
               timesteps where the hcth file has been written to.
    '''



    def __init__(self, hcth_fname):
        self.varNames, self.times, self.tcube, self.mcube, self.gcube = read_hcth(hcth_fname)

        # trim the redundant first and last entries from the data
        self.times = self.times[1:-1]
        self.tcube = self.tcube[:, :, 1:-1]
        self.mcube = self.mcube[:, :, 1:-1]
        self.gcube = self.gcube[:, 1:-1]

        # These are listed in the order in which they appear in 'varNames'
        # NOTE: The user should set these before using any of the 'HcthReader'
        #       methods
        # NOTE: The values used here should be an upper bound
        self.numTrcVars = 38
        self.numMatVars = 20
        self.numGlbVars = 20




    def getMasses():
        massi = -1
        for i, name in enumerate(self.varNames):
            if name == 'MASS':
                # gcube variable, so we must adjust i to fit the gcube array
                massi = i - self.numMatVars - self.numTrcVars

        if massi == -1:
            print "WARNING: MASS was not included in the history file"
            
        return self.gcube[massi]



    def getEtots():
        etoti = -1        
        for i, name in enumerate(self.varNames):
            if name == 'ETOT':
                # gcube variable, so we must adjust i to fit the gcube array
                etoti = i - self.numMatVars - self.numTrcVars

        if etoti == -1:
            print "WARNING: ETOT was not included in the history file"

        return self.gcube[etoti]

    

    def getEkens():
        ekeni = -1
        for i, name in enumerate(self.varNames):
            if name == 'EK':
                # gcube variable, so we must adjust i to fit the gcube array
                ekeni = i - self.numMatVars - self.numTrcVars

        if ekeni == -1:
            print "WARNING: KE was not included in the history file"

        return self.gcube[ekeni]



    def getEints():
        einti = -1
        for i, name in enumerate(self.varNames):
            if name == 'EINT':
                # gcube variable, so we must adjust i to fit the gcube array
                einti = i - self.numMatVars - self.numTrcVars

        if einti == -1:
            print "WARNING: EINT was not included in the history file"

        return self.gcube[einti]



# This routine opens an hcth binary file for the purpose of locating the 
# leading shock front in 1-D shock simulation. It does this by looking through
# the x-position, pressure and particle velocity of all the tracer particles.
# The routine looks for the highest gradient and defines this as the shock 
# front. Later times are excluded due to the fracture of material leading to 
# many possible shock fronts. However, the largest pressure differential should
# give the relevant front at any point.
#
# Returns: 
# - (varnames) The names of ALL variable in the inputfile
# - (timearr) A 1D array with the times at which the hcth file was written to
#
# - (tcube) A 3D array for tracer particle data. The 0th dimension is the
#    number of variables associated with each tracer particle. The 1st 
#    dimension is the number of tracers. Finally, the 2nd dimension is the 
#    number of timesteps where the hcth file has been written to (i.e. the 
#    time axis of the data).
#
# - (mcube) A 3D array for the material data. The 0th dimension is the number of
#   material variables. The 1st dimension is the number of materials. The 2nd
#   dimension is the number of timesteps where the hcth file has been 
#   written to.
#
# - (gcube) A 2D array for the global data. The 0th dimension is the number of 
#   global variables. The 1st dimension is the number of timesteps where the
#   hcth file has been written to.
def read_hcth(inputfile):
    f=open(inputfile,'rb')

    # Start reading in header
    temp=struct.unpack('f',f.read(4))[0] # spacer
    time=struct.unpack('f',f.read(4))[0]
    dt=struct.unpack('f',f.read(4))[0]
    cpu=struct.unpack('f',f.read(4))[0]
    icycle=struct.unpack('i',f.read(4))[0]
    d1=struct.unpack('f',f.read(4))[0]
    d2=struct.unpack('f',f.read(4))[0]
    d3=struct.unpack('f',f.read(4))[0]
    
    print 'icycle: ', icycle
    print time,dt,cpu,icycle,d1,d2,d3, '\n'
    
    name_len=80
    title="" # Define title

    for i in range(name_len):
        title=title+struct.unpack('c',f.read(1))[0]
    # print 'Title:\n', title, '\n'

    # Repeat the beginning stuff
    for i in range(4):
        temp=struct.unpack('f',f.read(4))[0] # More spacer
            
    time=struct.unpack('f',f.read(4))[0]
    dt=struct.unpack('f',f.read(4))[0]
    cpu=struct.unpack('f',f.read(4))[0]
    icycle=struct.unpack('i',f.read(4))[0]
    d1=struct.unpack('f',f.read(4))[0]
    d2=struct.unpack('f',f.read(4))[0]
    d3=struct.unpack('f',f.read(4))[0]

    print 'icycle: ', icycle
    # print time,dt,cpu,icycle,d1,d2,d3, '\n'

    # Datem time, jobid, cd
    b_len=46
    date=""
    for i in range(b_len):
        date=date+struct.unpack('c',f.read(1))[0]
    
    # print 'Date:\n', date, '\n'

    for i in range(2):
        temp=struct.unpack('f',f.read(4))[0] # More spacer
    
    time=struct.unpack('f',f.read(4))[0]
    dt=struct.unpack('f',f.read(4))[0]
    cpu=struct.unpack('f',f.read(4))[0]
    icycle=struct.unpack('i',f.read(4))[0]
    d1=struct.unpack('f',f.read(4))[0]
    d2=struct.unpack('f',f.read(4))[0]
    d3=struct.unpack('f',f.read(4))[0]

    print 'icycle: ', icycle
    # print time,dt,cpu,icycle,d1,d2,d3, '\n'

    # Read integer values of tracer numbers, mats, tvars, mvars, and gvars

    for i in range(2):
        temp=struct.unpack('f',f.read(4))[0] # More spacer
    
    numtracers=struct.unpack('i',f.read(4))[0]
    nummats=struct.unpack('i',f.read(4))[0]
    numtvars=struct.unpack('i',f.read(4))[0]
    nummvars=struct.unpack('i',f.read(4))[0]
    numgvars=struct.unpack('i',f.read(4))[0]

    print 'Number of tracers: ', numtracers
    print 'Number of materials: ', nummats
    print 'Number of tracer variables', numtvars
    print 'Number of material variables', nummvars
    print 'Number of global variables', numgvars

    nt=1 #  initialize cube arrays
    tvars=np.zeros([numtvars,numtracers,nt])
    mvars=np.zeros([nummvars,nummats,nt])
    gvars=np.zeros([numgvars,nt])
    # print(np.shape(gvars))

    for i in range(2):
        temp=struct.unpack('f',f.read(4))[0] # More spacer
    
    time=struct.unpack('f',f.read(4))[0]
    dt=struct.unpack('f',f.read(4))[0]
    cpu=struct.unpack('f',f.read(4))[0]
    icycle=struct.unpack('i',f.read(4))[0]
    for i in range(numtvars):
        for j in range(numtracers):
            tvars[i,j,0]=struct.unpack('f',f.read(4))[0]

    for i in range(nummvars):
        for j in range(nummats):
            mvars[i,j,0]=struct.unpack('f',f.read(4))[0]

    for i in range(numgvars):
        gvars[i,0]=struct.unpack('f',f.read(4))[0]

    print 'icycle: ', icycle
    # print time,dt,cpu,icycle,gvars, '\n'

    # variable names
    varnames_len=(2*(16*numtvars+16*nummvars+16*numgvars))
    ipts=np.empty(numtracers+nummats+numtracers,dtype='int16')

    varnames=""
    for i in range(2):
        temp=struct.unpack('f',f.read(4))[0] # More space
    for i in range(varnames_len):
        varnames=varnames+struct.unpack('c',f.read(1))[0]
    
    # print 'Variable names:\n', varnames, '\n'

    for i in range(len(ipts)):
        ipts[i]=struct.unpack('i',f.read(4))[0]
    # print 'ipts:\n', ipts, '\n'

    # end Header information
    # Read in tracer
    # Begin reading in the tracer data for each time step
    # Number of dumps
    timearr=time
    cyclearr=icycle
    # tcube=np.empty([nt,numtvars,numtracers])
    tcube=tvars
    # mcube=np.empty([nt,nummvars,nummats])
    mcube=mvars
    # gcube=np.empty([nt,numgvars])
    gcube=gvars
    # print(np.shape(gcube))

    numdumps = 0


    while icycle != -101:
        for i in range(2):
            temp=struct.unpack('f',f.read(4))[0] # More spacer
        
        time=struct.unpack('f',f.read(4))[0]
        dt=struct.unpack('f',f.read(4))[0]
        cpu=struct.unpack('f',f.read(4))[0]
        icycle=struct.unpack('i',f.read(4))[0]  
    
        timearr = np.append(timearr,time)
        cyclearr = np.append(cyclearr,icycle)
        # print(time, dt, cpu, icycle)
        for i in range(numtracers):
            for j in range(numtvars):
                tvars[j,i,0]=struct.unpack('f',f.read(4))[0]

        for i in range(nummats):
            for j in range(nummvars):
                mvars[j,i,0]=struct.unpack('f',f.read(4))[0]

        for i in range(numgvars):
            gvars[i,0]=struct.unpack('f',f.read(4))[0]  
    
        tcube=np.append(tcube,tvars,axis=2)
        mcube=np.append(mcube,mvars,axis=2)
        gcube=np.append(gcube,gvars,axis=1)

        numdumps = numdumps + 1
        # print(np.size(tcube))
        # if numdumps > 40:
        # icycle=-101
    
    date=""
    for i in range(b_len):   
        date=date+struct.unpack('c',f.read(1))[0]

    # print 'Date:\n', date, '\n'
    print 'numdumps: ', numdumps

    varnames = varnames.split()
    last_i = -1
    for i, name in enumerate(varnames):
        if name == 'VOLUME':
            last_i = i

    if last_i == -1:
        exit('Error when retrieving variable names, VOLUME not found')

    return varnames[:last_i + 1], timearr, tcube, mcube, gcube


