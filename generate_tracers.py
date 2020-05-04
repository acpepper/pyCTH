'''
Script to generate tracers for target and projectile spheres

Created 05/01/2020
'''

import numpy as np



def compute_layer_points(r,dx):
    '''
    algorithm to place equidistant on sphere from:
    https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    r = radius of sphere
    dx = desired lateral spacing
    '''
    x = []
    y = []
    z = []
    V = []
    Ncount = 0
    Nlayer = np.int(4.*np.pi*r**2/dx**2)
    if Nlayer < 8:
        Nlayer = 8
    a = 4.*np.pi*r**2/Nlayer
    d = np.sqrt(a)
    Mtheta = np.int(np.pi*r/d)
    dtheta = np.pi*r/Mtheta
    dphi = a/dtheta
    for m in range(0,Mtheta):
        theta = np.pi*(m+0.5)/Mtheta
        Mphi = np.int(2.*np.pi*r*np.sin(theta)/dphi)
        for n in range(0,Mphi):
            phi = 2.*np.pi*n/Mphi
            Ncount += 1
            x.append(r*np.sin(theta)*np.cos(phi))
            y.append(r*np.sin(theta)*np.sin(phi))
            z.append(r*np.cos(theta))
    return x,y,z,Ncount

def compute_layer(rinner,router,dx):
    '''
    Given inner and outer radius, and spacing dx
    computes the x,y,z points and volume of each point
    '''
    r = (rinner+router)/2.
    x,y,z,Ncount = compute_layer_points(r,dx)
    Vpt = 4.*np.pi/3.*(router**3-rinner**3)/Ncount
    V = len(x)*[Vpt]
    return x,y,z,V


def build_planet_tracers(Rtc,Rt,Rpc,Rp,res_tc,res_tm,res_pc,res_pm,xoff=0,yoff=0,zoff=0):
    '''
    Builds tracers in planet
    Rtc,Rt,Rpc,Rp = target and projectile core and total radii
    res_* = spatial resolution of layers in core and mantle
    x,y,zoff = spatial offset of projectile
    '''
    ######################
    # compute target
    ######################
    print('Building planet')
    print('xoff,yoff,zoff = %g  %g  %g' % (xoff,yoff,zoff))
    print('Rtc,Rt,res_tc,res_tm = %g  %g  %g  %g' % (Rtc,Rt,res_tc,res_tm))
    print('Rpc,Rp,res_pc,res_pm = %g  %g  %g  %g' % (Rpc,Rp,res_pc,res_pm))

    print('Computing Target Core:')
    Nlayers = int(np.ceil(Rtc/res_tc))
    dr = Rtc/Nlayers
    xtc = []
    ytc = []
    ztc = []
    Vtc = []
    for i in range(Nlayers):
        rinner = i*dr
        router = (i+1)*dr
        x,y,z,V = compute_layer(rinner,router,dr)
        print('rinner,router,Npts: %.1f %.1f %i' % (rinner,router,len(x)))
        xtc = xtc + x
        ytc = ytc + y
        ztc = ztc + z
        Vtc = Vtc + V

    print('N = ',len(xtc),'\n')



    print('Computing Target Mantle:')
    Nlayers = int(np.ceil((Rt-Rtc)/res_tm))
    dr = (Rt-Rtc)/Nlayers
    xtm = []
    ytm = []
    ztm = []
    Vtm = []
    #for i in range(Nlayers):
    for i in range(Nlayers):
        rinner = Rtc+i*dr
        router = Rtc+(i+1)*dr
        x,y,z,V = compute_layer(rinner,router,dr)
        print('rinner,router,Npts: %.1f %.1f %i' % (rinner,router,len(x)))
        xtm = xtm + x
        ytm = ytm + y
        ztm = ztm + z
        Vtm = Vtm + V

    print('N = ',len(xtm),'\n')



    #####################
    # compute projectile
    #####################
    print('Computing Projectile Core:')
    Nlayers = int(np.ceil(Rpc/res_pc))
    dr = Rpc/Nlayers
    xpc = []
    ypc = []
    zpc = []
    Vpc = []
    #for i in range(Nlayers):
    for i in range(Nlayers):
        rinner = i*dr
        router = (i+1)*dr
        x,y,z,V = compute_layer(rinner,router,dr)
        print('rinner,router,Npts: %.1f %.1f %i' % (rinner,router,len(x)))
        xpc = xpc + x
        ypc = ypc + y
        zpc = zpc + z
        Vpc = Vpc + V

    print('N = ',len(xpc),'\n')


    print('Computing Projectile Mantle:')
    Nlayers = int(np.ceil((Rp-Rpc)/res_pm))
    dr = (Rp-Rpc)/Nlayers
    xpm = []
    ypm = []
    zpm = []
    Vpm = []
    #for i in range(Nlayers):
    for i in range(Nlayers):
        rinner = Rpc+i*dr
        router = Rpc+(i+1)*dr
        x,y,z,V = compute_layer(rinner,router,dr)
        print('rinner,router,Npts: %.1f %.1f %i' % (rinner,router,len(x)))
        xpm = xpm + x
        ypm = ypm + y
        zpm = zpm + z
        Vpm = Vpm + V

    print('N = ',len(xpm),'\n')



    print('\nTotal tracers:',len(xtc)+len(xtm)+len(xpc)+len(xpm))
    print('Target Core:',len(xtc))
    print('Target Mantle:',len(xtm))
    print('Projectile Core:',len(xpc))
    print('Projectile Mantle:',len(xpm))


    # correct projectile values
    xpc = list(np.array(xpc)+xoff)
    xpm = list(np.array(xpm)+xoff)
    ypc = list(np.array(ypc)+yoff)
    ypm = list(np.array(ypm)+yoff)
    zpc = list(np.array(zpc)+zoff)
    zpm = list(np.array(zpm)+zoff)


    # create labels array
    lpc = ['proj_core']*len(xpc)
    lpm = ['proj_mant']*len(xpm)
    ltc = ['targ_core']*len(xtc)
    ltm = ['targ_mant']*len(xtm)



    x = xtc + xtm + xpc + xpm
    y = ytc + ytm + ypc + ypm
    z = ztc + ztm + zpc + zpm
    V = Vtc + Vtm + Vpc + Vpm
    l = ltc + ltm + lpc + lpm
    return np.array(x),np.array(y),np.array(z),np.array(V),np.array(l)


def load_tracers(fname='tracers.npz'):
    '''
    Load tracer info for post processing
    Gives initial tracer positions, representative volumes, and 
      position in target/projectile core/mantle
    '''
    data = np.load(fname)
    xt = data['x']
    yt = data['y']
    zt = data['z']
    Vt = data['V']
    labels = data['label']
    return xt,yt,zt,Vt,labels

if __name__ == "__main__":

    # define target/projectile dimensions/resolution
    Rtc = 3400.
    Rt = 6370.
    Rpc = 535. # projectile core radius in km
    Rp = 1000.

    res_tc = 3400.
    res_tm = 500.
    res_pc = 200.
    res_pm = 200.


    # create tracers
    x,y,z,V,label = build_planet_tracers(Rtc,Rt,Rpc,Rp,res_tc,res_tm,res_pc,res_pm,zoff=Rt+Rp)

    # write tracers to file
    f = open('tracers.out','w')
    f.write('tracer\n')
    # write tracers line by line, converting km -> cm
    for i in range(np.size(x)):
        f.write('ADD %.0f, %.0f, %.0f\n' % (x[i]*1.e5,y[i]*1.e5,z[i]*1.e5)) 
    f.write('endtracer\n')
    f.write('**\n')
    f.write('*---------------------------------------------------------\n')
    f.write('endinput')

    f.close()


    np.savez('tracers.npz',x=x,y=y,z=z,V=V,label=label)

    # test loading tracers
    xt,yt,zt,Vt,labels = load_tracers()

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
##
#ax.scatter(xtc, ytc, ztc, c='r', marker='o')
##
#plt.show()


        



