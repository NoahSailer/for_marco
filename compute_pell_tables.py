import numpy as np
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from utils import speed_of_light,kvec,ref_dist,get_cosmo

def compute_pell_tables(pars, z=0.61, klin_max=10.,cutoff=2.):

    Hzfid, chizfid = ref_dist(z)
    cosmo = get_cosmo(pars,klin_max=klin_max)
    h = cosmo.h()

    ki = np.logspace(-3.0,np.log10(klin_max),200)
    pi = np.array( [cosmo.pk_cb(k*h, z ) * h**3 for k in ki] )
        
    # Caluclate AP parameters
    Hz = cosmo.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz = cosmo.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    apar, aperp = Hzfid / Hz, chiz / chizfid
    
    # Calculate growth rate
    f   = cosmo.scale_independent_growth_factor_f(z)

    # Now do the RSD
    modPT = LPT_RSD(ki, pi, kIR=0.2,\
                cutoff=cutoff, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
    modPT.make_pltable(f, kv=kvec, apar=apar, aperp=aperp, ngauss=3)
    
    # restrict the tables to the terms that we actually want
    # 1,b1,b1**2,b2,b1*b2,b2**2,bs,b1*bs,b2*bs,bs**2,alp0,alp2,sn0,sn2
    I = [0,1,2,3,4,5,6,7,8,9,12,13,16,17]
    p0t = modPT.p0ktable[:,I]
    p2t = modPT.p2ktable[:,I]
    p4t = modPT.p4ktable[:,I]

    return kvec, p0t, p2t, p4t    


def compute_binned_pell_tables(pars, fnameM, fnameW, z=0.61, klin_max=10.,cutoff=2.):
    
    kvec, p0ktable, p2ktable, p4ktable = compute_pell_tables(pars, z=z, klin_max=klin_max,cutoff=cutoff)
    
    # number of monomials
    Nmono = p0ktable.shape[-1] 
    # kvalues to evaluate theory on
    kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005; Nv = len(kv)
    # interpolate! concatenate P0,P2,P4
    thy = np.zeros( (3*Nv , Nmono) )
    for i in range(Nmono):
        thy[:Nv,i]       = Spline(kvec,p0ktable[:,i],ext=3)(kv)
        thy[Nv:2*Nv,i]   = Spline(kvec,p2ktable[:,i],ext=3)(kv)
        thy[2*Nv:3*Nv,i] = Spline(kvec,p4ktable[:,i],ext=3)(kv)
    
    # convolve
    M = np.loadtxt(fnameM)
    W = np.loadtxt(fnameW)
    expanded_model = np.matmul(M, thy )
    convolved_model = np.matmul(W, expanded_model )
    
    # only retain monopole and quad
    kdat = np.genfromtxt('data/kdat.txt')
    yeses = kdat > 0
    nos = kdat < 0
    mono = np.concatenate( (yeses, nos, nos, nos, nos ) )
    quad = np.concatenate( (nos, nos, yeses, nos, nos ) )  

    p0 = convolved_model[mono]
    p2 = convolved_model[quad]
    
    return kdat,p0,p2