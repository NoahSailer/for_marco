#!/usr/bin/env python3
#
# Code to compute angular power spectra using Limber's approximation,
# ignoring higher-order corrections such as curved sky or redshift-space
# distortions (that predominantly affect low ell).
#
import numpy as np

from scipy.integrate   import simps
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from classy import Class
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from utils import speed_of_light,kvec,ref_dist,get_cosmo

dndz_51 = np.genfromtxt('data/gal_s51_dndz.txt')
dndz_53 = np.genfromtxt('data/gal_s53_dndz.txt')
wlx = np.genfromtxt('data/wlx_NGC_pr4_apod.txt')
ll = np.genfromtxt('data/ell.txt')

class AngularPowerSpectra():
    """Computes angular power spectra using the Limber approximation."""        
        
    def mag_bias_kernel(self,s,Nchi_mag=101):
        """Returns magnification bias kernel if 's' is the slope of
           the number counts dlog10N/dm."""
        zval    = self.zchi(self.chival)
        cmax    = np.max(self.chival) * 1.1
        zupper  = lambda x: np.linspace(x,cmax,Nchi_mag)
        chivalp = np.array(list(map(zupper,self.chival))).transpose()
        zvalp   = self.zchi(chivalp)
        dndz_n  = np.interp(zvalp,self.zz,self.dndz,left=0,right=0)
        Ez      = self.Eofz(zvalp)
        g       = (chivalp-self.chival[np.newaxis,:])/chivalp
        g      *= dndz_n*Ez/2997.925
        g       = self.chival * simps(g,x=chivalp,axis=0)
        mag_kern= 1.5*(self.OmM)/2997.925**2*(1+zval)*g*(5*s-2.)
        return(mag_kern)
        
        
    def shot3to2(self):
        """Returns the conversion from 3D shotnoise to 2D shotnoise power."""
        Cshot = self.fchi**2/self.chival**2
        Cshot = simps(Cshot,x=self.chival)
        return(Cshot)
        
    def __init__(self,pars,dndz,zeff,Nchi=201,Nz=251,cutoff=2.,klin_max=10.):
        """Set up the class.
            OmM:  The value of Omega_m(z=0) for the cosmology.
            chils:The (comoving) distance to last scattering (in Mpc/h).
            dndz: A numpy array (Nbin,2) containing dN/dz vs. z.
            zeff: The 'effective' redshift for computing P(k)."""
        self.cutoff = cutoff
        cosmo = get_cosmo(pars,klin_max=klin_max)    
        self.cosmo = cosmo
        h = cosmo.pars['h']
        zls = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
        chils = cosmo.comoving_distance(zls)*h
        # Copy the arguments, setting up the z-range.
        self.Nchi = Nchi
        self.OmM = cosmo.get_current_derived_parameters(['Omega_m'])['Omega_m']
        self.zmin = np.min([0.05,dndz[0,0]])
        self.zmax = dndz[-1,0]
        self.zz   = np.linspace(self.zmin,self.zmax,Nz)
        self.dndz = Spline(dndz[:,0],dndz[:,1],ext=1)(self.zz)
        # Normalize dN/dz.
        self.dndz = self.dndz/simps(self.dndz,x=self.zz)
        # Make LCDM class and spline for E(z).
        zgrid     = np.logspace(0,3.1,128)-1.0
        EE = lambda z: cosmo.Hubble(z) * speed_of_light / h / 100.
        self.Eofz = Spline(zgrid,[EE(zz) for zz in zgrid])
        # Set up the chi(z) array and z(chi) spline.
        self.chiz = np.array([cosmo.comoving_distance(z) for z in self.zz])*h
        self.zchi = Spline(self.chiz,self.zz)
        # Work out W(chi) for the objects whose dNdz is supplied.
        chimin    = np.min(self.chiz)
        chimax    = np.max(self.chiz)
        self.chival= np.linspace(chimin,chimax,Nchi)
        zval      = self.zchi(self.chival)
        self.fchi = Spline(self.zz,self.dndz*self.Eofz(self.zz))(zval)
        self.fchi/= simps(self.fchi,x=self.chival)
        # and W(chi) for the CMB
        self.chistar= chils
        self.fcmb = 1.5*self.OmM*(1.0/2997.925)**2*(1+zval)
        self.fcmb*= self.chival*(self.chistar-self.chival)/self.chistar
        # Set the effective redshift.
        self.zeff = zeff
        # and save linear growth.
        DD = lambda z: cosmo.scale_independent_growth_factor(z)
        self.ld2  = (np.array([DD(zz) for zz in zval])/DD(zeff))**2
        #
        self.preal_tables(pars,z=zeff,kmax_lin=klin_max)
        # compute Pmm with halofit
        khfi = np.logspace(-3.0,np.log10(klin_max),200)
        phfi = np.array( [cosmo.pk(k*h, zeff ) * h**3 for k in khfi] )
        self.Pmm = Spline(khfi,phfi,ext=1) # Extrap. w/ zeros.
        
        
  
        
    def __call__(self,bparsX,smag=0.4,Nell=64,Lmax=1001):
        """Computes C_l^{kg} given the emulator for P_{ij}, the
           cosmological parameters (cpars) plus bias params for cross (bparsX) 
           and the magnification slope
           (smag)."""
        # Set up arrays to hold kernels for C_l.
        ell    = np.logspace(1,np.log10(Lmax),Nell) # More ell's are cheap.
        Ckg = np.zeros( (Nell,self.Nchi) )
        # The magnification bias kernel.
        fmag   = self.mag_bias_kernel(smag)
        # Fit splines to our P(k).  The spline extrapolates as needed.
        kgm,pgm = self.get_spectrum(bparsX)
        Pgm    = Spline(kgm,pgm)
        # Work out the integrands for C_l^{gg} and C_l^{kg}.
        for i,chi in enumerate(self.chival):
            kval     = (ell+0.5)/chi        # The vector of k's.            
            f1f2     = self.fchi[i]*self.fcmb[i]/chi**2 * Pgm(kval)
            m1f2     =      fmag[i]*self.fcmb[i]/chi**2 * self.Pmm(kval)*self.ld2[i]
            Ckg[:,i] = f1f2 + m1f2
        # and then just integrate them.
        Ckg = simps(Ckg,x=self.chival,axis=-1)
        # Now interpolate onto a regular ell grid.
        lval= np.arange(Lmax)
        Ckg = Spline(ell,Ckg)(lval)
        return lval,Ckg
    
    
    def preal_tables(self,pars,z=0.61,kmax_lin=10.):
        # use velocileptors to get the real-space P(k)
        cosmo = self.cosmo
    
        # Calculate power spectrum
        ki = np.logspace(-3.0,np.log10(kmax_lin),200)
        h = cosmo.pars['h']
        pi_cb = np.array( [cosmo.pk_cb_lin(k*h, z ) * h**3 for k in ki] )
        pi_m = np.array( [cosmo.pk_lin(k*h, z ) * h**3 for k in ki] )
        pi = np.sqrt(pi_cb * pi_m) 
        
        # Now do the PT
        f = cosmo.scale_independent_growth_factor_f(z)
        modPT = LPT_RSD(ki, pi, kIR=0.2, jn=5,cutoff=self.cutoff, extrap_min = -4, extrap_max = 3, N = 2000, threads=1)
        modPT.make_ptable(f, 0, kv=kvec)
        self.ptab = modPT.pktables[0]      
     
               
    def get_spectrum(self,bpars):   
        b1,b2,bs,alpha = bpars
        b3,sn = 0,0
        bias_monomials = np.array([1, 0.5*b1, 0,\
                               0.5*b2, 0, 0,\
                               0.5*bs, 0, 0, 0,\
                               0.5*b3, 0])  
        za   = self.ptab[:,-1]
        # the first row is kv, last row is za for countrterm
        res = np.sum(self.ptab[:,1:-1] * bias_monomials,axis=1)\
              + alpha * kvec**2 * za + sn
        return kvec,res
    
def get_Cl_table(aps):
   
   bpars = np.zeros(4) # b1,b2,bs,alpha
   ell,clgk_1 = aps(bpars,smag=0.,Lmax=1251)

   _,clgk_s = aps(bpars,smag=1.,Lmax=1251)
   clgk_s -= clgk_1
   
   bpars = np.array([1,0,0,0])
   _,clgk_b1 = aps(bpars,smag=0.,Lmax=1251)
   clgk_b1 -= clgk_1
   
   bpars = np.array([0,1,0,0])
   _,clgk_b2 = aps(bpars,smag=0.,Lmax=1251)
   clgk_b2 -= clgk_1
   
   bpars = np.array([0,0,1,0])
   _,clgk_bs = aps(bpars,smag=0.,Lmax=1251)
   clgk_bs -= clgk_1
   
   bpars = np.array([0,0,0,1])
   _,clgk_alp = aps(bpars,smag=0.,Lmax=1251)
   clgk_alp -= clgk_1
   
   return ell,np.array([clgk_1,clgk_b1,clgk_b2,clgk_bs,clgk_alp,clgk_s]).T 


def get_Clt_z1(pars):
   aps = AngularPowerSpectra(pars,dndz_51,0.38,klin_max=20.,cutoff=5)
   ell, Clt = get_Cl_table(aps)
   return ell, Clt

def get_Clt_z3(pars):
   aps = AngularPowerSpectra(pars,dndz_53,0.59,klin_max=20.,cutoff=5)
   ell, Clt = get_Cl_table(aps)
   return ell, Clt

def get_binned_Clt_z1(pars):
   ell, Clt = get_Clt_z1(pars)
   Clt_new = np.zeros((wlx.shape[1],Clt.shape[1]))
   Clt_new[:len(ell),:] = Clt.copy()
   Clt_new = np.matmul(wlx,Clt_new)
   return ll, Clt_new
   
def get_binned_Clt_z3(pars):
   ell, Clt = get_Clt_z3(pars)
   Clt_new = np.zeros((wlx.shape[1],Clt.shape[1]))
   Clt_new[:len(ell),:] = Clt.copy()
   Clt_new = np.matmul(wlx,Clt_new)
   return ll, Clt_new