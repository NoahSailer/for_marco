import numpy as np
from classy import Class

speed_of_light = 2.99792458e5

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )

def convert(pars):
   # the input is assumed to be of the form
   # pars = [ln(A_s*1e10), n_s, h, omega_b, omega_cdm, xi_idr, np.log10(a_dark)]
   pars = np.array(pars)
   tau_reio = 0.06
   result = np.array(list(pars[:5]) + [tau_reio] + list(pars[-2:]))
   result[0] = np.exp(pars[0]) * 1e-10
   result[-1] = 10**pars[-1] 
   # returns a result of the form
   # [A_s, n_s, h, omega_b, omega_cdm, tau_reio, xi_idr, a_dark]
   return list(result)


def ref_dist(z):
   # Reference Cosmology:
   Omega_M = 0.31
   fb = 0.1571
   h = 0.6766
   ns = 0.9665

   pkparams = {'A_s': np.exp(3.040)*1e-10,
               'n_s': 0.9665,
               'h': h,
               'N_ur': 3.046,
               'N_ncdm': 0,
               'tau_reio': 0.0568,
               'omega_b': h**2 * fb * Omega_M,
               'omega_cdm': h**2 * (1-fb) * Omega_M}

   pkclass = Class()
   pkclass.set(pkparams)
   pkclass.compute()

   Hz_fid = pkclass.Hubble(z) * speed_of_light / h 
   chiz_fid = pkclass.angular_distance(z) * (1.+z) * h 
   
   return Hz_fid,chiz_fid


def get_cosmo(pars,klin_max=10.):
    
    A_s, n_s, h, omega_b, omega_cdm, tau_reio, xi_idr, a_dark, n_idm_dr, frac = pars

    params = {
        'A_s': A_s,
        'n_s': n_s,
        'h': h,
        'N_ur': 1.0196,
        'N_ncdm': 2,
        'm_ncdm': '0.01,0.05',
        'tau_reio': tau_reio,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'a_idm_dr': a_dark,
        'xi_idr': xi_idr,
        'nindex_idm_dr': n_idm_dr,
        'f_idm': frac}
    
    pert_params = {'output': 'mPk',
                   'P_k_max_h/Mpc': klin_max,
                   'z_pk': '0,1'}

    params = {**pert_params,**params}
   
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    return cosmo