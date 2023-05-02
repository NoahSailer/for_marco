from compute_pell_tables import compute_binned_pell_tables
from compute_xiell_tables import compute_binned_xiell_tables
from utils import convert

# pars = [ln(A_s*1e10), n_s, h, omega_b, omega_cdm, xi_idr, np.log10(a_dark)]

def ptable_NGC_z1(pars,n=4,frac=1.):
   pars_ = convert(pars)+[n,frac]
   fnameM = 'data/M_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_1200_2000.matrix.gz'
   fnameW = 'data/W_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix.gz'
   k, p0t, p2t = compute_binned_pell_tables(pars_, fnameM, fnameW, z=0.38, klin_max=10., cutoff=2.)
   return k, p0t[:,:12], p2t[:,:12]
   
def ptable_NGC_z3(pars,n=4,frac=1.):
   pars_ = convert(pars)+[n,frac]
   fnameM = 'data/M_BOSS_DR12_NGC_z3_V6C_1_1_1_1_1_1200_2000.matrix.gz'
   fnameW = 'data/W_BOSS_DR12_NGC_z3_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix.gz'
   k, p0t, p2t = compute_binned_pell_tables(pars_, fnameM, fnameW, z=0.59, klin_max=10., cutoff=2.)
   return k, p0t[:,:12], p2t[:,:12]

def ptable_SGC_z1(pars,n=4,frac=1.):
   pars_ = convert(pars)+[n,frac]
   fnameM = 'data/M_BOSS_DR12_SGC_z1_V6C_1_1_1_1_1_1200_2000.matrix.gz'
   fnameW = 'data/W_BOSS_DR12_SGC_z1_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix.gz'
   k, p0t, p2t = compute_binned_pell_tables(pars_, fnameM, fnameW, z=0.38, klin_max=10., cutoff=2.)
   return k, p0t[:,:12], p2t[:,:12]
   
def ptable_SGC_z3(pars,n=4,frac=1.):
   pars_ = convert(pars)+[n,frac]
   fnameM = 'data/M_BOSS_DR12_SGC_z3_V6C_1_1_1_1_1_1200_2000.matrix.gz'
   fnameW = 'data/W_BOSS_DR12_SGC_z3_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix.gz'
   k, p0t, p2t = compute_binned_pell_tables(pars_, fnameM, fnameW, z=0.59, klin_max=10., cutoff=2.)
   return k, p0t[:,:12], p2t[:,:12]
   
def xitable_z1(pars,n=4,frac=1.):
   pars_ = convert(pars)+[n,frac]
   r, xi0t, xi2t = compute_binned_xiell_tables(pars_, z=0.38, klin_max=12.)
   return r, xi0t[:,:6], xi2t[:,:6]
   
def xitable_z3(pars,n=4,frac=1.):
   pars_ = convert(pars)+[n,frac]
   r, xi0t, xi2t = compute_binned_xiell_tables(pars_, z=0.59, klin_max=12.)
   return r, xi0t[:,:6], xi2t[:,:6]   