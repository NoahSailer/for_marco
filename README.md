**Running the code:** Clone this repository and run the following (from within ```for_marco/```) to get the theory predictions:

```
from wrapper_for_marco import *

# ln(A_s*1e10), n_s, h, omega_b, omega_cdm, xi_idr, np.log10(a_dark)
pars = [2.75, 0.9665, 0.6921, 0.02242, 0.1192, 0.001, 4]

# power spectra
# NGC
k, p0t_NGC_z1, p2t_NGC_z1 = ptable_NGC_z1(pars)
k, p0t_NGC_z3, p2t_NGC_z3 = ptable_NGC_z3(pars)

# SGC
k, p0t_SGC_z1, p2t_SGC_z1 = ptable_SGC_z1(pars)
k, p0t_SGC_z3, p2t_SGC_z3 = ptable_SGC_z3(pars)

# correlation functions
r, xi0t_z1, xi2t_z1 = xitable_z1(pars)
r, xi0t_z3, xi2t_z3 = xitable_z3(pars)
```

We want to emulate **every entry** in the above tables (e.g. ```p0t_NGC_z1```, ```xi0t_z3```). Ideally the emulator returns a numpy array with the same shape as the tables.

-------

**More information:**

Each power spectrum table (e.g. ```p0t_NGC_z1```) has shape ```(len(k),N_polyspectra)=(40, 12)```. The 12 polyspectra correspond to the coefficients: 
```
1,b1,b1**2,b2,b1*b2,b2**2,bs,b1*bs,b2*bs,bs**2,alp0,alp2
```

Each correlation function table (e.g. ```xi0t_z3```) has shape ```(len(r),N_polyspectra)=(36, 6)```. The 6 polyspectra correspond to the coefficients: 
```
1,B1,F,B1**2,F**2,B1*F
```

I have removed all trivial polyspectra (e.g. terms proportional to ```bs```).