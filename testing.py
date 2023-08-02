from classy import Class
import numpy as np
from fishchips.experiments import CMB_Primary
from fishchips.cosmo import Observables
import fishchips.util

parameters = {
    'omega_b':0.0223828,
    'omega_cdm':0.1201075,
    'h':0.67810,
    'A_s':2.100549e-09,
    'n_s':0.9660499,
    'tau_reio':0.05430842,
}

obs = Observables(**dict(zip(('fiducial', 'left', 'right'), *zip(*tuple(parameters)))))

class_settings =     {
    'output':'tCl,pCl,lCl,mPk',
    'lensing':'yes',
    'P_k_max_1/Mpc':3.0
}

print("Building model...")
lcdm = Class()
lcdm.set(parameters)
lcdm.compute()
print("Done!")

print(lcdm.comoving_distance(1))