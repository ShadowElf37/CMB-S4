from classy import Class
import numpy as np
from fishchips.experiments import CMB_Primary
from fishchips.cosmo import Observables
import fishchips.util
import matplotlib.pyplot as plot
from lib import *

params = {
    'output': 'tCl lCl',
    'l_max_scalars': 2000,
    'lensing': 'yes',
    'omega_cdm': 0.120,
    'omega_b': 0.0224,
    'h': 0.674
}

h_step = 0.01
left_params = params.copy()
left_params['h'] = params['h'] - h_step
right_params = params.copy()
right_params['h'] = params['h'] + h_step

model = Model(params)
# get the C_l^TT and then compute the derivative!
fiducial = model.get_cl('tt')
dCltt_dh = model.get_dcl('tt', 'h')

plot.plot(dCltt_dh / fiducial)
plot.ylabel(r'$(\partial C_{\ell}^{TT} / \partial h) / C_{\ell}^{TT}$')
plot.xlabel(r'$\ell$')

plot.show()