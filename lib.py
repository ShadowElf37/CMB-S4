from classy import Class
import numpy as np
from fishchips.experiments import CMB_Primary, Experiment
from fishchips.cosmo import Observables
import fishchips.util
import matplotlib.pyplot as plot
from multiprocessing import Pool

# TODO: noise, mp, plotting, modularize


class CMB_S4(Experiment):
    def __init__(self, f_sky=0.65, l_min=2, l_max=2500, verbose=False, noise_curves={}):
        # no noise for now
        self.verbose = verbose

        self.f_sky = f_sky

        self.l_min = l_min
        self.l_max = l_max
        self.l = np.arange(self.l_min, self.l_max + 1)

        self.channels = 'te'

        self.noise = noise_curves

    def get_cov(self, model, lensed_cl=True):
        if lensed_cl:
            cl = model.lensed_cl(self.l_max)
        else:
            cl = model.raw_cl(self.l_max)

        covmat = np.zeros((self.l_max+1, len(self.channels), len(self.channels)))
        for i in range(len(self.channels)):
            for j in range(0, i + 1):
                chan_name = self.channels[j] + self.channels[i]
                covmat[:, i, j] = cl[chan_name] + self.noise.get(chan_name, 0)
                covmat[:, j, i] = cl[chan_name] + self.noise.get(chan_name, 0)
        return covmat[self.l_min:]

    def get_dcov(self, inputs):
        derivatives = {}
        for i,p in enumerate(inputs.parameters):
            derivatives[p] = (self.get_cov(inputs.models_right[p]) - self.get_cov(inputs.models_left[p])) / (2*inputs.dx[i])
        return derivatives

    def get_fisher(self, inputs, lensed_cl=True):
        T_cmb_normed = inputs.model_fid.T_cmb()
        T_cmb = (T_cmb_normed * 1.0e6) ** 2
        coeffs = (2 * self.l + 1) / 2 * self.f_sky

        covs = self.get_cov(inputs.model_fid) * T_cmb

        print("making derivatives...")
        dcovs = self.get_dcov(inputs)
        dcovs = {k:v*T_cmb for k,v in dcovs.items()}
        invs = np.linalg.inv(covs)

        print('making fisher...')
        fisher = np.zeros((len(inputs.parameters), len(inputs.parameters)))
        for j in range(len(inputs.parameters)):
            for i in range(0, j+1):
                multiplied = np.matmul(np.matmul(invs, dcovs[inputs.parameters[i]]), np.matmul(invs, dcovs[inputs.parameters[j]]))
                fisher[i, j] = fisher[j, i] = np.dot(np.trace(multiplied, axis1=1, axis2=2), coeffs)
        return fisher

class Inputs:
    def __init__(self, classy_template, params, fiducial, dx):
        self.params = params
        self.classy_params = classy_template | dict(zip(params, fiducial))

        self.fiducial = np.array(fiducial)
        self.dx = np.array(dx)

        self.model_fid = Class()
        self.models_left = {p:Class() for p in params}
        self.models_right = {p:Class() for p in params}

        # set params
        self.model_fid.set(self.classy_params)
        for i,p in enumerate(self.params):
            left = self.classy_params.copy()
            left[p] -= self.dx[i]
            self.models_left[p].set(left)

            right = self.classy_params.copy()
            right[p] += self.dx[i]
            self.models_right[p].set(right)

    def compute(self):
        self.compute_fid()
        self.compute_rightleft()

    def compute_fid(self):
        self.model_fid.compute()

    def compute_rightleft(self):
        for model in self.models_left.values():
            model.compute()
        for model in self.models_right.values():
            model.compute()



if __name__ == "__main__":
    import time
    t1 = time.time()

    obs = Inputs(
        classy_template={'output': 'tCl pCl lCl',
                'l_max_scalars': 2500,
                'lensing': 'yes'},
        params=['A_s', 'n_s', 'tau_reio'],
        fiducial=[2.1e-9, 0.968, 0.066],
        dx=np.array([1.e-10, 2.e-02, 1.e-02])
    )

    print('fid')
    obs.compute()

    experiment = CMB_S4()
    print('fish')
    print(experiment.get_fisher(obs))

    print(time.time()-t1)