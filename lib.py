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
        #print(cl)

        covmat = np.zeros((self.l_max+1, len(self.channels), len(self.channels)))
        for i in range(len(self.channels)):
            for j in range(0, i + 1):
                chan_name = self.channels[j] + self.channels[i]
                covmat[:, i, j] = cl[chan_name] + self.noise.get(chan_name, 0)
                covmat[:, j, i] = cl[chan_name] + self.noise.get(chan_name, 0)
        return covmat[self.l_min:]

    def get_dcov(self, inputs):
        derivatives = {}
        for i,p in enumerate(inputs.params):
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
        fisher = np.zeros((len(inputs.params), len(inputs.params)))
        for j in range(len(inputs.params)):
            for i in range(0, j+1):
                multiplied = np.matmul(np.matmul(invs, dcovs[inputs.params[i]]), np.matmul(invs, dcovs[inputs.params[j]]))
                fisher[i, j] = fisher[j, i] = np.dot(np.trace(multiplied, axis1=1, axis2=2), coeffs)
        return fisher

class Model:
    def __init__(self, params, channels='te'):
        self.params = params
        self.channels = channels
        self.model = Class()
        self.model.set(params)
        self.model.compute()
    def clean(self):
        self.model.struct_cleanup()
        self.model.empty()

    def covmats(self, noise=dict(), l_max=2000):
        # noise should be {channel:array}
        cl = self.model.lensed_cl(l_max)
        print(cl)
        covmat = np.zeros((l_max+1, len(self.channels), len(self.channels)))
        for i in range(len(self.channels)):
            for j in range(0,i+1):
                chan_name = self.channels[j]+self.channels[i]
                covmat[:, i, j] = cl[chan_name] + noise.get(chan_name, 0)
                covmat[:, j, i] = cl[chan_name] + noise.get(chan_name, 0)
        return covmat

    def d_covmat(self, dparam, noise=dict(), stepsize=0.01, l_min=2, l_max=2000):
        left = self.params.copy()
        left[dparam] -= stepsize
        mleft = Model(left).covmats(noise, l_max=l_max)

        right = self.params.copy()
        right[dparam] += stepsize
        mright = Model(right).covmats(noise, l_max=l_max)

        return ((mright - mleft) / (2*stepsize))[l_min:]


    def fisher(self, inputs=('omega_b', 'omega_cdm', 'h'), f_sky=1, l_min=2, l_max=2000):
        coeffs = (2*np.arange(l_min, l_max+1)+1)/2 * f_sky
        covs = self.covmats()[l_min:]
        invs = np.linalg.inv(covs)
        print("making derivatives...")
        derivs = {inp: self.d_covmat(inp) for inp in inputs}

        print('making fishers...')
        fisher = np.zeros((len(inputs), len(inputs)))
        for i in range(len(inputs)):
            for j in range(len(inputs)):
                step1 = np.matmul(invs, derivs[inputs[j]])
                step2 = np.matmul(derivs[inputs[i]], step1)
                step3 = np.matmul(invs, step2)
                step4 = np.trace(step3, axis1=1, axis2=2)
                fisher[i, j] = np.sum(step4 * coeffs)
        return fisher

class Inputs:
    def __init__(self, classy_template, params, fiducial, dx, compute_mp=False):
        self.params = params
        self.classy_params = classy_template | dict(zip(params, fiducial))
        self.compute_mp = compute_mp

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

    def compute_fid(self):
        self.model_fid.compute()

    def compute_rightleft(self):
        if self.compute_mp:
            with Pool(None) as pool:
                for model in self.models_left.values():
                    pool.apply_async(model.compute)
                for model in self.models_right.values():
                    pool.apply_async(model.compute)
                pool.close()
                pool.join()
        else:
            for model in self.models_left.values():
                model.compute()
            for model in self.models_right.values():
                model.compute()


"""
OFFICIAL 18.2s
[[ 9.69010597e+23  1.72513230e+15 -3.95366149e+15]
 [ 1.72513230e+15  3.92145650e+06 -7.08319138e+06]
 [-3.95366149e+15 -7.08319138e+06  1.64234074e+07]]

"""

"""
UNOFFICIAL 17.9s
[[ 9.69794363e+23  1.72729118e+15 -3.95684717e+15]
 [ 1.72729118e+15  3.92743841e+06 -7.09200414e+06]
 [-3.95684717e+15 -7.09200414e+06  1.64363986e+07]]
 
[[ 9.69808507e+23  1.72716250e+15 -3.94810141e+15]
 [ 1.72716250e+15  3.92674236e+06 -7.07550820e+06]
 [-3.94810141e+15 -7.07550820e+06  1.63612089e+07]]
"""

"""
produced equal covs (please verify for all?)

contrib at 998 official
0 0 2.9173303374270546e+20
0 1 303949301169.05206
0 2 -1229470744078.2896
1 1 319.8539391879595
1 2 -1277.0941990790868
2 2 5188.434750574307

0 0 2.9173303374270582e+20
0 1 303949301169.0525
0 2 -1229470744078.29
1 1 319.85393918796
1 2 -1277.0941990790877
2 2 5188.434750574306


dlib
[[3204994.92761979  -68700.01495644]
 [ -68700.01495644  118075.50803467]]
 
doff
[[3204994.92761978  -68700.01495644]
 [ -68700.01495644  118075.50803467]]


"""

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
    obs.compute_fid()

    experiment = CMB_S4()
    print('fish')
    print(experiment.get_fisher(obs))

    print(time.time()-t1)