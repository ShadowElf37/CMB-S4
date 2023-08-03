from classy import Class
import numpy as np
from fishchips.experiments import Experiment
import fishchips.util
import matplotlib.pyplot as plot
from multiprocessing import Pool

def white_noise(theta_fwhm=7., sigma_T=33., sigma_P=56., l_max=2500):
    arcmin_to_radian = np.pi / 60. / 180.
    theta_fwhm *= arcmin_to_radian
    sigma_T *= arcmin_to_radian
    sigma_P *= arcmin_to_radian

    l = np.arange(0, l_max+1)
    NT = sigma_T ** 2 * np.exp(l*(l+1) * theta_fwhm**2 / (8*np.log(2)))
    NP = sigma_P ** 2 * np.exp(l * (l + 1) * theta_fwhm ** 2 / (8 * np.log(2)))

    return {'tt': NT, 'ee': NP}

class CMB_S4(Experiment):
    def __init__(self, f_sky=0.65, l_min=2, l_max=2500, verbose=False, noise_curves={}):
        # NOISE SHOULD BE DIMENSIONFUL (µK^2)
        self.verbose = verbose

        self.f_sky = f_sky

        self.l_min = l_min
        self.l_max = l_max
        self.l = np.arange(self.l_min, self.l_max + 1)

        self.channels = 'te'

        self.noise = noise_curves
        self.T_cmb = 1

    def get_cov(self, cl):
        covmat = np.zeros((self.l_max+1, len(self.channels), len(self.channels)))
        for i in range(len(self.channels)):
            for j in range(0, i + 1):
                chan_name = self.channels[j] + self.channels[i]
                covmat[:, i, j] = cl[chan_name] + self.noise.get(chan_name, 0)/self.T_cmb
                covmat[:, j, i] = cl[chan_name] + self.noise.get(chan_name, 0)/self.T_cmb
        return covmat[self.l_min:]

    def get_dcov(self, inputs):
        derivatives = {}
        for i,p in enumerate(inputs.parameters):
            derivatives[p] = (self.get_cov(inputs.cl_right[p]) - self.get_cov(inputs.cl_left[p])) / (2*inputs.dx[i])
        return derivatives

    def get_fisher(self, inputs):
        T_cmb_normed = inputs.model_fid.T_cmb()
        self.T_cmb = (T_cmb_normed * 1.0e6) ** 2
        coeffs = (2 * self.l + 1) / 2 * self.f_sky

        covs = self.get_cov(inputs.cl_fid) * self.T_cmb

        print("making derivatives...")
        dcovs = self.get_dcov(inputs)
        dcovs = {k:v*self.T_cmb for k,v in dcovs.items()}
        invs = np.linalg.inv(covs)

        print('making fisher...')
        fisher = np.zeros((len(inputs.parameters), len(inputs.parameters)))
        for j in range(len(inputs.parameters)):
            for i in range(0, j+1):
                multiplied = np.matmul(np.matmul(invs, dcovs[inputs.parameters[i]]), np.matmul(invs, dcovs[inputs.parameters[j]]))
                fisher[i, j] = fisher[j, i] = np.dot(np.trace(multiplied, axis1=1, axis2=2), coeffs)
        return fisher

class Observables:
    def __init__(self, classy_template, params, fiducial, dx):
        self.parameters = params
        self.classy_params = classy_template | dict(zip(params, fiducial))

        self.fiducial = np.array(fiducial)
        self.dx = np.array(dx)

        self.model_fid = Class()
        self.model_fid.set(self.classy_params)
        self.cl_fid = None
        self.cl_left = {p:None for p in params}
        self.cl_right = {p: None for p in params}

    @classmethod
    def make_and_compute_cl(cls, params, l_max, lensed_cl):
        model = Class()
        model.set(params)
        model.compute()
        if lensed_cl:
            return model.lensed_cl(l_max)
        return model.raw_cl(l_max)

    def compute(self, l_max=2500, lensed_cl=True):
        with Pool(None) as pool:
            promises = []
            #print('spawning...')
            for i, p in enumerate(self.parameters):
                left = self.classy_params.copy()
                left[p] -= self.dx[i]
                promises.append(pool.apply_async(self.make_and_compute_cl, (left, l_max, lensed_cl)))

                right = self.classy_params.copy()
                right[p] += self.dx[i]
                promises.append(pool.apply_async(self.make_and_compute_cl, (right, l_max, lensed_cl)))

            #print('getting...')
            self.model_fid.compute()
            if lensed_cl:
                self.cl_fid = self.model_fid.lensed_cl(l_max)#pool.apply_async(self.make_and_compute_cl, self.classy_params, l_max, lensed_cl).get()
            else:
                self.cl_fid = self.model_fid.raw_cl(l_max)

            for i, p in enumerate(self.parameters):
                self.cl_left[p] = promises[2*i].get()
                self.cl_right[p] = promises[2*i+1].get()
            #print('done!')

"""
CV-limited Fisher matrix (same as fishchips!)
[[ 9.69794363e+23  1.72729118e+15 -3.95684717e+15]
 [ 1.72729118e+15  3.92743841e+06 -7.09200414e+06]
 [-3.95684717e+15 -7.09200414e+06  1.64363986e+07]]
 
White noise Fisher matrix (same as fishchips!)
[[ 1.92839065e+23  1.87259072e+14 -8.06537981e+14]
 [ 1.87259072e+14  3.91572241e+05 -7.89960544e+05]
 [-8.06537981e+14 -7.89960544e+05  3.43740135e+06]]
"""


if __name__ == "__main__":
    import time
    t1 = time.time()

    obs = Observables(
        classy_template={'output': 'tCl pCl lCl',
                'l_max_scalars': 2500,
                'lensing': 'yes'},
        params=['A_s', 'n_s', 'tau_reio'],
        fiducial=[2.1e-9, 0.968, 0.066],
        dx=[1.e-10, 2.e-02, 1.e-02]
    )

    print('fid')
    obs.compute(l_max=2500)

    experiment = CMB_S4(noise_curves=white_noise())
    print('fish')
    print(time.time() - t1)
    f = experiment.get_fisher(obs)
    print(f)

    print(time.time()-t1)

    cov = np.linalg.inv(f)
    fishchips.util.plot_triangle(obs, cov)
    plot.show()
