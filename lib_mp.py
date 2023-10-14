from classy import Class
import numpy as np
from fishchips.experiments import Experiment
import fishchips.util
import matplotlib.pyplot as plot
from multiprocessing import Pool


class Noise:
    CHANNELS = 'TT', 'EE', 'PP', 'TE'

    def __init__(self, l_max=2500, **from_dict):
        self.l = np.arange(0, l_max+1)
        self.l_max = l_max
        self.N = {chan: from_dict.get(chan, np.zeros_like(self.l))[0:l_max + 1] for chan in self.CHANNELS}

    def __getitem__(self, item):
        return self.N.get(item, 0)

    def __add__(self, other):
        # adds inverse variances
        assert self.l_max == other.l_max
        new_dict = {chan: 1/(1/self[chan] + 1/other[chan]) for chan in self.CHANNELS}
        return Noise(l_max=self.l_max, **new_dict)

    @staticmethod
    def white_noise(theta_fwhm=(7.,), sigma_T=(33.,), sigma_P=(56.,), l_max=2500):
        theta_fwhm = np.array(theta_fwhm, dtype=float)
        sigma_T = np.array(sigma_T, dtype=float)
        sigma_P = np.array(sigma_P, dtype=float)

        if len(theta_fwhm) != len(sigma_T):
            raise ValueError('Theta and at least sigma_T need matching lengths')

        arcmin_to_radian = np.pi / 60. / 180.
        theta_fwhm *= arcmin_to_radian
        sigma_T *= arcmin_to_radian
        sigma_P *= arcmin_to_radian

        l = np.arange(0, l_max+1)
        NT = np.zeros_like(l, dtype=float)
        NP = np.zeros_like(l, dtype=float)
        for i in range(len(sigma_T)):
            NT += sigma_T[i] ** 2 * np.exp(l*(l+1) * theta_fwhm[i] ** 2 / (8*np.log(2)))
        for i in range(len(sigma_P)):
            NP += sigma_P[i] ** 2 * np.exp(l * (l + 1) * theta_fwhm[i] ** 2 / (8 * np.log(2)))

        return Noise(TT=NT, EE=NP, l_max=l_max)

class CMB_S4(Experiment):
    def __init__(self, f_sky=0.65, l_min=2, l_max=2500, verbose=False, noise_curves: Noise=None):
        # NOISE SHOULD BE DIMENSIONFUL (µK^2)
        self.verbose = verbose

        self.f_sky = f_sky

        self.l_min = l_min
        self.l_max = l_max
        self.l = np.arange(self.l_min, self.l_max + 1)

        self.channels = 'TEP'

        self.noise = noise_curves or Noise(l_max=l_max)
        self.T_cmb = 1

    def get_cov(self, cl):
        covmat = np.zeros((self.l_max+1, len(self.channels), len(self.channels)))
        for i in range(len(self.channels)):
            for j in range(0, i + 1):
                chan_name = self.channels[j] + self.channels[i]
                if chan_name in ['TP', 'EP']:
                    covmat[:, i, j] = covmat[:, j, i] = 0
                    continue
                covmat[:, i, j] = covmat[:, j, i] = cl[chan_name.lower()] + self.noise[chan_name]/self.T_cmb
        return covmat[self.l_min:]

    def get_dcov(self, inputs):
        derivatives = {}
        for i,p in enumerate(inputs.parameters):
            derivatives[p] = (self.get_cov(inputs.cl_right[p]) - self.get_cov(inputs.cl_left[p])) / (inputs.dx_right[i] + inputs.dx_left[i])
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
    def __init__(self, classy_template, params, fiducial, dx_left, dx_right):
        self.parameters = params
        self.classy_params = classy_template | dict(zip(params, fiducial))
        print(self.classy_params)

        self.fiducial = np.array(fiducial)
        self.dx_left = np.array(dx_left)
        self.dx_right = np.array(dx_right)

        self.model_fid = Class()
        self.model_fid.set(self.classy_params)
        self.cl_fid = None
        self.cl_left = {p:None for p in params}
        self.cl_right = {p: None for p in params}

    @classmethod
    def make_and_compute_cl(cls, params: dict, l_max=2500, lensed_cl=True):
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
                left[p] -= self.dx_left[i]
                promises.append(pool.apply_async(self.make_and_compute_cl, (left, l_max, lensed_cl)))

                right = self.classy_params.copy()
                right[p] += self.dx_right[i]
                promises.append(pool.apply_async(self.make_and_compute_cl, (right, l_max, lensed_cl)))

            #print('getting...')
            self.model_fid.compute()
            if lensed_cl:
                self.cl_fid = self.model_fid.lensed_cl(l_max)#pool.apply_async(self.make_and_compute_cl, self.classy_params, l_max, lensed_cl).get()
            else:
                self.cl_fid = self.model_fid.raw_cl(l_max)

            for i, p in enumerate(self.parameters):
                self.cl_left[p] = promises[2*i].get()
                print(p)
                self.cl_right[p] = promises[2*i+1].get()
            #print('done!')






if __name__ == "__main__" and 1:
    import draft_modules.tools as tools

    els = np.arange(1, 100)
    nlfile1 = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/results/s4_cmb_ilc.npy'
    nlfile2 = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/products/20220726/s4wide_ilc_galaxy0_27-39-93-145-225-278_TT-EE_for7years.npy'

    #nlfile1 = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/products/20220726/lensing_noise_curves/ilc/s4wide_lmin100_lmax5000.npy'

    nl_dic1 = tools.get_nldic(nlfile1, els)
    #nl_dic2, dic2 = tools.get_nldic(nlfile2, els)
    print(nl_dic1.keys())

    ppfile = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/products/20220726/lensing_noise_curves/ilc/s4wide_lmin100_lmax5000.npy'
    ppf = np.load(ppfile, allow_pickle=1, encoding='latin1')
    nl_dic2 = ppf.item()
    print(nl_dic2.keys())



    print(nl_dic2['cl_kk'][:200])
    print(nl_dic2['els'])




if __name__ == "__main__" and 0:
    cls = Class()
    cls.set({'output': 'tCl pCl lCl',
            'l_max_scalars': 2000,
            'lensing': 'yes',
            'DM_annihilation_efficiency': 0,
        })
    cls.compute()
    print(cls.lensed_cl(2000))
