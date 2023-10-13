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

    return {'TT': NT, 'EE': NP}

class CMB_S4(Experiment):
    def __init__(self, f_sky=0.65, l_min=2, l_max=2500, verbose=False, noise_curves={}):
        # NOISE SHOULD BE DIMENSIONFUL (µK^2)
        self.verbose = verbose

        self.f_sky = f_sky

        self.l_min = l_min
        self.l_max = l_max
        self.l = np.arange(self.l_min, self.l_max + 1)

        self.channels = 'tep'

        self.noise = noise_curves
        self.T_cmb = 1

    def get_cov(self, cl):
        covmat = np.zeros((self.l_max+1, len(self.channels), len(self.channels)))
        for i in range(len(self.channels)):
            for j in range(0, i + 1):
                chan_name = self.channels[j] + self.channels[i]
                if chan_name in ['tp', 'ep']:
                    covmat[:, i, j] = covmat[:, j, i] = 0
                    continue
                covmat[:, i, j] = covmat[:, j, i] = cl[chan_name] + self.noise.get(chan_name.upper(), 0)/self.T_cmb
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






if __name__ == "__main__" and 0:
    import draft_modules.tools as tools

    els = np.arange(1, 100)
    nlfile1 = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/results/s4_cmb_ilc.npy'
    nlfile2 = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/products/20220726/s4wide_ilc_galaxy0_27-39-93-145-225-278_TT-EE_for7years.npy'

    nlfile1 = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/products/20220726/lensing_noise_curves/ilc/s4wide_lmin100_lmax5000.npy'

    nl_dic1, dic1 = tools.get_nldic(nlfile1, els)
    #nl_dic2, dic2 = tools.get_nldic(nlfile2, els)
    print(nl_dic1.keys())



if __name__ == "__main__" and 1:
    cls = Class()
    cls.set({'output': 'tCl pCl lCl',
            'l_max_scalars': 2000,
            'lensing': 'yes',
            'DM_annihilation_efficiency': 1.0e-7,
        })
    cls.compute()
    print(cls.lensed_cl(2000))

if __name__ == "__main__" and 0:
    import time
    t1 = time.time()

    LMAX = 100

    fid = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.06, 0.010409*100, 0])
    dx_left = np.array([0.0030,0.0008,0.1e-9,0.010,0.020,0.000050*100,0])
    dx_right = np.array([0.0030,0.0008,0.1e-9,0.010,0.020,0.000050*100,1.0e-5])

    obs = Observables(
        classy_template={
            'output': 'tCl pCl lCl',
            'l_max_scalars': LMAX,
            'lensing': 'no',
            'DM_annihilation_efficiency': 0,
            #'DM_annihilation_variation': 0,
            #'DM_annihilation_z': 600,
            #'DM_annihilation_zmax': 2500,
            #'DM_annihilation_zmin': 30
        },
        params=['omega_cdm', 'omega_b', 'A_s', 'n_s', 'tau_reio', '100*theta_s', 'DM_annihilation_efficiency'],
        fiducial=fid,
        dx_left=dx_left,
        dx_right=dx_right
    )

    import draft_modules.tools as tools
    els = np.arange(0, LMAX+1)
    nlfile = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/products/20220726/s4wide_ilc_galaxy0_27-39-93-145-225-278_TT-EE_for7years.npy'
    nl_dic = tools.get_nldic(nlfile, els)


    if 1:
        obs.compute(l_max=LMAX, lensed_cl=False)

        experiment = CMB_S4(noise_curves=nl_dic, l_max=LMAX)
        f = experiment.get_fisher(obs)

        print('Time elapsed:', time.time()-t1)

        cov = np.linalg.inv(f)

        import plot_triangle

        SCALES = [0, 0, 9, 0, 0, 2, 7]
        FID = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.06, 0.010409*100, 0])
        PANN = r'$p_{\mathrm{ann}}$ ($10^{-7}\mathrm{m}^3\mathrm{s}^{-1}\mathrm{kg}^{-1}$)'
        LABELS = [r'$\Omega_{DM} h^2$', r'$\Omega_b h^2$', r'$10^{9}A_s$', r'$n_s$', r'$\tau_{reio}$', r'$10^{2}\theta_s$',PANN]
        plot_triangle.triplot(LABELS, SCALES, FID, cov)

        plot.show()
