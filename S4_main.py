import time
import numpy as np
from lib_mp import Noise, CMB_S4, Observables
import matplotlib.pyplot as plot

if __name__ == "__main__":
    t1 = time.time()

    c = 2.997e8
    LMAX = 100
    els = np.arange(0, LMAX + 1)

    # get planck noise for low L
    planck_noise = Noise.white_noise(theta_fwhm=[33,23,14,10,7,5,5], sigma_T=[45,149,137,65,43,66,200], sigma_P=[0], l_max=LMAX)

    # get DRAFT noise curves
    ppfile = '/Users/yovel/PycharmProjects/CMB-S4/CMB-S4_DRAFT/products/20220726/lensing_noise_curves/ilc/s4wide_lmin100_lmax5000.npy'
    pp = np.load(ppfile, allow_pickle=1, encoding='latin1').item()
    draft_noise = Noise(l_max=LMAX, TT=pp['Nl_TT'], EE=pp['Nl_EE'], PP=pp['cl_kk']) # TE=pp['Nl_ET']

    # add inversely (happens internally)
    noise = draft_noise + planck_noise

    fid = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.06, 0.010409*100, 0])
    dx_left = np.array([0.0030,0.0008,0.1e-9,0.010,0.020,0.000050*100, 0])
    dx_right = np.array([0.0030,0.0008,0.1e-9,0.010,0.020,0.000050*100,1.0e-8/c**2])

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

    obs.compute(l_max=LMAX, lensed_cl=False)

    experiment = CMB_S4(noise_curves=noise, l_max=LMAX)
    f = experiment.get_fisher(obs)

    print('Time elapsed:', time.time()-t1)

    cov = np.linalg.inv(f)

    # fix pann factors of c
    cov[-1, :] *= c ** 2
    cov[:, -1] *= c ** 2

    # graph
    import plot_triangle
    SCALES = [0, 0, 9, 0, 0, 2, 7]
    FID = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.06, 0.010409*100, 0])
    PANN = r'$p_{\mathrm{ann}}$ ($10^{-7}\mathrm{m}^3\mathrm{s}^{-1}\mathrm{kg}^{-1}$)'
    LABELS = [r'$\Omega_{DM} h^2$', r'$\Omega_b h^2$', r'$10^{9}A_s$', r'$n_s$', r'$\tau_{reio}$', r'$10^{2}\theta_s$',PANN]
    plot_triangle.triplot(LABELS, SCALES, FID, cov)

    plot.show()