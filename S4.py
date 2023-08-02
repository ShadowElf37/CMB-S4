from fishchips.experiments import *


class CMB_S4(Experiment):
    """
    Class for computing Fisher matrices from the CMB primary (TT/TE/EE).

    This experiment class requires some instrument parameters, and computes
    white noise for each multipole. The computation of the Fisher matrix
    follows equation 4 of arxiv:1402.4108.
    """

    def __init__(self, theta_fwhm=(10., 7., 5.),
                 sigma_T=(68.1, 42.6, 65.4),
                 sigma_P=(109.4, 81.3, 133.6),
                 f_sky=0.65, l_min=2, l_max=2500,
                 verbose=False):
        """
        Initialize the experiment with noise parameters.

        Uses the Planck bluebook parameters by default.

        Parameters
        ----------
            theta_fwhm (list of float): beam resolution in arcmin
            sigma_T (list of float): temperature resolution in muK
            sigma_P (list of float): polarization resolution in muK
            f_sky (float): sky fraction covered
            l_min (int): minimum ell for CMB power spectrum
            l_max (int): maximum ell for CMB power spectrum
            verbose (boolean): flag for printing out debugging output

        """
        self.verbose = verbose

        # convert from arcmin to radians
        self.theta_fwhm = theta_fwhm * np.array([np.pi / 60. / 180.])
        self.sigma_T = sigma_T * np.array([np.pi / 60. / 180.])
        self.sigma_P = sigma_P * np.array([np.pi / 60. / 180.])
        self.num_channels = len(theta_fwhm)
        self.f_sky = f_sky
        self.ells = np.arange(l_max + 1)

        self.l_min = l_min
        self.l_max = l_max

        # compute noise in muK**2, adapted from Monte Python
        self.noise_T = np.zeros(self.l_max + 1, 'float64')
        self.noise_P = np.zeros(self.l_max + 1, 'float64')
        self.noise_TE = np.zeros(self.l_max + 1, 'float64')

        for l in range(self.l_min, self.l_max + 1):
            self.noise_T[l] = 0
            self.noise_P[l] = 0
            for channel in range(self.num_channels):
                self.noise_T[l] += self.sigma_T[channel] ** -2 * \
                                   np.exp(
                                       -l * (l + 1) * self.theta_fwhm[channel] ** 2 / 8. / np.log(2.))
                self.noise_P[l] += self.sigma_P[channel] ** -2 * \
                                   np.exp(
                                       -l * (l + 1) * self.theta_fwhm[channel] ** 2 / 8. / np.log(2.))
            self.noise_T[l] = 1 / self.noise_T[l]
            self.noise_P[l] = 1 / self.noise_P[l]

        self.noise_T[self.ells < self.l_min] = 1e100
        self.noise_P[self.ells < self.l_min] = 1e100
        self.noise_T[self.ells > self.l_max] = 1e100
        self.noise_P[self.ells > self.l_max] = 1e100

    def compute_fisher_from_spectra(self, fid, df, pars):
        """
        Compute the Fisher matrix given fiducial and derivative dicts.

        This function is for generality, to enable easier interfacing with
        codes like CAMB. The input parameters must be in the units of the
        noise, muK^2.

        Parameters
        ----------
        fid (dictionary) : keys are '{parameter_XY}' with XY in {tt, te, ee}.
            These keys point to the actual power spectra.

        df (dictionary) :  keys are '{parameter_XY}' with XY in {tt, te, ee}.
            These keys point to numerically estimated derivatives generated
            from precomputed cosmologies.

        pars (list of strings) : the parameters being constrained in the
            Fisher analysis.

        """
        npar = len(pars)
        self.fisher = np.zeros((npar, npar))
        self.fisher_ell = np.zeros(self.l_max)

        for i, j in itertools.combinations_with_replacement(range(npar), r=2):
            # following eq 4 of https://arxiv.org/pdf/1402.4108.pdf
            fisher_ij = 0.0
            # probably a more efficient way to do this exists
            for l in range(self.l_min, self.l_max):
                Cl = np.array([[fid['tt'][l] + self.noise_T[l], fid['te'][l] + self.noise_TE[l]],
                               [fid['te'][l] + self.noise_TE[l], fid['ee'][l] + self.noise_P[l]]])
                invCl = np.linalg.inv(Cl)

                dCl_i = np.array([[df[pars[i] + '_tt'][l], df[pars[i] + '_te'][l]],
                                  [df[pars[i] + '_te'][l], df[pars[i] + '_ee'][l]]])
                dCl_j = np.array([[df[pars[j] + '_tt'][l], df[pars[j] + '_te'][l]],
                                  [df[pars[j] + '_te'][l], df[pars[j] + '_ee'][l]]])

                inner_term = np.dot(np.dot(invCl, dCl_i), np.dot(invCl, dCl_j))
                fisher_contrib = (2 * l + 1) / 2. * self.f_sky * np.trace(inner_term)
                fisher_ij += fisher_contrib

            # fisher is diagonal, so we get half of the matrix for free
            self.fisher[i, j] = fisher_ij
            self.fisher[j, i] = fisher_ij

        return self.fisher

    def get_fisher(self, obs, lensed_Cl=True):
        """
        Return a Fisher matrix using a dictionary full of CLASS objects.

        This function wraps the functionality of `compute_fisher_from_spectra`,
        for use with a dictionary filled with CLASS objects.

        Parameters
        ----------
            obs (Observations instance) : contains many evaluated CLASS cosmologies, at
                both the derivatives and the fiducial in the cosmos object.

        Returns
        -------
            Numpy array of floats with dimensions (len(params), len(params))

        """
        # first compute the fiducial
        fid_cosmo = obs.cosmos['fiducial']
        Tcmb = fid_cosmo.T_cmb()
        if lensed_Cl:
            fid_cl = fid_cosmo.lensed_cl(self.l_max)
        else:
            fid_cl = fid_cosmo.raw_cl(self.l_max)
        fid = {'tt': (Tcmb * 1.0e6) ** 2 * fid_cl['tt'],
               'te': (Tcmb * 1.0e6) ** 2 * fid_cl['te'],
               'ee': (Tcmb * 1.0e6) ** 2 * fid_cl['ee']}

        # the primary task of this function is to compute the derivatives from `cosmos`,
        # the dictionary of computed CLASS cosmologies
        dx_array = np.array(obs.right) - np.array(obs.left)

        df = {}
        # loop over parameters, and compute derivatives
        for par, dx in zip(obs.parameters, dx_array):
            if lensed_Cl:
                cl_left = obs.cosmos[par + '_left'].lensed_cl(self.l_max)
                cl_right = obs.cosmos[par + '_right'].lensed_cl(self.l_max)
            else:
                cl_left = obs.cosmos[par + '_left'].raw_cl(self.l_max)
                cl_right = obs.cosmos[par + '_right'].raw_cl(self.l_max)

            for spec_xy in ['tt', 'te', 'ee']:
                df[par + '_' + spec_xy] = (Tcmb * 1.0e6) ** 2 * \
                                          (cl_right[spec_xy] - cl_left[spec_xy]) / dx

        return self.compute_fisher_from_spectra(fid,
                                                df,
                                                obs.parameters)