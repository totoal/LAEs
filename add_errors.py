import numpy as np
import pandas as pd
from my_functions import central_wavelength, Zero_point_error, mag_to_flux, flux_to_mag


def add_errors(pm_SEDs, apply_err=True, survey_name='minijpasAEGIS001'):
    if survey_name == 'jnep':
        err_fit_params_jnep = np.load('./npy/err_fit_params_jnep.npy')
    elif survey_name[:8] == 'minijpas':
        err_fit_params_001 = np.load(
            './npy/err_fit_params_minijpas_AEGIS001.npy')
        err_fit_params_002 = np.load(
            './npy/err_fit_params_minijpas_AEGIS002.npy')
        err_fit_params_003 = np.load(
            './npy/err_fit_params_minijpas_AEGIS003.npy')
        err_fit_params_004 = np.load(
            './npy/err_fit_params_minijpas_AEGIS004.npy')
    else:
        raise ValueError('Survey name not known')

    if survey_name[:8] == 'minijpas':
        detec_lim_001 = pd.read_csv('./csv/depth3arc5s_minijpas_2241.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_002 = pd.read_csv('./csv/depth3arc5s_minijpas_2243.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_003 = pd.read_csv('./csv/depth3arc5s_minijpas_2406.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_004 = pd.read_csv('./csv/depth3arc5s_minijpas_2470.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim = np.hstack(
            (
                detec_lim_001,
                detec_lim_002,
                detec_lim_003,
                detec_lim_004
            )
        )
        detec_lim.shape
    elif survey_name == 'jnep':
        detec_lim = pd.read_csv('./csv/depth3arc5s_jnep_2520.csv',
                                sep=',', header=0, usecols=[1]).to_numpy()

    if survey_name == 'jnep':
        a = err_fit_params_jnep[:, 0].reshape(-1, 1)
        b = err_fit_params_jnep[:, 1].reshape(-1, 1)
        c = err_fit_params_jnep[:, 2].reshape(-1, 1)
        def expfit(x): return a * np.exp(b * x + c)

        w_central = central_wavelength().reshape(-1, 1)

        mags = flux_to_mag(pm_SEDs, w_central)
        mags[np.isnan(mags) | np.isinf(mags)] = 99.

        # Zero point error
        zpt_err = Zero_point_error(np.ones(mags.shape[1]) * 2520, 'jnep')

        mag_err = (expfit(mags) ** 2 + zpt_err ** 2) ** 0.5
        where_himag = np.where(mags > detec_lim)

        mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)

        mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

        pm_SEDs_err = mag_to_flux(
            mags - mag_err, w_central) - mag_to_flux(mags, w_central)
    elif survey_name[:8] == 'minijpas':
        pm_SEDs_err = np.array([]).reshape(60, 0)

        # Split sources in 4 groups (tiles) randomly
        N_sources = pm_SEDs.shape[1]

        tile_id_Arr = [2241, 2243, 2406, 2470]

        i = int(survey_name[-1])

        detec_lim_i = detec_lim[:, i].reshape(-1, 1)

        if i == 1:
            a = err_fit_params_001[:, 0].reshape(-1, 1)
            b = err_fit_params_001[:, 1].reshape(-1, 1)
            c = err_fit_params_001[:, 2].reshape(-1, 1)
        if i == 2:
            a = err_fit_params_002[:, 0].reshape(-1, 1)
            b = err_fit_params_002[:, 1].reshape(-1, 1)
            c = err_fit_params_002[:, 2].reshape(-1, 1)
        if i == 3:
            a = err_fit_params_003[:, 0].reshape(-1, 1)
            b = err_fit_params_003[:, 1].reshape(-1, 1)
            c = err_fit_params_003[:, 2].reshape(-1, 1)
        if i == 4:
            a = err_fit_params_004[:, 0].reshape(-1, 1)
            b = err_fit_params_004[:, 1].reshape(-1, 1)
            c = err_fit_params_004[:, 2].reshape(-1, 1)

        def expfit(x): return a * np.exp(b * x + c)

        w_central = central_wavelength().reshape(-1, 1)

        mags = flux_to_mag(pm_SEDs, w_central)
        mags[np.isnan(mags) | np.isinf(mags)] = 99.

        # Zero point error
        tile_id = tile_id_Arr[i]
        zpt_err = Zero_point_error(
            np.ones(mags.shape[1]) * tile_id, 'minijpas')

        mag_err = (expfit(mags) ** 2 + zpt_err ** 2) ** 0.5
        where_himag = np.where(mags > detec_lim_i)

        mag_err[where_himag] = expfit(detec_lim_i)[where_himag[0]].reshape(-1,)

        mags[where_himag] = detec_lim_i[where_himag[0]].reshape(-1,)

        pm_SEDs_err_i = mag_to_flux(
            mags - mag_err, w_central) - mag_to_flux(mags, w_central)

        pm_SEDs_err = np.hstack((pm_SEDs_err, pm_SEDs_err_i))
    else:
        raise ValueError('Survey name not known')

    # Perturb according to the error
    if apply_err:
        pm_SEDs += np.random.normal(size=mags.shape) * pm_SEDs_err

    return pm_SEDs, pm_SEDs_err
