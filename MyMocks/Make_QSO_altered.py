#!/home/alberto/miniconda3/bin/python3

import os
import sys
import time

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

import pandas as pd

import numpy as np

from my_utilities import *

w_lya = 1215.67

def add_errors(pm_SEDs, apply_err=True, survey_name='minijpas'):
    if survey_name == 'jnep':
        err_fit_params_jnep = np.load('../npy/err_fit_params_jnep.npy')
    elif survey_name == 'minijpas':
        err_fit_params_001 = np.load('../npy/err_fit_params_minijpas_AEGIS001.npy')
        err_fit_params_002 = np.load('../npy/err_fit_params_minijpas_AEGIS002.npy')
        err_fit_params_003 = np.load('../npy/err_fit_params_minijpas_AEGIS003.npy')
        err_fit_params_004 = np.load('../npy/err_fit_params_minijpas_AEGIS004.npy')
    else:
        raise ValueError('Survey name not known')

    if survey_name == 'minijpas':
        detec_lim_001 = pd.read_csv('../csv/depth3arc5s_minijpas_2241.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_002 = pd.read_csv('../csv/depth3arc5s_minijpas_2243.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_003 = pd.read_csv('../csv/depth3arc5s_minijpas_2406.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_004 = pd.read_csv('../csv/depth3arc5s_minijpas_2470.csv',
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
    if survey_name == 'jnep':
        detec_lim = pd.read_csv('../csv/depth3arc5s_jnep_2520.csv',
                                sep=',', header=0, usecols=[1]).to_numpy()

    if survey_name == 'jnep':
        a = err_fit_params_jnep[:, 0].reshape(-1, 1)
        b = err_fit_params_jnep[:, 1].reshape(-1, 1)
        c = err_fit_params_jnep[:, 2].reshape(-1, 1)
        expfit = lambda x: a * np.exp(b * x + c)

        w_central = central_wavelength().reshape(-1, 1)

        mags = flux_to_mag(pm_SEDs, w_central)
        mags[np.isnan(mags) | np.isinf(mags)] = 99.

        # Zero point error
        zpt_err = Zero_point_error(np.ones(mags.shape[1]) * 2520, 'jnep')

        mag_err = (expfit(mags) ** 2 + zpt_err ** 2) ** 0.5
        where_himag = np.where(mags > detec_lim)

        mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)

        mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

        pm_SEDs_err = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)
    elif survey_name == 'minijpas':
        pm_SEDs_err = np.array([]).reshape(60, 0)

        # Split sources in 4 groups (tiles) randomly
        N_sources = pm_SEDs.shape[1]
        rand_perm = np.random.permutation(np.arange(N_sources))
        N_src_i = N_sources // 4

        tile_id_Arr = [2241, 2243, 2406, 2470]

        for i in range(4):
            if i < 3:
                idx = rand_perm[i * N_src_i : (i + 1) * N_src_i]
            if i == 3:
                idx = rand_perm[i * N_src_i:]
            
            detec_lim_i = detec_lim[:, i].reshape(-1, 1)

            if i == 0:
                a = err_fit_params_001[:, 0].reshape(-1, 1)
                b = err_fit_params_001[:, 1].reshape(-1, 1)
                c = err_fit_params_001[:, 2].reshape(-1, 1)
            if i == 1:
                a = err_fit_params_002[:, 0].reshape(-1, 1)
                b = err_fit_params_002[:, 1].reshape(-1, 1)
                c = err_fit_params_002[:, 2].reshape(-1, 1)
            if i == 2:
                a = err_fit_params_003[:, 0].reshape(-1, 1)
                b = err_fit_params_003[:, 1].reshape(-1, 1)
                c = err_fit_params_003[:, 2].reshape(-1, 1)
            if i == 3:
                a = err_fit_params_004[:, 0].reshape(-1, 1)
                b = err_fit_params_004[:, 1].reshape(-1, 1)
                c = err_fit_params_004[:, 2].reshape(-1, 1)
            expfit = lambda x: a * np.exp(b * x + c)
            
            w_central = central_wavelength().reshape(-1, 1)

            mags = flux_to_mag(pm_SEDs[:, idx], w_central)
            mags[np.isnan(mags) | np.isinf(mags)] = 99.

            # Zero point error
            tile_id = tile_id_Arr[i]
            zpt_err = Zero_point_error(np.ones(mags.shape[1]) * tile_id, 'minijpas')

            mag_err = (expfit(mags) ** 2 + zpt_err ** 2) ** 0.5
            where_himag = np.where(mags > detec_lim_i)

            mag_err[where_himag] = expfit(detec_lim_i)[where_himag[0]].reshape(-1,)

            mags[where_himag] = detec_lim_i[where_himag[0]].reshape(-1,)

            pm_SEDs_err_i = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)

            pm_SEDs_err = np.hstack((pm_SEDs_err, pm_SEDs_err_i))
    else:
        raise ValueError('Survey name not known')

    # Perturb according to the error
    if apply_err:
        pm_SEDs += np.random.normal(size=mags.shape) * pm_SEDs_err

    return pm_SEDs, pm_SEDs_err

def SDSS_QSO_line_fts(mjd, plate, fiber, correct, z, train_or_test):
    if train_or_test == 'train':
        Lya_fts = pd.read_csv('../csv/Lya_fts.csv')
    elif train_or_test == 'test':
        Lya_fts = pd.read_csv('../csv/Lya_fts_test.csv')

    N_sources = len(mjd)
    EW = np.empty(N_sources)
    L = np.empty(N_sources)
    Flambda = np.empty(N_sources)
    Flambda_err = np.empty(N_sources)

    for src in range(N_sources):
        where = np.where(
            (int(mjd[src]) == Lya_fts['mjd'].to_numpy().flatten())
            & (int(plate[src]) == Lya_fts['plate'].to_numpy().flatten())
            & (int(fiber[src]) == Lya_fts['fiberid'].to_numpy().flatten())
        )
        
        # Some sources are repeated, so we take the first occurence
        where = where[0][0]

        EW[src] = np.abs(Lya_fts['LyaEW'][where]) # Obs frame EW by now
        Flambda[src] = Lya_fts['LyaF'][where]
        Flambda_err[src] = Lya_fts['LyaF_err'][where]

    EW0 = EW / (1 + z) # Now it's rest frame EW0
    Flambda *= 1e-17 * correct # Correct units & apply correction
    Flambda_err *= 1e-17 * correct # Correct units & apply correction

    # From the EW formula:
    f_cont = Flambda / EW
    f_cont_err = Flambda_err / EW

    dL = cosmo.luminosity_distance(z).to(u.cm).value
    L = np.log10(Flambda * 4*np.pi * dL ** 2)

    return EW0, L, Flambda, Flambda_err, f_cont, f_cont_err

def load_QSO_prior_mock(train_or_test):
    filename = (
        '/home/alberto/cosmos/JPAS_mocks_sep2021/'
        'JPAS_mocks_classification_19nov_model11/'
        f'Fluxes_model_11/Qso_jpas_mock_flam_{train_or_test}.cat'
    )

    qso_flx = pd.read_csv(
        filename, sep=' ', usecols=[13, 29, 44] # 28 is rSDSS, 12 is gSDSS, 43 is iSDSS
    ).to_numpy()#.flatten()
    qso_r_err = pd.read_csv(
        filename, sep=' ', usecols=[29 + 60] # 28 is rSDSS, 12 is gSDSS, 43 is iSDSS
    ).to_numpy()#.flatten()

    format_string4 = lambda x: '{:04d}'.format(int(x))
    format_string5 = lambda x: '{:05d}'.format(int(x))
    convert_dict = {
        122: format_string4,
        123: format_string5,
        124: format_string4
    }
    plate_mjd_fiber = pd.read_csv(
        filename, sep=' ', usecols=[122, 123, 124],
        converters=convert_dict
    ).to_numpy().T

    return qso_flx, qso_r_err, plate_mjd_fiber

def schechter(L, phistar, Lstar, alpha):
    '''
    Just the regular Schechter function
    '''
    return (phistar / Lstar) * (L / Lstar)**alpha * np.exp(-L / Lstar)

def duplicate_sources(area, z_Arr, L_Arr, z_min, z_max, L_min, L_max):
    volume = z_volume(z_min, z_max, area)

    Lx = np.logspace(L_min, L_max, 10000)
    log_Lx = np.log10(Lx)
    phistar1 = 3.33e-6
    Lstar1 = 44.65
    alpha1 = -1.35
    Phi = schechter(Lx, phistar1, 10 ** Lstar1, alpha1) * Lx * np.log(10)

    LF_p_cum_x = np.linspace(L_min, L_max, 1000)
    N_sources_LAE = int(
        simpson(
            np.interp(LF_p_cum_x, log_Lx, Phi), LF_p_cum_x
        ) * volume
    )
    print(f'N_new_sources = {N_sources_LAE}')
    LF_p_cum = np.cumsum(np.interp(LF_p_cum_x, log_Lx, Phi))
    LF_p_cum /= np.max(LF_p_cum)
    
    # L_Arr is the L_lya distribution for our mock
    my_L_Arr = np.interp(np.random.rand(N_sources_LAE), LF_p_cum, LF_p_cum_x)

    # g-band LF from Palanque-Delabrouille (2016) PLE+LEDE model
    # We use the total values over all the magnitude bins
    # The original counts are for an area of 10000 deg2
    PD_z_Arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    PD_counts_Arr = np.array([1216538, 3276523, 2289589, 359429, 16003, 640])

    PD_z_cum_x = np.linspace(z_min, z_max, 1000)
    PD_counts_cum = np.cumsum(np.interp(PD_z_cum_x, PD_z_Arr, PD_counts_Arr))
    PD_counts_cum /= PD_counts_cum.max()

    my_z_Arr = np.interp(np.random.rand(N_sources_LAE), PD_counts_cum, PD_z_cum_x)

    # Index of the original mock closest source in redshift
    idx_closest = np.zeros(N_sources_LAE).astype(int)
    for src in range(N_sources_LAE):
        # Select sources with a redshift closer than 0.02
        closest_z_Arr = np.where(np.abs(z_Arr - my_z_Arr[src]) < 0.02)[0]
        # If less than 10 objects found with that z_diff, then select the 10 closer
        if len(closest_z_Arr < 10):
            closest_z_Arr = np.abs(z_Arr - my_z_Arr[src]).argsort()[:10]

        # Then, within the closest in z, we choose the 5 closest in L
        closest_L_Arr = np.abs(L_Arr[closest_z_Arr] - my_L_Arr[src]).argsort()[:5]

        idx_closest[src] = np.random.choice(closest_z_Arr[closest_L_Arr], 1)

    # The amount of w that we have to correct
    w_factor = (1 + my_z_Arr) / (1 + z_Arr[idx_closest])

    # The correction factor to achieve the desired L
    L_factor = 10 ** (my_L_Arr - L_Arr[idx_closest])

    # So, I need the source idx_closest, then correct its wavelength by adding w_offset
    # and finally multiplying its flux by L_factor
    return idx_closest, w_factor, L_factor, my_z_Arr

def flux_correct(fits_dir, plate, mjd, fiber, tcurves, qso_r_flx, qso_err_r_flx, t_or_t):
    '''
    Computes correct Arr and saves it to a .csv if it dont exist
    '''
    correct_dir = 'csv/QSO_mock_correct_files/'
    try:
        correct = np.load(f'{correct_dir}correct_arr_{t_or_t}.npy')
        z = np.load(f'{correct_dir}z_arr_{t_or_t}.npy')
        lya_band = np.load(f'{correct_dir}lya_band_arr_{t_or_t}.npy')
        print('Correct arr loaded')

        return correct, z, lya_band
    except:
        print('Ccomputing correct arr...')
        pass

    N_sources = len(fiber)
    correct = np.zeros(N_sources)

    # Declare some arrays
    z = np.empty(N_sources)
    lya_band = np.zeros(N_sources)

    # Do the integrated photometry
    # print('Extracting band fluxes from the spectra...')
    pm_calib_bands = np.empty((N_sources, 3))
    print('Making correct Arr')
    for src in range(N_sources):
        print(f'{src} / {N_sources}', end='\r')

        spec_name = fits_dir + f'spec-{plate[src]}-{mjd[src]}-{fiber[src]}.fits'

        spec = Table.read(spec_name, hdu=1, format='fits')
        spzline = Table.read(spec_name, hdu=3, format='fits')

        # Select the source's z as the z from any line not being Lya.
        # Lya z is biased because is taken from the position of the peak of the line,
        # and in general Lya is assymmetrical.
        z_Arr = spzline['LINEZ'][spzline['LINENAME'] != 'Ly_alpha']
        L_lya = np.atleast_1d(spzline['LINEAREA'][spzline['LINENAME'] == 'Ly_alpha'])[0]
        z_Arr = np.atleast_1d(z_Arr[z_Arr != 0.])
        if len(z_Arr) > 0:
            z[src] = z_Arr[-1]
        else:
            z[src] = 0.

        # The range of SDSS is 3561-10327 Angstroms. Beyond the range limits,
        # the flux will be 0
        pm_calib_bands[src] = JPAS_synth_phot(
            spec['flux'] * 1e-17, 10 ** spec['loglam'], tcurves, [-3, -2, -1]
        )

        # Synthetic band in Ly-alpha wavelength +- 200 Angstroms
        w_lya_obs = w_lya * (1 + z[src])
        lya_band_res = 500 # Resolution of the Lya band
        lya_band_hw = 75 # Half width of the Lya band in Angstroms

        lya_band_tcurves = {
            'tag': ['lyaband'],
            't': [np.ones(lya_band_res)],
            'w': [np.linspace(
                w_lya_obs - lya_band_hw, w_lya_obs + lya_band_hw, lya_band_res
            )]
        }
        # Extract the photometry of Ly-alpha (L_Arr)
        if z[src] > 0:
            lya_band[src] = JPAS_synth_phot(
                spec['flux'] * 1e-17, 10 ** spec['loglam'], lya_band_tcurves
            )
        if ~np.isfinite(lya_band[src]):
            lya_band[src] = 0

        # Adjust flux to match the prior mock
        if qso_r_flx[src, 1] > 0:
            correct[src] = qso_r_flx[src, 1] / pm_calib_bands[src, 1]
        elif qso_r_flx[src, 0] > 0:
            correct[src] = qso_r_flx[src, 0] / pm_calib_bands[src, 0]
        elif qso_r_flx[src, 2] > 0:
            correct[src] = qso_r_flx[src, 2] / pm_calib_bands[src, 2]
        else:
            correct[src] = qso_err_r_flx[src] / pm_calib_bands[src, 1]

        bad_src = (
            ~np.isfinite(correct[src])
            | ((L_lya > 0) & (lya_band[src] == 0))
        )
        if bad_src:
            correct[src] = 0

    os.makedirs(correct_dir, exist_ok=True)
    np.save(f'{correct_dir}correct_arr_{t_or_t}', correct)
    np.save(f'{correct_dir}z_arr_{t_or_t}', z)
    np.save(f'{correct_dir}lya_band_arr_{t_or_t}', lya_band)
        
    return correct, z, lya_band

def main(part, area, z_min, z_max, L_min, L_max, survey_name, train_or_test):
    dirname = '/home/alberto/almacen/Source_cats'
    filename = f'{dirname}/QSO_double_{train_or_test}_{survey_name}_highL_0'

    if not os.path.exists(filename):
        os.mkdir(filename)

    if train_or_test == 'train':
        fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/QSO/'
    elif train_or_test == 'test':
        fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/QSO/test/'

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Loading the Carolina's QSO mock
    qso_r_flx, qso_err_r_flx, plate_mjd_fiber = load_QSO_prior_mock(train_or_test)

    plate = plate_mjd_fiber[0]
    mjd = plate_mjd_fiber[1]
    fiber = plate_mjd_fiber[2]


    correct, z, lya_band= flux_correct(fits_dir, plate, mjd, fiber,
                                       tcurves, qso_r_flx, qso_err_r_flx,
                                       train_or_test)
    lya_band_hw = 75

    print('Extracting line features...')
    _, _, _, _, f_cont, _ =\
         SDSS_QSO_line_fts(mjd, plate, fiber, correct, z, train_or_test)

    ## Computing L using Lya_band
    f_cont *= correct
    lya_band *= correct

    F_line = (lya_band - f_cont) * 2 * lya_band_hw
    F_line_err = np.zeros(lya_band.shape)
    EW0 = F_line / f_cont / (1 + z)
    dL = cosmo.luminosity_distance(z).to(u.cm).value
    L = np.log10(F_line * 4*np.pi * dL ** 2)

    idx_closest, w_factor, L_factor, new_z = duplicate_sources(
        area, z, L, z_min, z_max, L_min, L_max
    )

    new_N_sources = len(w_factor)

    pm_SEDs = np.empty((60, new_N_sources))

    # Do the integrated photometry
    print('Extracting band fluxes from the spectra...')
    for new_src in range(new_N_sources):
        src = idx_closest[new_src]

        print(f'{new_src} / {new_N_sources}', end='\r')

        spec_name = fits_dir + f'spec-{plate[src]}-{mjd[src]}-{fiber[src]}.fits'

        spec = Table.read(spec_name, hdu=1, format='fits')
        # Correct spec
        spec_w = 10 ** spec['loglam'] * w_factor[new_src]
        spec_f = spec['flux'] * 1e-17 * L_factor[new_src]

        # The range of SDSS is 3561-10327 Angstroms. Beyond the range limits,
        # the flux will be 0
        pm_SEDs[:, new_src] = JPAS_synth_phot(spec_f, spec_w, tcurves)

    new_L = L[idx_closest] + np.log10(L_factor)
    new_F_line = F_line[idx_closest] * L_factor
    new_F_line_err = F_line_err[idx_closest] * L_factor
    new_EW0 = EW0[idx_closest] * (1 + z[idx_closest]) / (1 + new_z)

    print('Adding errors...')

    where_out_of_range = (pm_SEDs > 1) | ~np.isfinite(pm_SEDs)

    # Add infinite errors to bands out of the range of SDSS
    pm_SEDs, pm_SEDs_err = add_errors(pm_SEDs, apply_err=False, survey_name=survey_name)

    pm_SEDs[where_out_of_range] = 0.
    pm_SEDs_err[where_out_of_range] = 99.

    hdr = (
        tcurves['tag']
        + [s + '_e' for s in tcurves['tag']]
        + ['z', 'EW0', 'L_lya', 'F_line', 'F_line_err']
    )

    ## Let's remove the sources with very low r magnitudes
    low_r_mask = (pm_SEDs[-2] > 6e-19)
    print(f'Final N_sources = {len(np.where(low_r_mask)[0])}')

    pd.DataFrame(
        data=np.hstack(
            (
                pm_SEDs.T[low_r_mask], pm_SEDs_err.T[low_r_mask],
                new_z[low_r_mask].reshape(-1, 1), new_EW0[low_r_mask].reshape(-1, 1),
                new_L[low_r_mask].reshape(-1, 1), new_F_line[low_r_mask].reshape(-1, 1),
                new_F_line_err[low_r_mask].reshape(-1, 1)
            )
        )
    ).to_csv(filename + f'/data{part}.csv', header=hdr)

if __name__ == '__main__':
    t0 = time.time()
    part = sys.argv[1]

    z_min = 2
    z_max = 4.25
    L_min = 44
    L_max = 46
    area = 2000 / (12 * 2) # We have to do 2 runs of 12 parallel processes

    for survey_name in ['minijpas', 'jnep']:
        for train_or_test in ['train']:
            main(part, area, z_min, z_max, L_min, L_max, survey_name, train_or_test)

    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(time.time() - t0, 60)))