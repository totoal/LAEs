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

def add_errors(pm_SEDs, apply_err=True, survey_name='minijpasAEGIS001'):
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
    elif survey_name[:8] == 'minijpas':
        pm_SEDs_err = np.array([]).reshape(60, 0)

        # Split sources in 4 groups (tiles) randomly
        N_sources = pm_SEDs.shape[1]
        rand_perm = np.random.permutation(np.arange(N_sources))
        N_src_i = N_sources // 4

        tile_id_Arr = [2241, 2243, 2406, 2470]

        i = float(survey_name[-1])

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
        expfit = lambda x: a * np.exp(b * x + c)
        
        w_central = central_wavelength().reshape(-1, 1)

        mags = flux_to_mag(pm_SEDs, w_central)
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

def source_f_cont(mjd, plate, fiber):
    try:
        f_cont = np.load('npy/f_cont_DR16.npy')
        print('f_cont Arr loaded')
        return f_cont
    except:
        pass
    print('Computing f_cont Arr')

    Lya_fts = pd.read_csv('../csv/Lya_fts_DR16.csv')

    N_sources = len(mjd)
    EW = np.empty(N_sources)
    Flambda = np.empty(N_sources)

    for src in range(N_sources):
        if src % 1000 == 0:
            print(f'{src} / {N_sources}', end='\r')

        where = np.where(
            (int(mjd[src]) == Lya_fts['mjd'].to_numpy().flatten())
            & (int(plate[src]) == Lya_fts['plate'].to_numpy().flatten())
            & (int(fiber[src]) == Lya_fts['fiberid'].to_numpy().flatten())
        )
        
        # Some sources are repeated, so we take the first occurence
        where = where[0][0]

        EW[src] = np.abs(Lya_fts['LyaEW'][where]) # Obs frame EW by now
        Flambda[src] = Lya_fts['LyaF'][where]

    Flambda *= 1e-17 # Correct units & apply correction

    # From the EW formula:
    f_cont = Flambda / EW

    np.save('npy/f_cont_DR16.npy', f_cont)

    return f_cont

def load_QSO_prior_mock():
    filename = ('../csv/J-SPECTRA_QSO_Superset_DR16.csv')

    format_string4 = lambda x: '{:04d}'.format(int(x))
    format_string5 = lambda x: '{:05d}'.format(int(x))
    convert_dict = {
        122: format_string4,
        123: format_string5,
        124: format_string4
    }
    plate_mjd_fiber = pd.read_csv(
        filename, sep=',', usecols=[61, 62, 63],
        converters=convert_dict
    ).to_numpy().T

    plate_mjd_fiber = plate_mjd_fiber[np.array([1, 0, 2])]

    return plate_mjd_fiber

def schechter(L, phistar, Lstar, alpha):
    '''
    Just the regular Schechter function
    '''
    return (phistar / Lstar) * (L / Lstar)**alpha * np.exp(-L / Lstar)

def duplicate_sources(area, z_Arr, L_Arr, z_min, z_max, L_min, L_max):
    volume = z_volume(z_min, z_max, area)

    Lx = np.logspace(L_min, L_max, 10000)
    log_Lx = np.log10(Lx)

    # Daniele's LF
    phistar1 = 3.33e-6
    Lstar1 = 44.65
    alpha1 = -1.35
    # Zhang's LF
    # phistar1 = 10 ** -5.85
    # Lstar1 = 44.6
    # alpha1 = -1.2
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

    # r-band LF from Palanque-Delabrouille (2016) PLE+LEDE model
    # We use the total values over all the magnitude bins
    # The original counts are for an area of 10000 deg2
    PD_z_Arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    PD_counts_Arr = np.array([975471, 2247522, 1282573, 280401, 31368, 4322])

    PD_z_cum_x = np.linspace(z_min, z_max, 1000)
    PD_counts_cum = np.cumsum(np.interp(PD_z_cum_x, PD_z_Arr, PD_counts_Arr))
    PD_counts_cum /= PD_counts_cum.max()

    my_z_Arr = np.interp(np.random.rand(N_sources_LAE), PD_counts_cum, PD_z_cum_x)

    # Index of the original mock closest source in redshift
    idx_closest = np.zeros(N_sources_LAE).astype(int)
    print('Looking for the closest QSOs...')

    # Randomize the whole QSO set and split in:
    # 70% train, 25% test, 5% validation
    np.random.seed(48713043)
    N_QSO_set = len(z_Arr)
    perm = np.random.permutation(np.arange(N_QSO_set))
    old_z_Arr = np.copy(z_Arr)
    if train_or_test == 'train':
        slc = slice(0, int(np.floor(N_QSO_set * 0.7)))
        z_Arr = z_Arr[perm][slc]
        L_Arr = L_Arr[perm][slc]
    if train_or_test == 'test':
        slc = slice(int(np.floor(N_QSO_set * 0.7)),
                    int(np.floor(N_QSO_set * 0.95)))
        z_Arr = z_Arr[perm][slc]
        L_Arr = L_Arr[perm][slc]

    for src in range(N_sources_LAE):
        if src % 500 == 0:
            print(f'Part {part}: {src} / {N_sources_LAE}')
        # Select sources with a redshift closer than 0.02
        closest_z_Arr = np.where(np.abs(z_Arr - my_z_Arr[src]) < 0.02)[0]
        # If less than 10 objects found with that z_diff, then select the 10 closer
        if len(closest_z_Arr < 10):
            closest_z_Arr = np.abs(z_Arr - my_z_Arr[src]).argsort()[:10]

        # Then, within the closest in z, we choose the 5 closest in L
        # Or don't select by L proximity (uncomment one)
        closest_L_Arr = np.abs(L_Arr[closest_z_Arr] - my_L_Arr[src]).argsort()[:5]
        # closest_L_Arr = np.abs(L_Arr[closest_z_Arr] - my_L_Arr[src]).argsort()

        idx_closest[src] = np.random.choice(closest_z_Arr[closest_L_Arr], 1)

    # The amount of w that we have to correct
    w_factor = (1 + my_z_Arr) / (1 + z_Arr[idx_closest])

    # The correction factor to achieve the desired L
    L_factor = 10 ** (my_L_Arr - L_Arr[idx_closest])

    # Re-convert idx_closest to initial units pre randomization
    idx_closest = perm[slc][idx_closest]

    # So, I need the source idx_closest, then correct its wavelength by adding w_offset
    # and finally multiplying its flux by L_factor
    return idx_closest, w_factor, L_factor, old_z_Arr[idx_closest]

def lya_band_z(fits_dir, plate, mjd, fiber, t_or_t):
    '''
    Computes correct Arr and saves it to a .csv if it dont exist
    '''
    lya_band_res = 1000 # Resolution of the Lya band
    lya_band_hw = 150 # Half width of the Lya band in Angstroms

    correct_dir = 'csv/QSO_mock_correct_files/'
    try:
        z = np.load(f'{correct_dir}z_arr_{t_or_t}_dr16.npy')
        lya_band = np.load(f'{correct_dir}lya_band_arr_{t_or_t}_dr16.npy')
        print('Correct arr loaded')

        return z, lya_band, lya_band_hw
    except:
        print('Computing correct arr...')
        pass

    N_sources = len(fiber)

    # Declare some arrays
    z = np.empty(N_sources)
    lya_band = np.zeros(N_sources)

    # Do the integrated photometry
    # print('Extracting band fluxes from the spectra...')
    print('Making lya_band Arr')
    plate = plate.astype(int)
    mjd = mjd.astype(int)
    fiber = fiber.astype(int)
    for src in range(N_sources):
        print(f'{src} / {N_sources}', end='\r')

        spec_name = fits_dir + f'spec-{plate[src]:04d}-{mjd[src]:05d}-{fiber[src]:04d}.fits'

        spec = Table.read(spec_name, hdu=1, format='fits')
        spzline = Table.read(spec_name, hdu=3, format='fits')

        # Select the source's z as the z from any line not being Lya.
        # Lya z is biased because is taken from the position of the peak of the line,
        # and in general Lya is assymmetrical.
        z_Arr = spzline['LINEZ'][spzline['LINENAME'] != 'Ly_alpha']
        z_Arr = np.atleast_1d(z_Arr[z_Arr != 0.])
        if len(z_Arr) > 0:
            z[src] = z_Arr[-1]
        else:
            z[src] = 0.

        if z[src] < 2:
            continue

        # Synthetic band in Ly-alpha wavelength +- 200 Angstroms
        w_lya_obs = w_lya * (1 + z[src])

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
                spec['FLUX'] * 1e-17, 10 ** spec['LOGLAM'], lya_band_tcurves
            )
        if ~np.isfinite(lya_band[src]):
            lya_band[src] = 0

    os.makedirs(correct_dir, exist_ok=True)
    np.save(f'{correct_dir}z_arr_{t_or_t}_dr16', z)
    np.save(f'{correct_dir}lya_band_arr_{t_or_t}_dr16', lya_band)
        
    return z, lya_band, lya_band_hw

def main(part, area, z_min, z_max, L_min, L_max, survey_name, train_or_test, surname):
    dirname = '/home/alberto/almacen/Source_cats'
    filename = f'{dirname}/QSO_double_{train_or_test}_{survey_name}_DR16_{surname}0'
    os.makedirs(filename, exist_ok=True)

    fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/DR16/QSO/'

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Loading the Carolina's QSO mock
    plate_mjd_fiber = load_QSO_prior_mock()

    plate = plate_mjd_fiber[0]
    mjd = plate_mjd_fiber[1]
    fiber = plate_mjd_fiber[2]

    z, lya_band, lya_band_hw = lya_band_z(fits_dir, plate, mjd, fiber, train_or_test)

    f_cont = source_f_cont(mjd, plate, fiber)

    F_line = (lya_band - f_cont) * 2 * lya_band_hw
    F_line_err = np.zeros(lya_band.shape)
    EW0 = F_line / f_cont / (1 + z)
    dL = cosmo.luminosity_distance(z).to(u.cm).value
    L = np.log10(F_line * 4*np.pi * dL ** 2)

    idx_closest, _, L_factor, new_z = duplicate_sources(
        area, z, L, z_min, z_max, L_min, L_max
    )

    # Load the DR16 PM
    filename_pm_DR16 = ('../csv/J-SPECTRA_QSO_Superset_DR16.csv')
    pm_SEDs_DR16 = pd.read_csv(
        filename_pm_DR16, usecols=np.arange(1, 64)
    ).to_numpy()[:, :60].T

    print('Sampling from DR16 pm...')
    pm_SEDs = pm_SEDs_DR16[:, idx_closest] * L_factor
    print('Ok')

    new_L = L[idx_closest] + np.log10(L_factor)
    new_F_line = F_line[idx_closest] * L_factor
    new_F_line_err = F_line_err[idx_closest] * L_factor
    new_EW0 = EW0[idx_closest] * (1 + z[idx_closest]) / (1 + new_z)

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
    low_r_mask = (pm_SEDs[-2] > 4e-19)
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
    L_min = 42
    L_max = 46
    area = 400 / (16 * 2) # We have to do 2 runs of 16 parallel processes

    for survey_name in ['minijpas', 'jnep']:
        for train_or_test in ['test', 'train']:
            main(part, area, z_min, z_max, L_min, L_max, survey_name, train_or_test, 'D_')

    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(time.time() - t0, 60)))

    t0 = time.time()
    z_min = 2
    z_max = 4.25
    L_min = 44
    L_max = 46
    area = 4000 / (16 * 2) # We have to do 2 runs of 12 parallel processes

    for survey_name in ['minijpas', 'jnep']:
        for train_or_test in ['test', 'train']:
            main(part, area, z_min, z_max, L_min, L_max, survey_name, train_or_test, 'highL2_D_')

    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(time.time() - t0, 60)))