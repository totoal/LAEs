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
w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))

def load_QSO_prior_mock():
    filename = ('../csv/J-SPECTRA_QSO_Superset_DR16_v2.csv')

    def format_string4(x): return '{:04d}'.format(int(x))
    def format_string5(x): return '{:05d}'.format(int(x))
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


def duplicate_sources(area, z_Arr, L_Arr, z_min, z_max, L_min, L_max, EW0):
    volume = z_volume(z_min, z_max, area)

    Lx = np.logspace(L_min, L_max, 10000)
    log_Lx = np.log10(Lx)

    # Daniele's LF
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

    my_z_Arr = np.interp(np.random.rand(N_sources_LAE),
                         PD_counts_cum, PD_z_cum_x)

    # Index of the original mock closest source in redshift
    idx_closest = np.zeros(N_sources_LAE).astype(int)

    for src in range(N_sources_LAE):
        # if src % 500 == 0:
            # print(f'Part {part}: {src} / {N_sources_LAE}')
        # Select sources with a redshift closer than 0.02
        closest_z_Arr = np.where((np.abs(z_Arr - my_z_Arr[src]) < 0.05)
                                 & (EW0 > 0) & np.isfinite(EW0))[0]
        # If less than 10 objects found with that z_diff, then select the 10 closer
        if len(closest_z_Arr) < 1:
            closest_z_Arr = np.abs(z_Arr - my_z_Arr[src]).argsort()
            print(z_Arr[closest_z_Arr[0]], my_z_Arr[src])
            print(f'Best I can do is: delta_z = {z_Arr[closest_z_Arr[0]] - my_z_Arr[src]:0.4f}')

        closest_L_Arr = np.abs(L_Arr[closest_z_Arr] - my_L_Arr[src]).argsort()

        # Pick the closest in L
        idx_closest[src] = np.random.choice(closest_z_Arr[closest_L_Arr], 1)

    # The amount of w that we have to correct
    w_factor = (1 + my_z_Arr) / (1 + z_Arr[idx_closest])

    # The correction factor to achieve the desired L
    L_factor = 10 ** (my_L_Arr - L_Arr[idx_closest])

    # So, I need the source idx_closest, then correct its wavelength by adding w_offset
    # and finally multiplying its flux by L_factor
    return idx_closest, w_factor, L_factor, z_Arr[idx_closest]


def lya_band_z(fits_dir, plate, mjd, fiber, t_or_t):
    '''
    Computes correct Arr and saves it to a .csv if it dont exist
    '''
    # lya_band_res = 1000  # Resolution of the Lya band
    # lya_band_hw = 150  # Half width of the Lya band in Angstroms

    correct_dir = 'csv/QSO_mock_correct_files/'
    try:
        z = np.load(f'{correct_dir}z_arr_{t_or_t}_dr16.npy')
        print('Correct arr loaded')

        return z
    except:
        print('Computing correct arr...')
        pass

    N_sources = len(fiber)

    # Declare some arrays
    z = np.empty(N_sources)
    # lya_band = np.zeros(N_sources)

    # Do the integrated photometry
    print('Making lya_band Arr')
    plate = plate.astype(int)
    mjd = mjd.astype(int)
    fiber = fiber.astype(int)
    for src in range(N_sources):
        if src % 500 == 0:
            print(f'{src} / {N_sources}', end='\r')

        spec_name = fits_dir + \
            f'spec-{plate[src]:04d}-{mjd[src]:05d}-{fiber[src]:04d}.fits'

        # spec = Table.read(spec_name, hdu=1, format='fits')
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

        if z[src] < 1.9:
            continue


    os.makedirs(correct_dir, exist_ok=True)
    np.save(f'{correct_dir}z_arr_{t_or_t}_dr16', z)

    return z


def main(part, area, z_min, z_max, L_min, L_max, surname):
    dirname = '/home/alberto/almacen/Source_cats'
    filename = f'{dirname}/QSO_400deg_z{z_min:0.1f}-{z_max:0.1f}_DR16_{surname}0'
    os.makedirs(filename, exist_ok=True)

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Loading the Carolina's QSO mock
    plate_mjd_fiber = load_QSO_prior_mock()

    plate = plate_mjd_fiber[0]
    mjd = plate_mjd_fiber[1]
    fiber = plate_mjd_fiber[2]

    Lya_fts = pd.read_csv('../csv/Lya_fts_DR16_v2.csv')
    z = Lya_fts['Lya_z'].to_numpy().flatten()
    z[z == 0] = -1

    F_line = np.array(Lya_fts['LyaF']) * 1e-17
    F_line_err = np.array(Lya_fts['LyaF_err']) * 1e-17
    EW0 = np.array(Lya_fts['LyaEW']) / (1 + z)
    EW_err = np.array(Lya_fts['LyaEW_err'])
    dL = cosmo.luminosity_distance(z).to(u.cm).value
    L = np.log10(F_line * 4*np.pi * dL ** 2)

    F_line_NV = np.array(Lya_fts['NVF']) * 1e-17
    F_line_NV_err = np.array(Lya_fts['NVF_err']) * 1e-17
    EW0_NV = np.array(Lya_fts['NVEW']) / (1 + z)
    L_NV = np.log10(F_line_NV * 4*np.pi * dL ** 2)
    
    # Mask poorly measured EWs
    # EW_snr = EW0 * (1 + z) / EW_err
    mask_neg_EW0 = (EW0 < 0) | ~np.isfinite(EW0)
    L[mask_neg_EW0] = -1
    z[mask_neg_EW0] = -1

    # Load the DR16 PM
    filename_pm_DR16 = ('../csv/J-SPECTRA_QSO_Superset_DR16_v2.csv')
    pm_SEDs_DR16 = pd.read_csv(
        filename_pm_DR16, usecols=np.arange(1, 64)
    ).to_numpy()[:, 0:60].T

    idx_closest, _, L_factor, new_z = duplicate_sources(area, z, L, z_min, z_max,
                                                        L_min, L_max, EW0)

    pm_SEDs = pm_SEDs_DR16[:, idx_closest] * np.array(L_factor)

    new_L = L[idx_closest] + np.log10(L_factor)
    new_F_line = F_line[idx_closest] * L_factor
    new_F_line_err = F_line_err[idx_closest] * L_factor
    new_EW0 = EW0[idx_closest] * (1 + z[idx_closest]) / (1 + new_z)

    new_L_NV = L_NV[idx_closest] + np.log10(L_factor)
    new_F_NV_line = F_line_NV[idx_closest] * L_factor
    new_F_NV_line_err = F_line_NV_err[idx_closest] * L_factor
    new_EW0_NV = EW0_NV[idx_closest] * (1 + z[idx_closest]) / (1 + new_z)

    where_out_of_range = (pm_SEDs > 1) | ~np.isfinite(pm_SEDs)

    # Add infinite errors to bands out of the range of SDSS
    pm_SEDs_err = np.zeros_like(pm_SEDs)

    pm_SEDs[where_out_of_range] = 0.
    pm_SEDs_err[where_out_of_range] = 99.

    hdr = (
        tcurves['tag']
        + [s + '_e' for s in tcurves['tag']]
        + ['z', 'EW0', 'L_lya', 'F_line', 'F_line_err']
        + ['EW0_NV', 'L_NV', 'F_line_NV', 'F_line_NV_err']
        + ['mjd', 'fiber', 'plate']
    )

    # Let's remove the sources with very low r magnitudes
    low_r_mask = (pm_SEDs[-2] > 1e-21)

    pd.DataFrame(
        data=np.hstack(
            (
                pm_SEDs.T[low_r_mask], pm_SEDs_err.T[low_r_mask],
                new_z[low_r_mask].reshape(-1, 1),
                new_EW0[low_r_mask].reshape(-1, 1),
                new_L[low_r_mask].reshape(-1, 1),
                new_F_line[low_r_mask].reshape(-1, 1),
                new_F_line_err[low_r_mask].reshape(-1, 1),
                new_EW0_NV[low_r_mask].reshape(-1, 1),
                new_L_NV[low_r_mask].reshape(-1, 1),
                new_F_NV_line[low_r_mask].reshape(-1, 1),
                new_F_NV_line_err[low_r_mask].reshape(-1, 1),
                mjd[idx_closest][low_r_mask].reshape(-1, 1),
                fiber[idx_closest][low_r_mask].reshape(-1, 1),
                plate[idx_closest][low_r_mask].reshape(-1, 1),
            )
        )
    ).to_csv(filename + f'/data{part}.csv', header=hdr)


if __name__ == '__main__':
    t0 = time.time()
    part = sys.argv[1]
    
    area_loL = 400 / (16 * 2)  # We have to do 2 runs of 16 parallel processes
    area_hiL = 4000 / (16 * 2)  # We have to do 2 runs of 16 parallel processes
    nbs_list = [[1, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 24]]

    for nb_min, nb_max in nbs_list:
        z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1 - 0.1
        z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1 + 0.1

        L_min = 42
        L_max = 46

        main(part, area_loL, z_min, z_max, L_min, L_max, 'loL_')

        if part == 1:
            print('loL in: {0:0.0f} m {1:0.1f} s'.format(
                *divmod(time.time() - t0, 60)))

        t0 = time.time()
        L_min = 44
        L_max = 46

        main(part, area_hiL, z_min, z_max, L_min, L_max, 'hiL_')

        if part == 1:
            print('hiL in: {0:0.0f} m {1:0.1f} s'.format(
                *divmod(time.time() - t0, 60)))
    
    print(f'Part {part} done.')