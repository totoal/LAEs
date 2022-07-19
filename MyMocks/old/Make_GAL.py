import os
from time import perf_counter

import pandas as pd

import numpy as np

from my_utilities import *
from Make_QSO_altered import add_errors

w_lya = 1215.67

def load_GAL_prior_mock():
    filename = (
        '/home/alberto/cosmos/JPAS_mocks_sep2021/'
        'JPAS_mocks_classification_19nov_model11/'
        'Fluxes_model_11/Gal_jpas_mock_flam_train.cat'
    )

    gal_flx = pd.read_csv(
        filename, sep=' ', usecols=[13, 29, 44] # 28 is rSDSS, 12 is gSDSS, 43 is iSDSS
    ).to_numpy()#.flatten()
    gal_r_err = pd.read_csv(
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

    return gal_flx, gal_r_err, plate_mjd_fiber

def main(survey_name):
    filename = f'/home/alberto/almacen/Source_cats/GAL_100000_{survey_name}_0'

    if not os.path.exists(filename):
        os.mkdir(filename)

    fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/GAL/'

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Loading the Carolina's GAL mock
    gal_r_flx, gal_err_r_flx, plate_mjd_fiber = load_GAL_prior_mock()
    N_sources = len(gal_r_flx)

    # Declare some arrays
    z = np.empty(N_sources)

    pm_SEDs = np.empty((60, N_sources))
    plate = np.zeros(N_sources).astype(str)
    mjd = np.zeros(N_sources).astype(str)
    fiber = np.zeros(N_sources).astype(str)
    correct = np.zeros(N_sources)

    # Do the integrated photometry
    print('Extracting band fluxes from the spectra...')
    for src in range(N_sources):
        print(f'{src} / {N_sources}', end='\r')

        plate[src] = plate_mjd_fiber[0, src]
        mjd[src] = plate_mjd_fiber[1, src]
        fiber[src] = plate_mjd_fiber[2, src]

        spec_name = fits_dir + f'spec-{plate[src]}-{mjd[src]}-{fiber[src]}.fits'

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

        pm_SEDs[:, src] = JPAS_synth_phot(
            spec['flux'] * 1e-17, 10 ** spec['loglam'], tcurves
        )

        # Adjust flux to match the prior mock
        if gal_r_flx[src, 1] > 0:
            correct[src] = gal_r_flx[src, 1] / pm_SEDs[-2, src]
        elif gal_r_flx[src, 2] > 0:
            correct[src] = gal_r_flx[src, 2] / pm_SEDs[-1, src]
        else:
            correct[src] = gal_err_r_flx[src] / pm_SEDs[-2, src]

        correct[~np.isfinite(correct)] = 0.
        pm_SEDs[:, src] *= correct[src]

    print('Adding errors...')

    where_out_of_range = (pm_SEDs > 1) | ~np.isfinite(pm_SEDs)

    # Add infinite errors to bands out of the range of SDSS
    pm_SEDs, pm_SEDs_err = add_errors(pm_SEDs, False, survey_name)

    pm_SEDs[where_out_of_range] = 0.
    pm_SEDs_err[where_out_of_range] = 99.

    hdr = (
        tcurves['tag']
        + [s + '_e' for s in tcurves['tag']]
        + ['z']
    )

    pd.DataFrame(
        data=np.hstack(
            (
                pm_SEDs.T, pm_SEDs_err.T, z.reshape(-1, 1)
            )
        )
    ).to_csv(filename + f'/data.csv', header=hdr)

if __name__ == '__main__':
    t0 = perf_counter()
    for survey_name in ['minijpas', 'jnep']:
        main(survey_name)
    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(perf_counter() - t0, 60)))