import os
from time import perf_counter

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

import pandas as pd

import numpy as np

from my_utilities import *

def add_errors(pm_SEDs):
    err_fit_params = np.load('../npy/err_fit_params_minijpas.npy')

    # Load limit mags
    detec_lim = np.vstack(
        (
            pd.read_csv('csv/5sigma_depths_NB.csv', header=None),
            pd.read_csv('csv/5sigma_depths_BB.csv', header=None)
        )
    )[:, 1].reshape(-1, 1)

    # Add errors
    a = err_fit_params[:, 0].reshape(-1, 1)
    b = err_fit_params[:, 1].reshape(-1, 1)
    c = err_fit_params[:, 2].reshape(-1, 1)
    expfit = lambda x: a * np.exp(b * x + c)

    w_central = central_wavelength().reshape(-1, 1)

    mags = flux_to_mag(pm_SEDs, w_central)
    mags[np.isnan(mags) | np.isinf(mags)] = 99.

    mag_err = expfit(mags)
    where_himag = np.where(mags > detec_lim)

    mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)
    mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

    pm_SEDs_err = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    # Perturb according to the error
    # pm_SEDs += np.random.normal(size=mags.shape) * pm_SEDs_err
    !!!!

    # Now recompute the error
    # mags = flux_to_mag(pm_SEDs, w_central)
    # mags[np.isnan(mags) | np.isinf(mags) | (mags > 26)] = 99.
    # mag_err = expfit(mags)
    # where_himag = np.where(mags > detec_lim)

    # mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)
    # mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

    # pm_SEDs_err = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    return pm_SEDs, pm_SEDs_err

def SDSS_QSO_line_fts(mjd, plate, fiber, correct):
    Lya_fts = pd.read_csv('../csv/Lya_fts.csv')

    N_sources = len(mjd)
    z = np.empty(N_sources)
    EW0 = np.empty(N_sources)
    L = np.empty(N_sources)
    Flambda = np.empty(N_sources) # Provisional

    for src in range(N_sources):
        where = np.where(
            (int(mjd[src]) == Lya_fts['mjd'].to_numpy().flatten())
            & (int(plate[src]) == Lya_fts['plate'].to_numpy().flatten())
            & (int(fiber[src]) == Lya_fts['fiberid'].to_numpy().flatten())
        )
        
        # Sources are repeated, so we take the first occurence
        where = where[0][0]

        z[src] = Lya_fts['Lya_z'][where]
        EW0[src] = np.abs(Lya_fts['LyaEW'][where]) # Obs frame EW by now
        Flambda[src] = Lya_fts['LyaF'][where]

    EW0 /= (1 + z) # Now it's rest frame EW0 & apply correction
    Flambda *= 1e-17 * correct # Correct units & apply correction

    dL = cosmo.luminosity_distance(z).to(u.cm).value
    L = np.log10(Flambda * 4*np.pi * dL ** 2)

    return z, EW0, L, Flambda

def load_QSO_prior_mock():
    filename = ('/home/alberto/cosmos/JPAS_mocks_sep2021/'
        'JPAS_mocks_classification_19nov_model11/Fluxes_model_11/Qso_jpas_mock_flam_train.cat')

    qso_flx = pd.read_csv(
        filename, sep=' ', usecols=[28] # 28 is rSDSS
    ).to_numpy().flatten()

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

    return qso_flx, plate_mjd_fiber

def main():
    filename = f'/home/alberto/cosmos/LAEs/MyMocks/QSO_100001'

    if not os.path.exists(filename):
        os.mkdir(filename)

    fits_dir = '/home/alberto/almacen/SDSS_QSO_fits/fits/'

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Loading the Carolina's QSO mock
    qso_r_flx, plate_mjd_fiber = load_QSO_prior_mock()
    N_sources = len(qso_r_flx)

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
        pm_SEDs[:, src] = JPAS_synth_phot(
            spec['flux'] * 1e-17, 10 ** spec['loglam'], tcurves
        )

        # Adjust flux to match the prior mock
        correct[src] = qso_r_flx[src] / pm_SEDs[-2, src]
        pm_SEDs[:, src] *= correct[src]

    print('Adding errors...')
    pm_SEDs, pm_SEDs_err = add_errors(pm_SEDs)

    print('Extracting line features...')
    z, EW0, L, F_line = SDSS_QSO_line_fts(mjd, plate, fiber, correct)

    hdr = tcurves['tag'] + [s + '_e' for s in tcurves['tag']] + ['z', 'EW0', 'L_lya', 'F_line']

    pd.DataFrame(
        data=np.hstack(
            (
                pm_SEDs.T, pm_SEDs_err.T, z.reshape(-1, 1), EW0.reshape(-1, 1),
                L.reshape(-1, 1), F_line.reshape(-1, 1)
            )
        )
    ).to_csv(filename + f'/data.csv', header=hdr)

if __name__ == '__main__':
    t0 = perf_counter()
    main()
    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(perf_counter() - t0, 60)))