import os
from time import perf_counter

import pandas as pd

import numpy as np

from my_utilities import *

w_lya = 1215.67

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

    # Zero point error
    zpt_err = Zero_point_error(np.ones(mags.shape[1]) * 2243, 'minijpas')

    mag_err = (expfit(mags) ** 2 + zpt_err ** 2) ** 0.5
    where_himag = np.where(mags > detec_lim)

    mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)

    mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

    pm_SEDs_err = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    # Perturb according to the error
    pm_SEDs += np.random.normal(size=mags.shape) * pm_SEDs_err

    # Now recompute the error
    # mags = flux_to_mag(pm_SEDs, w_central)
    # mags[np.isnan(mags) | np.isinf(mags) | (mags > 26)] = 99.
    # mag_err = expfit(mags)
    # where_himag = np.where(mags > detec_lim)

    # mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)
    # mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

    # pm_SEDs_err = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    return pm_SEDs, pm_SEDs_err

'''
def SDSS_GAL_line_fts(mjd, plate, fiber, correct, z):
    Lya_fts = pd.read_csv('../csv/Lya_fts.csv')

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
'''

def load_GAL_prior_mock():
    filename = (
        '/home/alberto/cosmos/JPAS_mocks_sep2021/'
        'JPAS_mocks_classification_19nov_model11/'
        'Fluxes_model_11/Gal_jpas_mock_flam_train.cat'
    )

    gal_flx = pd.read_csv(
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

    return gal_flx, plate_mjd_fiber

def main():
    filename = f'/home/alberto/cosmos/LAEs/MyMocks/GAL_100000'

    if not os.path.exists(filename):
        os.mkdir(filename)

    fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/GAL/'

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Loading the Carolina's GAL mock
    gal_r_flx, plate_mjd_fiber = load_GAL_prior_mock()
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
        correct[src] = gal_r_flx[src] / pm_SEDs[-2, src]
        pm_SEDs[:, src] *= correct[src]

    print('Adding errors...')
    pm_SEDs, pm_SEDs_err = add_errors(pm_SEDs)

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
    main()
    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(perf_counter() - t0, 60)))