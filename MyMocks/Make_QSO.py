import os
import sys
import glob
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

    pm_SEDs_err = mag_to_flux(mags + mag_err, w_central) - mag_to_flux(mags, w_central)

    # Perturb according to the error
    pm_SEDs += np.random.normal(size=mags.shape) * pm_SEDs_err

    # Now recompute the error
    mags = flux_to_mag(pm_SEDs, w_central)
    mags[np.isnan(mags) | np.isinf(mags) | (mags > 26)] = 99.
    mag_err = expfit(mags)
    where_himag = np.where(mags > detec_lim)

    mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)
    mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

    pm_SEDs_err = mag_to_flux(mags + mag_err, w_central) - mag_to_flux(mags, w_central)

    return pm_SEDs, pm_SEDs_err

def SDSS_QSO_line_fts(mjd, plate, fiber):
    Lya_fts = pd.read_csv('csv/Lya_fts.csv')

    N_sources = len(mjd)
    z = np.empty(N_sources)
    EW0 = np.empty(N_sources)
    L = np.empty(N_sources)
    Flambda = np.empty(N_sources) # Provisional

    for src in range(N_sources):
        where = (
            (mjd[src] == Lya_fts['mjd'])
            & (plate[src] == Lya_fts['plate'])
            & (fiber[src] == Lya_fts['fiber'])
        )

        z[src] = Lya_fts['Lya_z'][src]
        EW0[src] = np.abs(Lya_fts['LyaEW'][src]) # Obs frame EW by now
        Flambda[src] = Lya_fts['LyaF']

    EW0 /= 1 + z # Now it's rest frame EW0

    dL = cosmo.luminosity_distance(z).to(u.cm).value
    L = np.log10(Flambda * 4*np.pi * dL ** 2)

    return z, EW0, L

def main(part):
    filename = f'/home/alberto/cosmos/LAEs/MyMocks/QSO_100000'

    if not os.path.exists(filename):
        os.mkdir(filename)

    files = glob.glob('/home/alberto/almacen/SDSS_QSO_fits/fits/*')
    N_sources = len(files)

    pm_SEDs = np.empty((60, N_sources))
    mjd = np.zeros(N_sources).astype(int)
    plate = np.zeros(N_sources).astype(int)
    fiber = np.zeros(N_sources).astype(int)

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Do the integrated photometry
    print('Extracting band fluxes from the spectra...')
    for src in range(N_sources):
        print(f'{src} / {N_sources}', end='\r')
        spec_name = files[src]
        spec = Table.read(spec_name, hdu=1)
        pm_SEDs[:, src] = JPAS_synth_phot(
            spec['flux'] * 1e-17, 10 ** spec['loglam'], tcurves
        )

        mjd[src] = int(spec_name[-20 : -16])
        plate[src] = int(spec_name[-15 : -10])
        mjd[src] = int(spec_name[-9 : -5])

    print('Adding errors...')
    pm_SEDs, pm_SEDs_err = add_errors(pm_SEDs)

    print('Extracting line features...')
    z, EW0, L = SDSS_QSO_line_fts(mjd, plate, fiber)

    hdr = tcurves['tag'] + [s + '_e' for s in tcurves['tag']] + ['z', 'EW0', 'L_lya']

    pd.DataFrame(
        data=np.hstack(
            (
                pm_SEDs.T, pm_SEDs_err.T, z.reshape(-1, 1), EW0.reshape(-1, 1),
                L.reshape(-1, 1)
            )
        )
    ).to_csv(filename + '/data{part}.csv', header=hdr)

if __name__ == '__main__':
    t0 = perf_counter()
    main(sys.argv[1])
    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(perf_counter() - t0, 60)))