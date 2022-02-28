import sys
import os
import glob

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

def main(part):
    filename = f'/home/alberto/cosmos/LAEs/MyMocks/QSO_100000'

    if not os.path.exists(filename):
        os.mkdir(filename)

    files = glob.glob('/home/alberto/cosmos/SDSS_Spectra/fits/*')
    N_sources = len(files)

    pm_SEDs = np.empty((60, N_sources))

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    # Do the integrated photometry
    for src in N_sources:
        spec_name = files[src]
        spec = Table.read(spec_name)
        pm_SEDs[:, src] = JPAS_synth_phot(
            spec['flux'] * 1e-17, 10 ** spec['loglam'], tcurves
        )

    pm_SEDs, pm_SEDs_err = add_errors(pm_SEDs)

    hdr = tcurves['tag'] + [s + '_e' for s in tcurves['tag']]

    pd.DataFrame(
            data=np.hstack((pm_SEDs.T, pm_SEDs_err.T))
    ).to_csv(filename + f'/data{part}.csv', header=hdr)

if __name__ == '__main__':
    main(sys.argv[1])