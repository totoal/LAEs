import glob
import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo

def load_QSO_mock(name, add_errs=True):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename + 'data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    data_qso = pd.concat(fi, axis=0, ignore_index=True)

    qso_flx = data_qso.to_numpy()[:, 1 : 60 + 1].T
    qso_err = data_qso.to_numpy()[:, 60 + 1 : 120 + 1].T

    if add_errs:
        qso_flx += qso_err * np.random.normal(size=qso_err.shape)

    qso_L = data_qso['L_lya'].to_numpy()
    EW_qso = data_qso['EW0'].to_numpy()
    qso_zspec = data_qso['z'].to_numpy()

    # Remove bad sources
    good_src = []
    for src in range(qso_err.shape[1]):
        bad_src = (
            np.any(qso_err[1:55, src] > 1) | np.any(qso_err[-3:, src] > 1)
            | np.all(qso_flx[:, src] == 0)
            | ((qso_L[src] > 0) & ((EW_qso[src] == 0) | (~np.isfinite(EW_qso[src]))))
            | (qso_zspec[src] > 2) & ~np.isfinite(EW_qso[src])
        )
        if bad_src:
            continue
        else:
            good_src.append(src)
    good_src = np.array(good_src)

    qso_flx[qso_err > 1] = 0.
    EW_qso[~np.isfinite(EW_qso)] = 0.

    qso_flx = qso_flx[:, good_src]
    qso_err = qso_err[:, good_src]

    EW_qso = EW_qso[good_src]
    qso_zspec = qso_zspec[good_src]
    qso_L = qso_L[good_src]

    return qso_flx, qso_err, EW_qso, qso_zspec, qso_L


def load_GAL_mock(name, add_errs=True):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename +'data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    data_gal = pd.concat(fi, axis=0, ignore_index=True)

    gal_flx = data_gal.to_numpy()[:, 1 : 60 + 1].T
    gal_err = data_gal.to_numpy()[:, 60 + 1 : 120 + 1].T

    if add_errs:
        gal_flx += gal_err * np.random.normal(size=gal_err.shape)

    # Remove bad sources
    good_src = []
    for src in range(gal_err.shape[1]):
        bad_src = (
            np.any(gal_err[1:55, src] > 1) | np.any(gal_err[-3:, src] > 1)
            & np.all(gal_flx[:, src] == 0)
        )
        if bad_src:
            continue
        else:
            good_src.append(src)
    good_src = np.array(good_src)

    gal_flx[gal_err > 1] = 0.

    gal_flx = gal_flx[:, good_src]
    gal_err = gal_err[:, good_src]

    EW_gal = np.zeros(data_gal['z'].to_numpy().shape)[good_src]
    gal_zspec = data_gal['z'].to_numpy()[good_src]
    gal_L = np.zeros(EW_gal.shape)

    return gal_flx, gal_err, EW_gal, gal_zspec, gal_L

def load_SF_mock(name, add_errs=True):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename +'data*')
    files.sort()
    fi = []

    for i, name in enumerate(files):
        if i == 10:
            break
        fi.append(pd.read_csv(name))

    data = pd.concat(fi, axis=0, ignore_index=True)

    sf_flx = data.to_numpy()[:, 1 : 60 + 1].T
    sf_err = data.to_numpy()[:, 60 + 1 : 120 + 1].T

    if add_errs:
        sf_flx += sf_err * np.random.normal(size=sf_err.shape)

    EW_sf = data['EW0'].to_numpy()
    sf_zspec = data['z'].to_numpy()
    sf_L = data['L_lya'].to_numpy()

    return sf_flx, sf_err, sf_zspec, EW_sf, sf_L

def ensemble_mock(name_qso, name_gal, name_sf):
    qso_flx, qso_err, EW_qso, qso_zspec, qso_L = load_QSO_mock(name_qso)
    gal_flx, gal_err, EW_gal, gal_zspec, gal_L = load_GAL_mock(name_gal)
    sf_flx, sf_err, sf_zspec, EW_sf, sf_L = load_SF_mock(name_sf)

    pm_flx = np.hstack((qso_flx, sf_flx, gal_flx))
    pm_err = np.hstack((qso_err, sf_err, gal_err))
    zspec = np.concatenate((qso_zspec, sf_zspec, gal_zspec))
    EW_lya = np.concatenate((EW_qso, EW_sf, EW_gal))

    N_sf = sf_flx.shape[1]
    N_qso = qso_flx.shape[1]
    N_gal = gal_flx.shape[1]

    L_lya = np.concatenate((qso_L, sf_L, gal_L))

    is_qso = np.concatenate((np.ones(N_qso), np.zeros(N_sf + N_gal))).astype(bool)
    is_sf = np.concatenate((np.zeros(N_qso), np.ones(N_sf), np.zeros(N_gal))).astype(bool)
    is_gal = np.concatenate((np.zeros(N_qso), np.zeros(N_sf), np.ones(N_gal))).astype(bool)

    return pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal