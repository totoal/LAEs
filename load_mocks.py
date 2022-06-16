import glob
import pandas as pd
import numpy as np

from my_functions import count_true

def load_QSO_mock(name, add_errs=True, how_many=-1):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename + 'data*')
    files.sort()
    fi = []

    for i, name in enumerate(files):
        if i == how_many:
            break
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

    qso_flx = qso_flx[:, good_src].astype(float)
    qso_err = qso_err[:, good_src].astype(float)

    EW_qso = EW_qso[good_src].astype(float)
    qso_zspec = qso_zspec[good_src].astype(float)
    qso_L = qso_L[good_src].astype(float)

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

    gal_flx = gal_flx[:, good_src].astype(float)
    gal_err = gal_err[:, good_src].astype(float)

    EW_gal = np.zeros(data_gal['z'].to_numpy().shape)[good_src].astype(float)
    gal_zspec = data_gal['z'].to_numpy()[good_src].astype(float)
    gal_L = np.zeros(EW_gal.shape).astype(float)

    # GAL mock is for contaminants, so we want sources with z>2 only
    where_low_z = (gal_zspec < 8)
    gal_flx = gal_flx[:, where_low_z]
    gal_err = gal_err[:, where_low_z]
    EW_gal = EW_gal[where_low_z]
    gal_zspec = gal_zspec[where_low_z]
    gal_L = gal_L[where_low_z]

    return gal_flx, gal_err, EW_gal, gal_zspec, gal_L

def load_SF_mock(name, add_errs=True, how_many=-1):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename +'data*')
    files.sort()
    fi = []

    for i, name in enumerate(files):
        if i == how_many:
            break
        fi.append(pd.read_csv(name))

    data = pd.concat(fi, axis=0, ignore_index=True)

    sf_flx = data.to_numpy()[:, 1 : 60 + 1].T.astype(float)
    sf_err = data.to_numpy()[:, 60 + 1 : 120 + 1].T.astype(float)

    if add_errs:
        sf_flx += sf_err * np.random.normal(size=sf_err.shape)

    EW_sf = data['EW0'].to_numpy().astype(float)
    sf_zspec = data['z'].to_numpy().astype(float)
    sf_L = data['L_lya'].to_numpy().astype(float)

    return sf_flx, sf_err, sf_zspec, EW_sf, sf_L

def ensemble_mock(name_qso, name_gal, name_sf, name_qso_bad='', name_qso_hiL=''):
    qso_flx, qso_err, EW_qso, qso_zspec, qso_L = load_QSO_mock(name_qso)
    gal_flx, gal_err, EW_gal, gal_zspec, gal_L = load_GAL_mock(name_gal)
    sf_flx, sf_err, sf_zspec, EW_sf, sf_L = load_SF_mock(name_sf)

    # If name_qso_bad given, load two catalogs of qso and give the relative
    # number: one with z < 2, another with z > 2
    if len(name_qso_bad) > 0:
        qso_flx_bad, qso_err_bad, EW_qso_bad, qso_zspec_bad, qso_L_bad =\
            load_QSO_mock(name_qso_bad)

        where_bad_qso = (qso_zspec_bad < 2)
        qso_flx = np.hstack((qso_flx_bad[:, where_bad_qso], qso_flx))
        qso_err = np.hstack((qso_err_bad[:, where_bad_qso], qso_err))
        EW_qso = np.hstack((EW_qso_bad[where_bad_qso], EW_qso))
        qso_zspec = np.hstack((qso_zspec_bad[where_bad_qso], qso_zspec))
        qso_L = np.hstack((qso_L_bad[where_bad_qso], qso_L))
    if len(name_qso_hiL) > 0:
        qso_flx_hiL, qso_err_hiL, EW_qso_hiL, qso_zspec_hiL, qso_L_hiL =\
            load_QSO_mock(name_qso_hiL)
        
        where_bad_loL = (qso_L <= 44)
        qso_flx = np.hstack((qso_flx[:, where_bad_loL], qso_flx_hiL))
        qso_err = np.hstack((qso_err[:, where_bad_loL], qso_err_hiL))
        EW_qso = np.hstack((EW_qso[where_bad_loL], EW_qso_hiL))
        qso_zspec = np.hstack((qso_zspec[where_bad_loL], qso_zspec_hiL))
        qso_L = np.hstack((qso_L[where_bad_loL], qso_L_hiL))

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
    is_LAE = (is_qso & (zspec > 2)) | is_sf
    where_hiL = (is_qso & (L_lya > 44))

    return pm_flx, pm_err, zspec, EW_lya, L_lya.astype(float), is_qso,\
        is_sf, is_gal, is_LAE, where_hiL