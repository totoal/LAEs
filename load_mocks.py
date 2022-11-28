import glob
import pandas as pd
import numpy as np

from my_functions import flux_to_mag, central_wavelength

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

w_central = central_wavelength()


def load_QSO_mock(name, add_errs=True, how_many=-1, mag_min=0, mag_max=99):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename + 'data*')
    files.sort()
    fi = []

    for i, name in enumerate(files):
        if i == how_many:
            break
        fi.append(pd.read_csv(name))

    data_qso = pd.concat(fi, axis=0, ignore_index=True)

    qso_flx = data_qso.to_numpy()[:, 1: 60 + 1].T
    # qso_err = data_qso.to_numpy()[:, 60 + 1: 120 + 1].T
    qso_err = np.zeros(qso_flx.shape)
    mag = flux_to_mag(qso_flx[-2], w_central[-2])

    if add_errs:
        qso_flx += qso_err * np.random.normal(size=qso_err.shape)

    qso_zspec = data_qso['z'].to_numpy()
    try:
        qso_L = data_qso['L_lya'].to_numpy()
        EW_qso = data_qso['EW0'].to_numpy()
        qso_L_NV = data_qso['L_NV'].to_numpy()
        EW_NV_qso = data_qso['EW0_NV'].to_numpy()
    except:
        qso_L = np.zeros(qso_zspec.shape)
        qso_L_NV = np.zeros(qso_zspec.shape)
        EW_qso = np.zeros(qso_zspec.shape)
        EW_NV_qso = np.zeros(qso_zspec.shape)

    # Remove bad sources
    good_src = []
    for src in range(qso_err.shape[1]):
        bad_src = (
            (mag[src] < mag_min - 0.25)
            | (mag[src] > mag_max + 0.25)
            | np.any(~np.isfinite(qso_flx[1:-4, src]))
        )
        if bad_src:
            continue
        else:
            good_src.append(src)
    good_src = np.array(good_src)
    # print(f'Bad QSO removed: {len(qso_L) - len(good_src)}')

    qso_flx[qso_err > 1] = 0.
    EW_qso[~np.isfinite(EW_qso)] = 0.
    EW_NV_qso[~np.isfinite(EW_NV_qso)] = 0.

    qso_flx = qso_flx[:, good_src].astype(float)
    qso_err = qso_err[:, good_src].astype(float)

    EW_qso = EW_qso[good_src].astype(float)
    EW_NV_qso = EW_NV_qso[good_src].astype(float)
    qso_zspec = qso_zspec[good_src].astype(float)
    qso_L = qso_L[good_src].astype(float)
    qso_L_NV = qso_L_NV[good_src].astype(float)

    return qso_flx, qso_err, EW_qso, qso_zspec, qso_L, qso_L_NV, EW_NV_qso


def angular_radius(R, z):
    '''
    Takes as input a distance R in comoving Mpc and a redshift z and returns the
    angular size in the sky.
    '''
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z)
    R = np.array(R) * u.Mpc * cosmo.h
    R_ang = R * arcsec_per_kpc
    return R_ang.to(u.deg).value


def load_GAL_mock(name, add_errs=False, mag_min=0, mag_max=99, return_src_list=False):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename + 'data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    data_gal = pd.concat(fi, axis=0, ignore_index=True)

    gal_flx = data_gal.to_numpy()[:, 1: 60 + 1].T
    gal_err = data_gal.to_numpy()[:, 60 + 1: 120 + 1].T

    if add_errs:
        gal_flx += gal_err * np.random.normal(size=gal_err.shape)

    gal_zspec = data_gal['z'].to_numpy().astype(float)

    # i = flux_to_mag(gal_flx[-1], w_central[-1])
    r = flux_to_mag(gal_flx[-2], w_central[-2])
    # g = flux_to_mag(gal_flx[-3], w_central[-3])
    # gr = g - r
    # ri = r - i
    # color_aux2 = (-1.5 * ri + 1.95 > gr) & (ri > 1.25) & (gr > 1.75)

    # Remove bad sources
    good_src = []
    for src in range(gal_err.shape[1]):
        bad_src = (
            (gal_zspec[src] > 2)
            # | color_aux2[src]
            | ((r[src] < mag_min - 0.4) | (r [src] > mag_max + 0.25))
        )
        if bad_src:
            continue
        else:
            good_src.append(src)
    good_src = np.array(good_src)
    # print(f'Bad GAL removed: {len(gal_zspec) - len(good_src)}')

    gal_flx[gal_err > 1] = 0.

    gal_flx = gal_flx[:, good_src].astype(float)
    gal_err = gal_err[:, good_src].astype(float)
    gal_zspec = gal_zspec[good_src]

    EW_gal = np.zeros(data_gal['z'].to_numpy().shape)[good_src].astype(float)
    gal_L = np.zeros(EW_gal.shape).astype(float)

    Rdisk = data_gal['Rdisk'][good_src]
    Rbulge = data_gal['Rbulge'][good_src]
    Mdisk = data_gal['Mdisk'][good_src]
    Mbulge = data_gal['Mbulge'][good_src]
    R_eff = (Rdisk * Mdisk + Rbulge * Mbulge) / (Mbulge + Mdisk)
    R_ang = angular_radius(R_eff, gal_zspec)

    if return_src_list:
        return gal_flx, gal_err, EW_gal, gal_zspec, gal_L, R_ang, good_src
    return gal_flx, gal_err, EW_gal, gal_zspec, gal_L, R_ang


def load_SF_mock(name, add_errs=True, how_many=-1, mag_min=0, mag_max=99):
    filename = f'/home/alberto/almacen/Source_cats/{name}/'
    files = glob.glob(filename + 'data*')
    files.sort()
    fi = []

    for i, name in enumerate(files):
        if i == how_many:
            break
        fi.append(pd.read_csv(name))

    data = pd.concat(fi, axis=0, ignore_index=True)

    sf_flx = data.to_numpy()[:, 1: 60 + 1].T.astype(float)
    sf_err = data.to_numpy()[:, 60 + 1: 120 + 1].T.astype(float)
    mag = flux_to_mag(sf_flx[-2], w_central[-2])

    if add_errs:
        sf_flx += sf_err * np.random.normal(size=sf_err.shape)

    EW_sf = data['EW0'].to_numpy().astype(float)
    sf_zspec = data['z'].to_numpy().astype(float)
    sf_L = data['L_lya'].to_numpy().astype(float)

    # Remove bad sources
    good_src = []
    for src in range(sf_err.shape[1]):
        bad_src = (mag[src] < mag_min - 0.25) | (mag[src] > mag_max + 0.25)
        if bad_src:
            continue
        else:
            good_src.append(src)
    good_src = np.array(good_src)
    # print(f'Bad SF removed: {len(sf_zspec) - len(good_src)}')

    sf_flx = sf_flx[:, good_src].astype(float)
    sf_err = sf_err[:, good_src].astype(float)
    sf_zspec = sf_zspec[good_src]
    EW_sf = EW_sf[good_src]
    sf_L = sf_L[good_src]

    return sf_flx, sf_err, sf_zspec, EW_sf, sf_L


def ensemble_mock(name_qso, name_gal, name_sf, name_qso_bad='', name_qso_hiL='',
                  add_errs=False, qso_LAE_frac=1., sf_frac=1., mag_min=0, mag_max=99,
                  how_many_sf=-1):
    qso_flx, qso_err, EW_qso, qso_zspec, qso_L, qso_L_NV, EW_NV_qso = load_QSO_mock(
        name_qso, add_errs, mag_min=mag_min, mag_max=mag_max)
    print('QSO mock loaded')
    gal_flx, gal_err, EW_gal, gal_zspec, gal_L, gal_R = load_GAL_mock(
        name_gal, add_errs, mag_min=mag_min, mag_max=mag_max)
    print('GAL mock loaded')
    sf_flx, sf_err, sf_zspec, EW_sf, sf_L = load_SF_mock(name_sf, add_errs, 
                                                         mag_min=mag_min, mag_max=mag_max,
                                                         how_many=how_many_sf)
    print('SFG mock loaded')

    # Truncate SF
    if sf_frac < 1:
        N_sf = sf_flx.shape[1]
        choice = np.random.choice(
            np.arange(N_sf), np.floor(N_sf * sf_frac).astype(int))
        sf_flx = sf_flx[:, choice]
        sf_err = sf_err[:, choice]
        EW_sf = EW_sf[choice]
        sf_zspec = sf_zspec[choice]
        sf_L = sf_L[choice]

    # If name_qso_bad given, load two catalogs of qso and give the relative
    # number: one with z < 2, another with z > 2
    if len(name_qso_bad) > 0:
        qso_flx_bad, qso_err_bad, EW_qso_bad, qso_zspec_bad, qso_L_bad, qso_L_NV_bad, EW_NV_qso_bad =\
            load_QSO_mock(name_qso_bad)

        # Truncate LAE QSOs
        if qso_LAE_frac < 1:
            N_qso = qso_flx_bad.shape[1]
            choice = np.random.choice(np.arange(N_qso), np.floor(
                N_qso * qso_LAE_frac).astype(int))
            qso_flx_bad = qso_flx_bad[:, choice]
            qso_err_bad = qso_err_bad[:, choice]
            EW_qso_bad = EW_qso_bad[choice]
            EW_NV_qso_bad = EW_NV_qso_bad[choice]
            qso_zspec_bad = qso_zspec_bad[choice]
            qso_L_bad = qso_L_bad[choice]
            qso_L_NV_bad = qso_L_NV_bad[choice]

        where_low_z = (qso_zspec < 2)
        qso_flx = np.hstack((qso_flx_bad, qso_flx[:, where_low_z]))
        qso_err = np.hstack((qso_err_bad, qso_err[:, where_low_z]))
        EW_qso = np.hstack((EW_qso_bad, EW_qso[where_low_z]))
        EW_NV_qso = np.hstack((EW_NV_qso_bad, EW_NV_qso[where_low_z]))
        qso_zspec = np.hstack((qso_zspec_bad, qso_zspec[where_low_z]))
        qso_L = np.hstack((qso_L_bad, qso_L[where_low_z]))
        qso_L_NV = np.hstack((qso_L_NV_bad, qso_L_NV[where_low_z]))

    if len(name_qso_hiL) > 0:
        qso_flx_hiL, qso_err_hiL, EW_qso_hiL, qso_zspec_hiL, qso_L_hiL, qso_L_NV_hiL, EW_NV_qso_hiL =\
            load_QSO_mock(name_qso_hiL)

        # Truncate LAE QSOs
        if qso_LAE_frac < 1:
            N_qso = qso_flx_hiL.shape[1]
            choice = np.random.choice(np.arange(N_qso),
                                      np.floor(N_qso * qso_LAE_frac).astype(int))
            qso_flx_hiL = qso_flx_hiL[:, choice]
            qso_err_hiL = qso_err_hiL[:, choice]
            EW_qso_hiL = EW_qso_hiL[choice]
            EW_NV_qso_hiL = EW_NV_qso_hiL[choice]
            qso_zspec_hiL = qso_zspec_hiL[choice]
            qso_L_hiL = qso_L_hiL[choice]
            qso_L_NV_hiL = qso_L_NV_hiL[choice]

        where_loL = (qso_L <= 44)
        qso_flx = np.hstack((qso_flx[:, where_loL], qso_flx_hiL))
        qso_err = np.hstack((qso_err[:, where_loL], qso_err_hiL))
        EW_qso = np.hstack((EW_qso[where_loL], EW_qso_hiL))
        EW_NV_qso = np.hstack((EW_NV_qso[where_loL], EW_NV_qso_hiL))
        qso_zspec = np.hstack((qso_zspec[where_loL], qso_zspec_hiL))
        qso_L = np.hstack((qso_L[where_loL], qso_L_hiL))
        qso_L_NV = np.hstack((qso_L_NV[where_loL], qso_L_NV_hiL))

    pm_flx = np.hstack((qso_flx, sf_flx, gal_flx))
    pm_err = np.hstack((qso_err, sf_err, gal_err))
    zspec = np.concatenate((qso_zspec, sf_zspec, gal_zspec))
    EW_lya = np.concatenate((EW_qso, EW_sf, EW_gal))

    N_sf = sf_flx.shape[1]
    N_qso = qso_flx.shape[1]
    N_gal = gal_flx.shape[1]

    L_lya = np.concatenate((qso_L, sf_L, gal_L))
    L_NV = np.concatenate((qso_L_NV, np.zeros_like(sf_L), np.zeros_like(gal_L)))
    EW_NV = np.concatenate((EW_NV_qso, np.zeros_like(EW_sf), np.zeros_like(EW_gal)))

    is_qso = np.concatenate(
        (np.ones(N_qso), np.zeros(N_sf + N_gal))).astype(bool)
    is_sf = np.concatenate(
        (np.zeros(N_qso), np.ones(N_sf), np.zeros(N_gal))).astype(bool)
    is_gal = np.concatenate(
        (np.zeros(N_qso), np.zeros(N_sf), np.ones(N_gal))).astype(bool)
    is_LAE = (is_qso & (zspec > 2)) | is_sf
    if len(name_qso_hiL) > 0:
        where_hiL = (is_qso & (L_lya > 44))
    else:
        where_hiL = np.zeros_like(is_qso).astype(bool)

    ang_R = np.zeros(EW_lya.shape)
    ang_R[is_gal] = gal_R

    return pm_flx, pm_err, zspec, EW_lya, L_lya.astype(float), is_qso,\
        is_sf, is_gal, is_LAE, where_hiL, ang_R, L_NV, EW_NV
