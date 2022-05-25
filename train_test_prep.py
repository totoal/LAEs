#!/home/alberto/miniconda3/bin/python3

# import sys
import time

import numpy as np

from my_functions import *
from load_mocks import load_QSO_mock, load_SF_mock

import pandas as pd

w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67

def nice_lya_search(flx, err, L_lya, mag_min, mag_max, nb_Arr):
    # Lya search
    cont_est_lya, cont_err_lya = estimate_continuum(flx, err, IGM_T_correct=True)
    line = is_there_line(flx, err, cont_est_lya, cont_err_lya, 30)
    lya_lines, lya_cont_lines, _ = identify_lines(
        line, flx, err, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    cont_est_other, cont_err_other = estimate_continuum(flx, err, IGM_T_correct=False)
    line_other = is_there_line(flx, err, cont_est_other, cont_err_other,
        400, obs=True)
    other_lines = identify_lines(line_other, flx, err)

    # Compute z
    N_sources = flx.shape[1]
    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    nb_min = 3
    nb_max = 20

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    z_cut = (z_min < z_Arr) & (z_Arr < z_max)

    mag = flux_to_mag(flx[-2], w_central[-2])
    mag[np.isnan(mag)] = 99.

    mask_nice_lya = np.zeros(len(lya_lines)).astype(bool)
    for nb in nb_Arr:
        mask_nice_lya = mask_nice_lya | (lya_lines == nb)

    nice_lya = nice_lya_select(
        lya_lines, other_lines, flx, err, cont_est_lya, z_Arr,
        mask_nice_lya
    )
    nice_lya = (
        z_cut
        & (L_lya > 0)
        & (mag > mag_min)
        & (mag < mag_max)
        & nice_lya
    )

    _, _, L_Arr, _, _, _ = EW_L_NB(
        flx, err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    return nice_lya, z_Arr, L_Arr

def sample_sources(flx, err, L_lya, mag_min, mag_max,
                   nb_Arr, N_samples=100_000):
    out_flx = np.array([])
    n_iter = 0
    while True:
        n_iter += 1
        this_flx = flx + err * np.random.normal(size=err.shape)

        this_flx[err > 1] = 0.

        this_nice_lya, this_z_Arr, this_L_Arr = nice_lya_search(
            this_flx, err, L_lya, mag_min, mag_max, nb_Arr
        )

        if len(out_flx) == 0:
            out_flx = flx[:, this_nice_lya]
            out_err = err[:, this_nice_lya]
            out_z = this_z_Arr[this_nice_lya]
            out_L = this_L_Arr[this_nice_lya]
            out_L_lya = L_lya[this_nice_lya]
        else:
            out_flx = np.hstack((out_flx, this_flx[:, this_nice_lya]))
            out_err = np.hstack((out_err, err[:, this_nice_lya]))
            out_z = np.concatenate((out_z, this_z_Arr[this_nice_lya]))
            out_L = np.concatenate((out_L, this_L_Arr[this_nice_lya]))
            out_L_lya = np.concatenate((out_L_lya, L_lya[this_nice_lya]))
        
        
        if len(out_z) >= N_samples:
            break

    randomize = np.random.choice(np.arange(0, len(out_L)), N_samples)

    out_flx = out_flx[:, randomize]
    out_err = out_err[:, randomize]
    out_z = out_z[randomize]
    out_L = out_L[randomize]
    out_L_lya = out_L_lya[randomize]

    return out_flx, out_err, out_z, out_L, out_L_lya

def ensemble_dataset(qso_flx, qso_err, sf_flx, sf_err, qso_L, sf_L,
                     N_samples, mag_min, mag_max, nb_Arr):
    N_sources_qso = qso_flx.shape[1]
    N_sources_sf = sf_flx.shape[1]
    random_perm_qso = np.random.permutation(np.arange(N_sources_qso))
    random_perm_sf = np.random.permutation(np.arange(N_sources_sf))

    dataset = np.array([]).reshape(0, 114)
    L_labels = np.array([])

    n_folds = 3
    N_sources_k_qso = N_sources_qso // n_folds
    N_sources_k_sf = N_sources_sf // n_folds
    N_samples_k = N_samples // n_folds
    for k in range(n_folds):
        print(f'Fold #{k + 1}')
        fold_idx_qso = random_perm_qso[k * N_sources_k_qso : (k + 1) * N_sources_k_qso]
        fold_idx_sf = random_perm_sf[k * N_sources_k_sf : (k + 1) * N_sources_k_sf]

        sf_flx_data, sf_err_data, sf_z_data, sf_L_data, sf_L_Lya_data =\
            sample_sources(
                sf_flx[:, fold_idx_sf], sf_err[:, fold_idx_sf], sf_L[fold_idx_sf],
                mag_min, mag_max, nb_Arr, N_samples_k // 2
            )
        qso_flx_data, qso_err_data, qso_z_data, qso_L_data, qso_L_Lya_data =\
            sample_sources(
                qso_flx[:, fold_idx_qso], qso_err[:, fold_idx_qso], qso_L[fold_idx_qso],
                mag_min, mag_max, nb_Arr, N_samples_k // 2
            )

        pm_flx = np.hstack((qso_flx_data, sf_flx_data))
        pm_err = np.hstack((qso_err_data, sf_err_data))
        z_Arr = np.concatenate((qso_z_data, sf_z_data))
        L_Arr = np.concatenate((qso_L_data, sf_L_data))

        L_labels_k = np.concatenate((qso_L_Lya_data, sf_L_Lya_data))

        dataset_k = np.hstack(
            (
                pm_flx[2:55].T,
                pm_flx[-3:].T,
                np.abs(pm_err[2:55].T / pm_flx[2:55].T),
                np.abs(pm_err[-3:].T / pm_flx[-3:].T),
                L_Arr.reshape(-1, 1),
                z_Arr.reshape(-1, 1)
            )
        )

        dataset = np.vstack((dataset, dataset_k))
        L_labels = np.concatenate((L_labels, L_labels_k))

    return dataset, L_labels

def make_set(train_or_test, mag_min, mag_max, nb_Arr):
    if train_or_test == 'test':
        qso_flx, qso_err, _, _, qso_L =\
            load_QSO_mock('QSO_double_test_0', add_errs=False)
        sf_flx, sf_err, _, _, sf_L =\
            load_SF_mock('LAE_12.5deg_z2-4.25_test_0', add_errs=False)
    elif train_or_test == 'train':
        qso_flx, qso_err, _, _, qso_L =\
            load_QSO_mock('QSO_double_train_0', add_errs=False)
        sf_flx, sf_err, _, _, sf_L =\
            load_SF_mock('LAE_12.5deg_z2-4.25_train_0', add_errs=False)
    else:
        raise ValueError('Set name must be "train" or "test"')

    # Remove sources out of mag range
    mag_qso = flux_to_mag(qso_flx[-2], w_central[-2])
    mag_sf = flux_to_mag(sf_flx[-2], w_central[-2])
    mag_qso[np.isnan(mag_qso)] = 99.
    mag_sf[np.isnan(mag_sf)] = 99.

    mask_mag_qso = (mag_qso > mag_min) & (mag_qso < mag_max)
    mask_mag_sf = (mag_sf > mag_min) & (mag_sf < mag_max)

    qso_flx = qso_flx[:, mask_mag_qso]
    qso_err = qso_err[:, mask_mag_qso]
    qso_L = qso_L[mask_mag_qso]
    sf_flx = sf_flx[:, mask_mag_sf]
    sf_err = sf_err[:, mask_mag_sf]
    sf_L = sf_L[mask_mag_sf]

    # Ensemble the dataset
    dataset, L_labels = ensemble_dataset(qso_flx, qso_err, sf_flx, sf_err,
                                         qso_L, sf_L, 20_000, mag_min, mag_max,
                                         nb_Arr)

    set_name = f'mag{mag_min}-{mag_max}_nb{nb[0]}_{train_or_test}.csv'
    dataset_filename = f'MLmodels/datasets/dataset_{set_name}'
    L_labels_filename = f'MLmodels/datasets/labels_{set_name}'
    pd.DataFrame(dataset).to_csv(dataset_filename)
    pd.DataFrame(L_labels).to_csv(L_labels_filename)

if __name__ == '__main__':
    # nb = np.atleast_1d(sys.argv[1])
    mag_min_Arr = [15, 23]
    mag_max_Arr = [23, 23.5]

    t0 = time.time()
    for nb in range(5, 24):
        nb = np.atleast_1d(nb)
        for train_or_test in ['train', 'test']:
            for mag_min, mag_max in zip(mag_min_Arr, mag_max_Arr):
                print(f'Generating: mag{mag_min}-{mag_max}_{train_or_test}_nb{nb[0]}')
                make_set(train_or_test, mag_min, mag_max, nb)

                print('Done in: {0:0.0f} m {1:0.1f} s'.format(*divmod(time.time() - t0, 60)))