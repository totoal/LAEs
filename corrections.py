import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

import pandas as pd

from my_functions import *

import glob

from scipy.integrate import simpson

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from time import perf_counter

# Supress warnings because fuck them
import warnings
warnings.filterwarnings("ignore")

global w_lya
w_lya = 1215.67

def load_mock(w_central, nb_fwhm_Arr):

    t0 = perf_counter()
    print('Loading QSO...\r')

    ## Load QSO catalog
    filename = ('/home/alberto/cosmos/JPAS_mocks_sep2021/'
        'JPAS_mocks_classification_01sep_model11/Fluxes/Qso_jpas_mock_flam_train.cat')

    my_filter_order = np.arange(60)
    my_filter_order[[-4, -3, -2, -1]] = np.array([1, 12, 28, 43])
    my_filter_order[1:-4] += 1
    my_filter_order[12:-4] += 1
    my_filter_order[28:-4] += 1
    my_filter_order[43:-4] += 1

    qso_flx = pd.read_csv(
        filename, sep=' ', usecols=range(2, 2 + 60)
    ).to_numpy().T[my_filter_order]
    qso_err = pd.read_csv(
        filename, sep=' ', usecols=range(2 + 60, 2 + 60 + 60)
    ).to_numpy().T[my_filter_order]
    qso_zspec = pd.read_csv(filename, sep=' ', usecols=[127]).to_numpy().reshape(-1, )

    # Randomly sample sources corresponding to 200 deg2
    # idx = np.random.randint(0, 100000, 510 * 200)
    idx = np.arange(100_000)
    qso_flx = qso_flx[:, idx]
    qso_err = qso_err[:, idx]
    qso_zspec = qso_zspec[idx]

    Lya_fts = pd.read_csv('csv/Lya_fts.csv')
    EW_qso = np.abs(Lya_fts.LyaEW)[idx] / (qso_zspec + 1)

    # Apply errors
    np.random.seed(22)
    qso_flx += qso_err * np.random.normal(size=qso_err.shape)

    print('Done in {0:0.1f} s'.format(perf_counter() - t0))
    
    t0 = perf_counter()
    print('Loading SF...')

    ## Load SF catalog

    filename = '/home/alberto/almacen/Source_cats/LAE_10deg_z2-5/'
    files = glob.glob(filename +'data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    data = pd.concat(fi, axis=0, ignore_index=True)

    sf_flx = data.to_numpy()[:, 1 : 60 + 1].T
    sf_err = data.to_numpy()[:, 60 + 1 : 120 + 1].T

    mag_noerr = flux_to_mag(sf_flx, w_central.reshape(-1, 1))
    mag_noerr[np.isnan(mag_noerr)] = 99.

    sf_flx += np.random.normal(size=(sf_err.shape)) * sf_err

    mag = flux_to_mag(sf_flx, w_central.reshape(-1, 1))
    mag[np.isnan(mag)] = 99.

    files2 = []
    files3 = []
    for i in range(len(files)):
        files2.append(f'{filename}SEDs{i + 1}.csv')
        files2.sort()
        files3.append(f'{filename}SEDs_no_line{i + 1}.csv')
        files3.sort()
    fi = []
    for name in files2:
        fi.append(pd.read_csv(name, header=None))
    fi3 = []
    for name in files3:
        fi3.append(pd.read_csv(name, header=None))

    EW_sf = data['EW0'].to_numpy()
    sf_zspec = data['z'].to_numpy()

    pm_flx = np.hstack((qso_flx, sf_flx))
    pm_err = np.hstack((qso_err, sf_err))
    zspec = np.concatenate((qso_zspec, sf_zspec))
    EW_lya = np.concatenate((EW_qso, EW_sf))

    N_sf = sf_flx.shape[1]
    N_qso = qso_flx.shape[1]

    is_qso = np.concatenate((np.ones(N_qso), np.zeros(N_sf))).astype(bool)

    qso_dL = cosmo.luminosity_distance(qso_zspec).to(u.cm).value
    sf_dL = cosmo.luminosity_distance(sf_zspec).to(u.cm).value

    qso_L = np.log10(EW_qso * Lya_fts.LyaCont[idx] * 1e-17 * (4 * np.pi * qso_dL**2))

    sf_L = data['L_lya'].to_numpy()

    sf_flambda = 10 ** sf_L / (4*np.pi * sf_dL **2)
    qso_flambda = Lya_fts.LyaF * 1e-17

    L_lya = np.concatenate((qso_L, sf_L))
    F_lya = np.concatenate((qso_flambda, sf_flambda))

    print('Done in {0:0.1f} s'.format(perf_counter() - t0))

    return pm_flx, pm_err, zspec, EW_lya, is_qso, L_lya, F_lya

def look_for_lines(EW0_lya_cut, EW_other_cut, pm_flx, pm_err, zspec):
    # Lya search
    cont_est_lya, cont_err_lya = estimate_continuum(pm_flx, pm_err, IGM_T_correct=True)
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, EW0_lya_cut)
    lya_lines, lya_cont_lines = identify_lines(line, pm_flx, pm_err, first=True)

    # Other lines
    cont_est_other, cont_err_other = estimate_continuum(pm_flx, pm_err, IGM_T_correct=False)
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
        EW_other_cut, obs=True)
    other_lines = identify_lines(line_other, pm_flx, pm_err)

    # Compute z
    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    nice_z = np.abs(z_Arr - zspec) < 0.12

    return cont_est_lya, cont_err_lya, lya_lines, other_lines, z_Arr, nice_z

def L_bias_corrections(L_lya, L_nb, lya_lines, nice_lya, nice_z):
    
    N_L_bins = 30
    L_bins = np.linspace(42, 45.5, N_L_bins + 1)
    L_bins_c = [(L_bins[i] + L_bins[i + 1]) / 2 for i in range(N_L_bins)]

    nb_min = 5
    nb_max = 25

    nb_list = np.arange(nb_min, nb_max)
    L_corr_Arr = np.empty((len(nb_list), N_L_bins))

    for j, nb_idx in enumerate(nb_list):
        L_corr = np.empty(N_L_bins)

        for i in range(N_L_bins):
            L_bin = (
                (L_nb > L_bins[i]) & (L_nb < L_bins[i + 1])
                & (lya_lines == nb_idx)
                & nice_lya & nice_z
            )
            L_corr[i] = np.median((L_nb - L_lya)[L_bin])

        L_corr_Arr[j] = L_corr
    
    data = np.hstack(
        (nb_list.reshape(-1, 1), L_corr_Arr)
    )
    columns = ['NB'] + [str(c) for c in L_bins_c]

    out_L_corr = pd.DataFrame(data=data, columns=columns)

    return out_L_corr

def completeness_corrections(L_lya, L_nb, nice_lya, nice_z):
    # First compute the total sources expected from the input L_lya values of the mock
    N_bins = 20
    bins = np.linspace(42, 45.5, N_bins + 1)

    nb_min = 5
    nb_max = 25
    nb_list = np.arange(nb_min, nb_max)

    correct = np.empty((len(nb_list), N_bins))

    for j, nb_idx in enumerate(range(nb_min, nb_max)):

        

        h_input, b = np.histogram(L_lya[zspec_cut], bins)
        b_c = [0.5 * (b[i] + b[i+1]) for i in range(len(b) - 1)]
        bw = b[1] - b[0]

        totals = []
        for b_i, _ in enumerate(b_c):
            Lx = np.linspace(b[b_i], b[b_i + 1], 100)

            totals.append(
                simpson(
                    np.interp(
                        Lx, b_c, h_input / bw
                    ),
                    Lx
                )
            )
        totals = np.array(totals)

        this_nb = (np.array(lya_lines) == nb_idx)
        goodh = L_nb[nice_lya & nice_z & this_nb]
        badh = L_nb[nice_lya & ~nice_z & this_nb]

        hg, _ = np.histogram(goodh, bins=bins)
        hb, _ = np.histogram(badh, bins=bins)

        correct[j] = hg / (hg + hb) / (hg / totals) 

    
    data = np.hstack(
        (nb_list.reshape(-1, 1), correct)
    )
    columns = ['NB'] + [str(c) for c in b_c]

    out_puricomp_corr = pd.DataFrame(data=data, columns=columns)

    return out_puricomp_corr

## MAIN ##
if __name__ == '__main__':
    print('\n\n')
    t00 = perf_counter()
    w_central = central_wavelength()
    nb_fwhm_Arr = nb_fwhm(range(60))
    w_lya = 1215.67

    pm_flx, pm_err, zspec, EW_lya, is_qso, L_lya, F_lya = load_mock(w_central, nb_fwhm_Arr)

    N_sources = pm_flx.shape[1]
    print(f'N_sources = {N_sources}')

    # Magnitudes Array
    mag = flux_to_mag(pm_flx, w_central.reshape(-1, 1))
    mag[np.isnan(mag)] = 99.

    ###############################333333

    t0 = perf_counter()
    print('Looking for lines...')

    EW0_lya_cut = 30
    EW_other_cut = 200

    cont, cont_err, lya_lines, other_lines, z_nb_Arr, nice_z =\
        look_for_lines(EW0_lya_cut, EW_other_cut, pm_flx, pm_err, zspec)

    mag_min = 17
    mag_max = 25
    mag_cut = (mag[-2] > mag_min) & (mag[-2] < mag_max)

    nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont, z_nb_Arr
    ) & mag_cut

    print('Done in {0:0.1f} s'.format(perf_counter() - t0))

    t0 = perf_counter()
    print('Computing Line Luminosities...')

    EW_nb_Arr, _, L_nb = EW_L_NB(pm_flx, pm_err, cont, cont_err, z_nb_Arr, lya_lines)

    print('Done in {0:0.1f} s'.format(perf_counter() - t0))

    t0 = perf_counter()
    print('Obtaining L biases')

    L_corr = L_bias_corrections(L_lya, L_nb, lya_lines, nice_lya, nice_z)

    print('Done in {0:0.1f} s'.format(perf_counter() - t0))

    t0 = perf_counter()
    print('Obtaining purity/completeness corrections')

    puricomp_corr = completeness_corrections(L_lya, L_nb, nice_lya, nice_z)

    print('Done in {0:0.1f} s'.format(perf_counter() - t0))



    print('All done in {0:0.0f} m {1:0.0f} s !'
        .format(*divmod(int(perf_counter() - t00), 60)))