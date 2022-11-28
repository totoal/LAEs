#!/home/alberto/miniconda3/bin/python3

import numpy as np
import pickle

from scipy.stats import binned_statistic

import os
import time

from three_filter import cont_est_3FM
from LumFunc_miniJPAS import LF_perturb_err
from load_jpas_catalogs import load_minijpas_jnep
from load_mocks import ensemble_mock
from my_functions import *
from add_errors import add_errors

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 13})
matplotlib.use('Agg')

np.seterr(all='ignore')

# Useful definitions
w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67
filter_tags = load_filter_tags()

z_nb_Arr = w_central[:-4] / w_lya - 1

sf_frac = 0.1


def load_mocks(add_errs=True, qso_LAE_frac=1., 
               mag_min=0, mag_max=99):
    name_qso = 'QSO_100000_0'
    name_qso_hiL = f'QSO_double_train_jnep_DR16_highL_good2_0'
    name_qso_bad = f'QSO_double_train_jnep_DR16_good2_0'
    name_gal = f'GAL_LC_lines_0'
    name_sf = f'LAE_12.5deg_z2-4.25_train_minijpas_VUDS_0'

    pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal,\
        is_LAE, where_hiL, _, L_NV = ensemble_mock(name_qso, name_gal, name_sf,
                                             name_qso_bad, name_qso_hiL, add_errs,
                                             qso_LAE_frac, sf_frac, mag_min, mag_max)

    N_gal = count_true(is_gal)
    N_qso_cont = count_true(is_qso & ~is_LAE)
    N_qso_loL = count_true(is_qso & ~where_hiL & is_LAE)
    N_qso_hiL = count_true(is_qso & where_hiL)
    N_sf = count_true(is_sf)
    print(f'N_gal = {N_gal}, N_qso_cont = {N_qso_cont}, N_qso_loL = {N_qso_loL}, '
          f'N_qso_hiL = {N_qso_hiL}, N_sf = {N_sf}')

    return pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal, is_LAE, where_hiL,\
        L_NV


def nb_or_3fm_cont(pm_flx, pm_err, cont_est_m):
    if cont_est_m == 'nb':
        est_lya, err_lya = estimate_continuum(
            pm_flx, pm_err, IGM_T_correct=True)
        est_oth, err_oth = estimate_continuum(
            pm_flx, pm_err, IGM_T_correct=False)
    elif cont_est_m == '3fm':
        est_lya, err_lya = cont_est_3FM(pm_flx, pm_err, np.arange(1, 28))
        est_oth = est_lya
        err_oth = err_lya
    else:
        print('Not a valid continuum estimation method')
    return est_lya, err_lya, est_oth, err_oth


def search_lines(pm_flx, pm_err, ew0_cut, ew_obs, zspec, cont_est_m):
    cont_est_lya, cont_err_lya, cont_est_other, cont_err_other =\
        nb_or_3fm_cont(pm_flx, pm_err, cont_est_m)

    # Lya search
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut)
    lya_lines, lya_cont_lines, _ = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True)
    lya_lines = np.array(lya_lines)

    # Other lines
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
                               ew_obs, obs=True, sigma=5)
    other_lines = identify_lines(line_other, pm_flx, cont_est_other)

    N_sources = pm_flx.shape[1]

    # Compute z
    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    nice_z = np.abs(z_Arr - zspec) < 0.16

    return cont_est_lya, cont_err_lya, lya_lines, other_lines, z_Arr, nice_z


def compute_L_Lbin_err(L_Arr, L_lya, L_binning):
    '''
    Computes the errors due to dispersion of L_retrieved with some L_retrieved binning
    '''
    L_Lbin_err_plus = np.ones(len(L_binning) - 1) * np.inf
    L_Lbin_err_minus = np.ones(len(L_binning) - 1) * np.inf
    median = np.ones(len(L_binning) - 1) * np.inf
    last = [np.inf, np.inf]
    for i in range(len(L_binning) - 1):
        in_bin = (10 ** L_Arr >= L_binning[i]) & (10 ** L_Arr < L_binning[i + 1])
        if count_true(in_bin) == 0:
            L_Lbin_err_plus[i] = last[0]
            L_Lbin_err_minus[i] = last[1]
            continue
        perc = np.nanpercentile((L_Arr - L_lya)[in_bin], [16, 50, 84])
        L_Lbin_err_plus[i] = perc[2] - perc[1]
        L_Lbin_err_minus[i] = perc[1] - perc[0]

        last = [L_Lbin_err_plus[i], L_Lbin_err_minus[i]]
        median[i] = perc[1]

    return L_Lbin_err_plus, L_Lbin_err_minus, median

def compute_EW_bin_err(EW_Arr, EW_lya, EW_binning):
    '''
    Computes the errors due to dispersion of L_retrieved with some L_retrieved binning
    '''
    EW_bin_err_plus = np.ones(len(EW_binning) - 1) * 99
    EW_bin_err_minus = np.ones(len(EW_binning) - 1) * 99
    median = np.ones(len(EW_binning) - 1) * 99
    last = [0., 0.]
    for i in range(len(EW_binning) - 1):
        in_bin = (EW_Arr >= EW_binning[i]) & (EW_Arr < EW_binning[i + 1])
        if count_true(in_bin) == 0:
            EW_bin_err_plus[i] = last[0]
            EW_bin_err_minus[i] = last[1]
            continue
        perc = np.nanpercentile((EW_Arr - EW_lya)[in_bin], [16, 50, 84])
        EW_bin_err_plus[i] = perc[2] - perc[1]

        last = [EW_bin_err_plus[i], EW_bin_err_minus[i]]
        median[i] = perc[1]

    return EW_bin_err_plus, median


def purity_or_completeness_plot(mag, nbs_to_consider, lya_lines,
                                nice_lya, nice_z, L_Arr, mag_max,
                                mag_min, ew0_cut, is_gal, is_sf, is_qso, is_LAE,
                                zspec, L_lya, dirname, ew_cut, where_hiL, survey_name):
    fig, ax = plt.subplots(figsize=(8, 4))

    bins2 = np.linspace(42, 45.5, 15)

    b_c = [0.5 * (bins2[i] + bins2[i + 1]) for i in range(len(bins2) - 1)]

    this_mag_cut = (mag < mag_max) & (mag > mag_min)

    nb_min = nbs_to_consider[0]
    nb_max = nbs_to_consider[-1]
    nb_mask = (lya_lines >= nb_min) & (lya_lines <= nb_max)
    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1
    this_zspec_cut = (z_min < zspec) & (zspec < z_max)

    totals_mask = this_zspec_cut & this_mag_cut & ew_cut

    goodh_puri_sf = L_Arr[nice_lya & nice_z &
                          is_sf & ew_cut & this_mag_cut & nb_mask]
    goodh_puri_qso_loL = L_Arr[nice_lya & nice_z &
                               is_qso & ~where_hiL & ew_cut & this_mag_cut & nb_mask]
    goodh_puri_qso_hiL = L_Arr[nice_lya & nice_z &
                               is_qso & where_hiL & ew_cut & this_mag_cut & nb_mask]
    goodh_comp_sf = L_lya[nice_lya & nice_z & is_sf & totals_mask]
    goodh_comp_qso_loL = L_lya[nice_lya & nice_z &
                               is_qso & ~where_hiL & totals_mask]
    goodh_comp_qso_hiL = L_lya[nice_lya &
                               nice_z & is_qso & where_hiL & totals_mask]
    badh_to_corr = L_Arr[nice_lya & ~nice_z & (
        is_qso & is_LAE & ~where_hiL) & nb_mask & this_mag_cut]
    badh_normal_qso = L_Arr[nice_lya & ~nice_z & (is_qso & ~is_LAE) & nb_mask & this_mag_cut]
    badh_sf = L_Arr[nice_lya & ~nice_z & is_sf & nb_mask & this_mag_cut]
    badh_gal = L_Arr[nice_lya & ~nice_z & is_gal & nb_mask & this_mag_cut]

    hg_puri_sf, _ = np.histogram(goodh_puri_sf, bins=bins2)
    hg_puri_qso_loL, _ = np.histogram(goodh_puri_qso_loL, bins=bins2)
    hg_puri_qso_hiL, _ = np.histogram(goodh_puri_qso_hiL, bins=bins2)
    hg_comp_sf, _ = np.histogram(goodh_comp_sf, bins=bins2)
    hg_comp_qso_loL, _ = np.histogram(goodh_comp_qso_loL, bins=bins2)
    hg_comp_qso_hiL, _ = np.histogram(goodh_comp_qso_hiL, bins=bins2)
    hb_to_corr, _ = np.histogram(badh_to_corr, bins=bins2)
    hb_normal_qso, _ = np.histogram(badh_normal_qso, bins=bins2)
    hb_sf, _ = np.histogram(badh_sf, bins=bins2)
    hb_gal, _ = np.histogram(badh_gal, bins=bins2)

    hg_puri = hg_puri_sf * sf_factor + hg_puri_qso_loL * \
        good_qso_factor + hg_puri_qso_hiL * hiL_factor
    hg_comp = hg_comp_sf * sf_factor + hg_comp_qso_loL * \
        good_qso_factor + hg_comp_qso_hiL * hiL_factor
    hb = hb_normal_qso + hb_sf * sf_factor + hb_to_corr * good_qso_factor + hb_gal * gal_factor
    totals_sf, _ = np.histogram(L_lya[totals_mask & is_sf], bins=bins2)
    totals_qso_loL, _ = np.histogram(
        L_lya[totals_mask & is_qso & ~where_hiL], bins=bins2)
    totals_qso_hiL, _ = np.histogram(
        L_lya[totals_mask & is_qso & where_hiL], bins=bins2)
    totals = totals_sf * sf_factor + totals_qso_loL * \
        good_qso_factor + totals_qso_hiL * hiL_factor
    totals_qso = totals_qso_loL * good_qso_factor + totals_qso_hiL * hiL_factor

    completeness = hg_comp / totals
    comp_sf = (hg_comp_sf * sf_factor) / totals_sf
    comp_qso = (hg_comp_qso_loL * good_qso_factor + hg_comp_qso_hiL * hiL_factor) / \
        (totals_qso_loL * good_qso_factor + totals_qso_hiL * hiL_factor)
    purity = hg_puri / (hg_puri + hb)
    purity[(purity == 0.) | ~np.isfinite(purity)] = 0.

    ax.plot(b_c, comp_sf, color='C1', ls='--', label='Completeness (only SF)')
    ax.plot(b_c, comp_qso, color='C0', ls='--',
            label='Completeness (only QSO)')
    ax.plot(b_c, completeness, marker='s', label='Completeness', c='C5')
    ax.plot(b_c, purity, marker='^', label='Purity', c='C6')

    # Save the arrays
    np.save(f'{dirname}/puri1d_{survey_name}.npy', purity)
    np.save(f'{dirname}/puri_denominator_{survey_name}.npy', (hg_puri + hb))
    np.save(f'{dirname}/comp1d_{survey_name}.npy', completeness)
    np.save(f'{dirname}/comp_sf_{survey_name}.npy', comp_sf)
    np.save(f'{dirname}/comp_qso_{survey_name}.npy', comp_qso)
    np.save(f'{dirname}/comp_denominator_{survey_name}.npy', totals)
    np.save(f'{dirname}/comp_qso_denominator_{survey_name}.npy', totals_qso)
    np.save(f'{dirname}/comp_sf_denominator_{survey_name}.npy', totals_sf)
    np.save(f'{dirname}/puricomp_bins.npy', bins2)

    ax.set_xlabel(r'$\log L$ (erg$\,$s$^{-1}$)')

    ax.set_xlim((42, 45.5))
    ax.set_ylim((0, 1))
    ax.legend(fontsize=12)

    plt.savefig(f'{dirname}/puricomp1d_{survey_name}.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()


def puricomp_corrections(mag_min, mag_max, L_Arr, L_e_Arr, nice_lya, nice_z,
                         mag, zspec_cut, z_cut, mag_cut, ew_cut, L_bins, L_lya,
                         is_gal, is_sf, is_qso, is_LAE, where_hiL, hiL_factor,
                         good_qso_factor, gal_factor):
    r_bins = np.linspace(mag_min, mag_max, 200 + 1)

    r_bins_c = bin_centers(r_bins)
    L_bins_c = bin_centers(L_bins)

    # Perturb L
    N_iter = 250
    h2d_nice_qso_loL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_nice_qso_hiL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_nice_sf_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_normal_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_sf_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_hiL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_loL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_gal_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))

    for k in range(N_iter):
        randN = np.random.randn(len(L_Arr))
        L_perturbed = np.empty_like(L_Arr)
        L_perturbed[randN <= 0] = (L_Arr + L_e_Arr[0] * randN)[randN <= 0]
        L_perturbed[randN > 0] = (L_Arr + L_e_Arr[1] * randN)[randN > 0]
        L_perturbed[np.isnan(L_perturbed)] = 0.

        h2d_nice_sf_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & nice_z & z_cut & is_sf],
            mag[nice_lya & nice_z & z_cut & is_sf],
            bins=[L_bins, r_bins]
        )

        h2d_nice_qso_loL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & nice_z & z_cut & is_qso & ~where_hiL],
            mag[nice_lya & nice_z & z_cut & is_qso & ~where_hiL],
            bins=[L_bins, r_bins]
        )

        h2d_nice_qso_hiL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & nice_z & z_cut & is_qso & where_hiL],
            mag[nice_lya & nice_z & z_cut & is_qso & where_hiL],
            bins=[L_bins, r_bins]
        )

        h2d_sel_sf_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & z_cut & is_sf],
            mag[nice_lya & z_cut & is_sf],
            bins=[L_bins, r_bins]
        )
        h2d_sel_normal_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & z_cut & (is_qso & ~is_LAE)],
            mag[nice_lya & z_cut & (is_qso & ~is_LAE)],
            bins=[L_bins, r_bins]
        )

        h2d_sel_loL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & z_cut &
                        is_qso & is_LAE & ~where_hiL],
            mag[nice_lya & z_cut & is_qso & is_LAE & ~where_hiL],
            bins=[L_bins, r_bins]
        )

        h2d_sel_hiL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & z_cut &
                        is_qso & is_LAE & where_hiL],
            mag[nice_lya & z_cut & is_qso & is_LAE & where_hiL],
            bins=[L_bins, r_bins]
        )

        h2d_sel_gal_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & is_gal & z_cut],
            mag[nice_lya & is_gal & z_cut],
            bins=[L_bins, r_bins]
        )

    # Take the median
    h2d_nice_qso_hiL = np.median(h2d_nice_qso_hiL_i, axis=2)
    h2d_nice_qso_loL = np.median(h2d_nice_qso_loL_i, axis=2)
    h2d_nice_sf = np.median(h2d_nice_sf_i, axis=2)
    h2d_sel_normal = np.median(h2d_sel_normal_i, axis=2)
    h2d_sel_sf = np.median(h2d_sel_sf_i, axis=2)
    h2d_sel_qso_hiL = np.median(h2d_sel_hiL_i, axis=2)
    h2d_sel_qso_loL = np.median(h2d_sel_loL_i, axis=2)
    h2d_sel_gal = np.median(h2d_sel_gal_i, axis=2)

    h2d_parent_sf, _, _ = np.histogram2d(
        L_lya[zspec_cut & mag_cut & ew_cut & is_sf],
        mag[zspec_cut & mag_cut & ew_cut & is_sf],
        bins=[L_bins, r_bins]
    )
    h2d_parent_qso_loL, _, _ = np.histogram2d(
        L_lya[zspec_cut & mag_cut & ew_cut & is_qso & ~where_hiL],
        mag[zspec_cut & mag_cut & ew_cut & is_qso & ~where_hiL],
        bins=[L_bins, r_bins]
    )
    h2d_parent_qso_hiL, _, _ = np.histogram2d(
        L_lya[zspec_cut & mag_cut & ew_cut & is_qso & where_hiL],
        mag[zspec_cut & mag_cut & ew_cut & is_qso & where_hiL],
        bins=[L_bins, r_bins]
    )
    h2d_parent = (
        h2d_parent_sf * sf_factor
        + h2d_parent_qso_loL * good_qso_factor
        + h2d_parent_qso_hiL * hiL_factor
    )
    h2d_nice = (
        h2d_nice_qso_hiL * hiL_factor
        + h2d_nice_qso_loL * good_qso_factor
        + h2d_nice_sf * sf_factor
    )
    h2d_sel = (
        h2d_sel_normal
        + h2d_sel_sf * sf_factor
        + h2d_sel_qso_hiL * hiL_factor
        + h2d_sel_qso_loL * good_qso_factor
        + h2d_sel_gal * gal_factor
    )

    # Make the mats smooooooth
    h2d_nice_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_nice, 0.15, 0.3)
    h2d_sel_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_sel, 0.15, 0.3)
    h2d_parent_smooth = smooth_Image(L_bins_c, r_bins_c, h2d_parent, 0.15, 0.3)

    puri2d = h2d_nice_smooth / h2d_sel_smooth
    comp2d = h2d_nice_smooth / h2d_parent_smooth

    return puri2d, comp2d, L_bins, r_bins


def all_corrections(params, pm_flx, pm_err, zspec, EW_lya, L_lya, is_gal,
                    is_qso, is_sf, is_LAE, where_hiL, survey_name,
                    hiL_factor, good_qso_factor, gal_factor, qso_frac, L_NV,
                    plot_it=True):
    mag_min, mag_max, nb_min, nb_max, ew0_cut, ew_oth, cont_est_m = params

    # Vector of magnitudes in r band
    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mag[np.isnan(mag)] = 99.

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    # Make the directory if it doesn't exist
    folder_name = (
        f'LF_r{mag_min}-{mag_max}_nb{nb_min}-{nb_max}_ew{ew0_cut}_ewoth{ew_oth}'
        f'_{cont_est_m}_{qso_frac}'
    )
    dirname = f'/home/alberto/cosmos/LAEs/Luminosity_functions/{folder_name}'
    os.makedirs(dirname, exist_ok=True)

    # Estimate continuum, search lines
    cont_est_lya, cont_err_lya, lya_lines, other_lines, z_Arr, nice_z =\
        search_lines(pm_flx, pm_err, ew0_cut, ew_oth, zspec, cont_est_m)

    z_cut = (lya_lines >= nb_min) & (lya_lines <= nb_max)
    zspec_cut = (z_min < zspec) & (zspec < z_max)
    mag_cut = (mag > mag_min) & (mag < mag_max)

    N_sources = len(mag_cut)
    snr = np.empty(N_sources)
    for src in range(N_sources):
        l = lya_lines[src]
        snr[src] = pm_flx[l, src] / pm_err[l, src]
    # nice_lya_mask = z_cut_nice & mag_cut & (snr > 6)
    nice_lya_mask = (lya_lines >= nb_min) & (lya_lines <= nb_max) & mag_cut & (snr > 6)

    # Nice lya selection
    nice_lya = nice_lya_select(lya_lines, other_lines, pm_flx, pm_err,
                               cont_est_lya, z_Arr, mask=nice_lya_mask)

    # Estimate Luminosity
    EW_nb_Arr, _, L_Arr, _, _, _ = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    # Compute EW bin err and bias correction
    EW_binning = np.linspace(0, 50, 50)
    Lmask = nice_z & nice_lya
    EW_bin_err, median_EW = compute_EW_bin_err(EW_nb_Arr[Lmask], EW_lya[Lmask], EW_binning)

    # Compute and save L corrections and errors
    L_binning = np.logspace(40, 47, 25 + 1)
    L_bin_c = [L_binning[i: i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]
    Lmask = nice_z & nice_lya & (L_lya > 43)
    L_Lbin_err_plus, L_Lbin_err_minus, median_L =\
        compute_L_Lbin_err(L_Arr[Lmask], L_lya[Lmask], L_binning)
    
    np.save('npy/L_nb_err_plus.npy', L_Lbin_err_plus)
    np.save('npy/L_nb_err_minus.npy', L_Lbin_err_minus)
    np.save('npy/L_bias.npy', median_L)
    np.save('npy/L_nb_err_binning.npy', L_binning)
    np.save('npy/EW_nb_err.npy', EW_bin_err)
    np.save('npy/EW_bias.npy', median_EW)
    np.save('npy/EW_nb_err_binning.npy', EW_binning)

    # Correct L_Arr with the median
    mask_median_L = (median_L < 10)
    L_Arr_corr = L_Arr - np.interp(L_Arr, np.log10(L_bin_c)
                              [mask_median_L], median_L[mask_median_L])
    # For the puricomp2d we include the L_NV
    L_lya_NV = np.log10(10**L_lya + 10**L_NV)

    # Apply bin err
    L_binning_position = binned_statistic(10 ** L_Arr, None,
                                          'count', bins=L_binning).binnumber
    L_binning_position[L_binning_position > len(L_binning) - 2] = len(L_binning) - 2
    L_e_Arr_pm = [L_Lbin_err_minus[L_binning_position],
                  L_Lbin_err_plus[L_binning_position]]

    ew_cut = EW_lya > ew0_cut # ew0_cut

    # Compute puri/comp 2D
    L_bins_cor = np.log10(np.logspace(40, 47, 200 + 1))
    puri2d, comp2d, _, r_bins = puricomp_corrections(
        mag_min, mag_max, L_Arr_corr, L_e_Arr_pm, nice_lya,
        nice_z, mag, zspec_cut, z_cut, mag_cut, ew_cut, L_bins_cor,
        L_lya, is_gal, is_sf, is_qso, is_LAE, where_hiL, hiL_factor,
        good_qso_factor, gal_factor
    )

    np.save(f'{dirname}/puri2d_{survey_name}.npy', puri2d)
    np.save(f'{dirname}/comp2d_{survey_name}.npy', comp2d)
    np.save(f'{dirname}/puricomp2d_L_bins.npy', L_bins_cor)
    np.save(f'{dirname}/puricomp2d_r_bins.npy', r_bins)

    if not plot_it:
        return

    nbs_to_consider = np.arange(nb_min, nb_max + 1)

    purity_or_completeness_plot(mag, nbs_to_consider, lya_lines, nice_lya,
                                nice_z, L_Arr_corr, mag_max, mag_min, ew0_cut, is_gal,
                                is_sf, is_qso, is_LAE, zspec, L_lya, dirname,
                                ew_cut, where_hiL, survey_name)


def make_corrections(params, qso_frac):

    survey_name_list = ['minijpasAEGIS001', 'minijpasAEGIS002', 'minijpasAEGIS003',
                        'minijpasAEGIS004', 'jnep']
    
    mag_min, mag_max = params[:2]
    pm_flx_0, _, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal, is_LAE, where_hiL, L_NV =\
        load_mocks(add_errs=False, mag_min=mag_min, mag_max=mag_max)
    print(f'Mock len = {len(zspec)}')

    for survey_name in survey_name_list:
        print(f'{survey_name}')

        # # Comment this section if you don't want to recompute corrections
        # try:
        #     np.load(f'npy/puri2d_{survey_name}.npy')
        #     np.load(f'npy/comp2d_{survey_name}.npy')
        # except:
        #     print('Making puricomp...')
        # else:
        #     print('Loaded.')
        #     continue
        # ######

        pm_flx, pm_err = add_errors(pm_flx_0, apply_err=True,
                                    survey_name=survey_name)

        where_bad_flx = ~np.isfinite(pm_flx)
        pm_flx[where_bad_flx] = 0.
        pm_err[where_bad_flx] = 9999999999.

        all_corrections(
            params, pm_flx, pm_err, zspec, EW_lya, L_lya, is_gal,
            is_qso, is_sf, is_LAE, where_hiL, survey_name,
            hiL_factor, good_qso_factor, gal_factor, qso_frac, L_NV
        )


def effective_volume(nb_min, nb_max, survey_name='both'):
    '''
    Due to NB overlap, specially when considering single filters, the volume probed by one
    NB has to be corrected because some sources could be detected in that NB or in either
    of the adjacent ones.

    ## Tile_IDs ##
    AEGIS001: 2241
    AEGIS002: 2243
    AEGIS003: 2406
    AEGIS004: 2470
    '''

    if survey_name == 'jnep':
        area = 0.24
    elif survey_name == 'minijpas':
        area = 0.895
    elif survey_name == 'both':
        area = 0.24 + 0.895
    else:
        # If the survey name is not known, try to use the given value as area
        try:
            area = float(survey_name)
        except:
            raise ValueError('Survey name not known or invalid area value')

    z_min_overlap = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max_overlap = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    z_min_abs = (w_central[nb_min - 1] +
                 nb_fwhm_Arr[nb_min - 1] * 0.5) / w_lya - 1
    z_max_abs = (w_central[nb_max + 1] -
                 nb_fwhm_Arr[nb_min + 1] * 0.5) / w_lya - 1

    # volume_abs is a single scalar value in case of 'jnep' and an array of
    # 4 values for each pointing in case of 'minijpas
    volume_abs = z_volume(z_min_abs, z_max_abs, area)
    volume_overlap = (
        z_volume(z_min_overlap, z_min_abs, area)
        + z_volume(z_max_abs, z_max_overlap, area)
    )

    return volume_abs + volume_overlap * 0.5


def make_the_LF(params, qso_frac, cat_list=['minijpas', 'jnep'], return_hist=False):
    mag_min, mag_max, nb_min, nb_max, ew0_cut, ew_oth, cont_est_m = params

    pm_flx, pm_err, tile_id, pmra_sn, pmdec_sn, parallax_sn, starprob, _,\
        spCl, zsp, _, _, _, N_minijpas, x_im, y_im,\
        ra, dec =\
        load_minijpas_jnep(cat_list)
    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mask = mask_proper_motion(parallax_sn, pmra_sn, pmdec_sn)

    cont_est_lya, cont_err_lya, cont_est_other, cont_err_other =\
        nb_or_3fm_cont(pm_flx, pm_err, cont_est_m)

    # Lya search
    line = is_there_line(pm_flx, pm_err, cont_est_lya,
                         cont_err_lya, ew0_cut, mask=mask)
    lya_lines, lya_cont_lines, _ = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
                               ew_oth, obs=True, mask=mask, sigma=5)
    other_lines = identify_lines(line_other, pm_flx, cont_est_other)

    N_sources = pm_flx.shape[1]

    mag_cut = (mag > mag_min) & (mag < mag_max)

    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    snr = np.empty(N_sources)
    for src in range(N_sources):
        l = lya_lines[src]
        snr[src] = pm_flx[l, src] / pm_err[l, src]

    mask = (lya_lines >= nb_min) & (lya_lines <= nb_max) & mag_cut & (snr > 6)
    nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont_est_lya, z_Arr, mask=mask
    )

    # Estimate Luminosity
    EW_Arr, _, L_Arr, _, _, _ = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    L_Lbin_err_plus = np.load('npy/L_nb_err_plus.npy')
    L_Lbin_err_minus = np.load('npy/L_nb_err_minus.npy')
    median_L = np.load('npy/L_bias.npy')
    L_binning = np.load('npy/L_nb_err_binning.npy')
    EW_bin_err = np.load('npy/EW_nb_err.npy')
    median_EW = np.load('npy/EW_bias.npy')
    EW_binning = np.load('npy/EW_nb_err_binning.npy')
    L_bin_c = [L_binning[i: i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]
    EW_bin_c = [EW_binning[i: i + 2].sum() * 0.5 for i in range(len(EW_binning) - 1)]

    EW_binning_position = binned_statistic(EW_Arr, None, 'count',
                                           bins=EW_binning).binnumber
    EW_binning_position[EW_binning_position > len(EW_binning) - 2] = len(EW_binning) - 2
    EW_Arr_err_corr = EW_bin_err[EW_binning_position]
    EW_Arr_corr = EW_Arr - np.interp(EW_Arr, np.log10(EW_bin_c), median_EW)

    # Apply bin err
    L_binning_position = binned_statistic(10 ** L_Arr, None,
                                          'count', bins=L_binning).binnumber
    L_binning_position[L_binning_position > len(L_binning) - 2] = len(L_binning) - 2
    L_e_Arr = (L_Lbin_err_plus + L_Lbin_err_minus)[L_binning_position] * 0.5

    # Correct L_Arr with the median
    mask_median_L = (median_L < 10)
    corr_L = np.interp(L_Arr, np.log10(L_bin_c)[mask_median_L], median_L[mask_median_L])
    L_Arr_corr = L_Arr - corr_L

    bins = np.log10(L_binning)

    N_sources = pm_flx.shape[1]
    is_minijpas_source = np.ones(N_sources).astype(bool)
    is_minijpas_source[N_minijpas:] = False

    print(f'nice miniJPAS = {count_true(nice_lya & is_minijpas_source)}')
    print(f'nice J-NEP = {count_true(nice_lya & ~is_minijpas_source)}')

    volume = effective_volume(nb_min, nb_max, 'both')

    b = bins

    LF_bins = np.array([(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)])

    bin_width = np.array([b[i + 1] - b[i] for i in range(len(b) - 1)])

    L_LF_err_plus_mj = np.zeros(len(bins) - 1)
    L_LF_err_minus_mj = np.zeros(len(bins) - 1)
    hist_median_mj = np.zeros(len(bins) - 1)

    nice_puri_list = np.zeros(count_true(nice_lya))

    folder_name = (
        f'LF_r{mag_min}-{mag_max}_nb{nb_min}-{nb_max}_ew{ew0_cut}_ewoth{ew_oth}'
        f'_{cont_est_m}_{qso_frac}'
    )
    dirname = f'/home/alberto/cosmos/LAEs/Luminosity_functions/{folder_name}'
    os.makedirs(dirname, exist_ok=True)

    tile_id_list = [2241, 2243, 2406, 2470]
    for i, this_id in enumerate(tile_id_list):
        this_mask = (tile_id == this_id)
        L_e_Arr_pm = [L_Lbin_err_minus[L_binning_position][this_mask],
                    L_Lbin_err_plus[L_binning_position][this_mask]]
        L_LF_err_percentiles, this_puri = LF_perturb_err(
            corr_L[this_mask] * 0, L_Arr_corr[this_mask], L_e_Arr_pm,
            nice_lya[this_mask], mag[this_mask], z_Arr[this_mask], starprob[this_mask],
            bins, f'minijpasAEGIS00{i + 1}', tile_id[this_mask],
            return_puri=True, dirname=dirname
        )
        L_LF_err_plus_mj += L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
        L_LF_err_minus_mj += L_LF_err_percentiles[1] - L_LF_err_percentiles[0]
        hist_median_mj += L_LF_err_percentiles[1]

        nice_puri_list[this_mask[nice_lya]] = this_puri

    L_e_Arr_pm = [L_Lbin_err_minus[L_binning_position][~is_minijpas_source],
                L_Lbin_err_plus[L_binning_position][~is_minijpas_source]]
    L_LF_err_percentiles, this_puri = LF_perturb_err(
        corr_L[~is_minijpas_source] * 0, L_Arr_corr[~is_minijpas_source],
        L_e_Arr_pm, nice_lya[~is_minijpas_source],
        mag[~is_minijpas_source], z_Arr[~is_minijpas_source],
        starprob[~is_minijpas_source], bins, 'jnep', tile_id[~is_minijpas_source],
        return_puri=True, dirname=dirname
    )
    L_LF_err_plus_jn = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    L_LF_err_minus_jn = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]
    hist_median_jn = L_LF_err_percentiles[1]

    nice_puri_list[~is_minijpas_source[nice_lya]] = this_puri

    hist_median = hist_median_jn + hist_median_mj
    L_LF_err_plus = L_LF_err_plus_jn + L_LF_err_plus_mj
    L_LF_err_minus = L_LF_err_minus_jn + L_LF_err_minus_mj

    ###### RAW LF ######
    LF_raw = np.histogram(L_Arr_corr[nice_lya], bins=bins)[0] / bin_width / volume
    ####################

    # Save the selection
    selection = {
        'src': np.where(nice_lya)[0],
        'tile_id': tile_id[nice_lya],
        'x_im': x_im[nice_lya],
        'y_im': y_im[nice_lya],
        'nb_sel': lya_lines[nice_lya],
        'SDSS_spCl': spCl[nice_lya],
        'SDSS_zspec': zsp[nice_lya],
        'RA': ra[nice_lya],
        'DEC': dec[nice_lya],
        'L_lya': L_Arr_corr[nice_lya],
        'L_lya_NV': L_Arr[nice_lya],
        'L_lya_err': L_e_Arr[nice_lya],
        'EW_lya': EW_Arr_corr[nice_lya],
        'EW_lya_err': EW_Arr_err_corr[nice_lya],
        'puri': nice_puri_list,
        'r': mag[nice_lya],
        'other_lines': [other_lines[idx] for idx in np.where(nice_lya)[0]]
    }

    with open(f'{dirname}/selection.npy', 'wb') as f:
        pickle.dump(selection, f)

    # Initialize dict to save the LFs
    LFs_dict = {'LF_bins': LF_bins}

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the corrected total LF
    yerr_cor_plus = (hist_median + L_LF_err_plus **
                     2) ** 0.5 / bin_width / volume
    yerr_cor_minus = (hist_median + L_LF_err_minus **
                      2) ** 0.5 / bin_width / volume
    xerr = bin_width / 2
    LF_values = hist_median / bin_width / volume
    ax.errorbar(LF_bins, LF_values,
                yerr=[yerr_cor_minus, yerr_cor_plus], xerr=xerr,
                marker='s', linestyle='', color='r', capsize=4,
                label='miniJPAS + J-NEP (corrected)', zorder=99)
    LFs_dict['LF_total'] = LF_values
    LFs_dict['LF_total_err'] = [yerr_cor_minus, yerr_cor_plus, xerr]

    # Plot the total raw LF
    ax.plot(LF_bins, LF_raw, ls='', markerfacecolor='none', markeredgecolor='dimgray',
            marker='^', markersize=11, zorder=4, label='miniJPAS + J-NEP (raw)')
    LFs_dict['LF_total_raw'] = LF_raw

    # Save the dict
    dirname = f'/home/alberto/cosmos/LAEs/Luminosity_functions/{folder_name}'
    os.makedirs(dirname, exist_ok=True)

    dict_filename = f'{dirname}/LFs.pkl'
    with open(dict_filename, 'wb') as file:
        pickle.dump(LFs_dict, file)

    # Plot the reference LF curves
    Lx = np.linspace(10 ** 42, 10 ** 46, 10000)
    phistar1 = 3.33e-6
    Lstar1 = 44.65
    alpha1 = -1.35

    phistar2 = -3.45
    Lstar2 = 42.93
    alpha2 = -1.93

    Phi_center = double_schechter(
        Lx, phistar1, 10 ** Lstar1, alpha1, 10 ** phistar2, 10 ** Lstar2, alpha2
    ) * Lx * np.log(10)

    ax.plot(
        np.log10(Lx), Phi_center, ls='-.', alpha=0.7,
        label='Spinoso2020 (2.2 < z < 3.25)', zorder=1,
        color='C6'
    )

    phistar1 = 10 ** -3.41
    Lstar1 = 10 ** 42.87
    alpha1 = -1.7

    phistar2 = 10 ** -5.85
    Lstar2 = 10 ** 44.6
    alpha2 = -1.2

    Phi_center = double_schechter(
        Lx, phistar1, Lstar1, alpha1, phistar2, Lstar2, alpha2
    ) * Lx * np.log(10)

    ax.plot(
        np.log10(Lx), Phi_center, ls='-.', alpha=0.7,
        label='Zhang2021 (2 < z < 3.2)', zorder=0,
        color='C7'
    )

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log L_{\mathrm{Ly}\alpha}$ (erg$\,$s$^{-1}$)')
    ax.set_ylabel(r'$\Phi$ (Mpc$^{-3}\,\Delta\logL^{-1}$)')
    ax.set_ylim(1e-8, 5e-3)
    ax.set_xlim(42.5, 45.5)
    ax.legend(fontsize=9)

    plt.savefig(f'{dirname}/LumFunc.pdf', bbox_inches='tight',
                facecolor='white')
    plt.close()

    if return_hist:
        return hist_median, bins


if __name__ == '__main__':
    # Parameters of the LF:
    # (min_mag, max_mag, nb_min, nb_max, ew0_cut, cont_est_method)
    # cont_est_method must be 'nb' or '3fm'
    LF_parameters = [
        (17, 24, 1, 4, 30, 100, 'nb'),
        (17, 24, 4, 8, 30, 100, 'nb'),
        (17, 24, 8, 12, 30, 100, 'nb'),
        (17, 24, 12, 16, 30, 100, 'nb'),
        (17, 24, 16, 20, 30, 100, 'nb'),
        (17, 24, 20, 24, 30, 100, 'nb'),

        # (17, 24, 1, 24, 30, 100, 'nb'),

        # (17, 24, 8, 12, 30, 100, 'nb'),
        # (17, 24, 8, 12, 20, 100, 'nb'),
        # (17, 24, 8, 12, 50, 100, 'nb'),
        # (17, 24, 8, 12, 30, 200, 'nb'),
        # (17, 24, 8, 12, 30, 50, 'nb'),
    ]
    
    for qso_frac in [1.0, 0.3, 0.5, 0.7]:
        print(f'QSO_frac = {qso_frac}\n')
        for params in LF_parameters:
            gal_area = 3
            bad_qso_area = 200
            good_qso_area = 400 / qso_frac
            hiL_qso_area = 4000 / qso_frac
            sf_area = 400 * sf_frac

            # the proportional factors are made in relation to bad_qso
            # so bad_qso_factor = 1
            gal_factor = bad_qso_area / gal_area
            good_qso_factor = bad_qso_area / good_qso_area
            hiL_factor = bad_qso_area / hiL_qso_area
            sf_factor = bad_qso_area / sf_area

            t00 = time.time()
            print(
                'mag{0}-{1}, nb{2}-{3}, ew0_lya={4}, ew_oth={5}, cont_est_method={6}'
                .format(*params))
            make_corrections(params, qso_frac)
            print('\nBuilding the LF...')
            make_the_LF(params, qso_frac)

            print('\n\n')
            h, m, s = hms_since_t0(t00)
            print(f'Total elapsed: {h}h {m}m {s}s')
            print('\n ########################## \n')
