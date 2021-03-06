#!/home/alberto/miniconda3/bin/python3

import numpy as np
import pickle
from scipy.stats import binned_statistic

import os

from three_filter import cont_est_3FM
from LumFunc_miniJPAS import LF_perturb_err
from load_jpas_catalogs import load_minijpas_jnep
from load_mocks import ensemble_mock
from my_functions import *

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 13})


np.seterr(all='ignore')


# Useful definitions
w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67
filter_tags = load_filter_tags()

gal_area = 5.54
bad_qso_area = 200
good_qso_area = 400
hiL_qso_area = 4000
# good_qso_area = 200
# hiL_qso_area = 2000

# the proportional factors are made in relation to bad_qso
# so bad_qso_factor = 1
gal_factor = bad_qso_area / gal_area
good_qso_factor = bad_qso_area / good_qso_area
hiL_factor = bad_qso_area / hiL_qso_area

z_nb_Arr = w_central[:-4] / w_lya - 1


def load_mocks(train_or_test, survey_name, add_errs=True, qso_LAE_frac=1.):
    name_qso = 'QSO_100000_0'
    name_qso_bad = f'QSO_double_{train_or_test}_{survey_name}_DR16_D_0'
    name_qso_hiL = f'QSO_double_{train_or_test}_{survey_name}_DR16_highL2_D_0'
    name_gal = f'GAL_LC_{survey_name}_0'
    name_sf = f'LAE_12.5deg_z2-4.25_{train_or_test}_{survey_name}_VUDS_0'

    pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal,\
        is_LAE, where_hiL, _ = ensemble_mock(name_qso, name_gal, name_sf,
                                             name_qso_bad, name_qso_hiL, add_errs,
                                             qso_LAE_frac)

    return pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal, is_LAE, where_hiL


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


def search_lines(pm_flx, pm_err, ew0_cut, zspec, cont_est_m):
    cont_est_lya, cont_err_lya, cont_est_other, cont_err_other =\
        nb_or_3fm_cont(pm_flx, pm_err, cont_est_m)

    # Lya search
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut)
    lya_lines, lya_cont_lines, _ = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
                               400, obs=True)
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
    L_Lbin_err_plus = np.ones(len(L_binning) - 1) * 99
    L_Lbin_err_minus = np.ones(len(L_binning) - 1) * 99
    median = np.ones(len(L_binning) - 1) * 99
    last = [99., 99.]
    for i in range(len(L_binning) - 1):
        in_bin = (10 ** L_Arr >= L_binning[i]
                  ) & (10 ** L_Arr < L_binning[i + 1])
        if count_true(in_bin) == 0:
            L_Lbin_err_plus[i] = last[0]
            L_Lbin_err_minus[i] = last[1]
            continue
        perc = np.nanpercentile((L_Arr - L_lya)[in_bin], [16, 50, 84])
        L_Lbin_err_plus[i] = perc[2] - perc[1]

        last = [L_Lbin_err_plus[i], L_Lbin_err_minus[i]]
        median[i] = perc[1]

    return L_Lbin_err_plus, median


def purity_or_completeness_plot(mag, nbs_to_consider, lya_lines,
                                nice_lya, nice_z, L_Arr, mag_max,
                                mag_min, ew0_cut, is_gal, is_sf, is_qso, is_LAE,
                                zspec, L_lya, dirname, ew_cut, where_hiL, survey_name):
    fig, ax = plt.subplots(figsize=(7, 4))

    bins2 = np.linspace(42, 45.5, 15)

    b_c = [0.5 * (bins2[i] + bins2[i + 1]) for i in range(len(bins2) - 1)]

    this_mag_cut = (mag < mag_max) & (mag > mag_min)

    # for nb in nbs_to_consider:
    #     nb_mask = (lya_lines == nb)

    #     z_min = (w_central[nb] - nb_fwhm_Arr[nb] * 0.5) / w_lya - 1
    #     z_max = (w_central[nb] + nb_fwhm_Arr[nb] * 0.5) / w_lya - 1

    #     this_zspec_cut = (z_min < zspec) & (zspec < z_max)
    #     totals_mask = this_zspec_cut & this_mag_cut & ew_cut

    #     goodh_puri_sf = L_Arr[nice_lya & nice_z &
    #                           is_sf & ew_cut & this_mag_cut & nb_mask]
    #     goodh_puri_qso_hiL = L_Arr[nice_lya & nice_z &
    #                                is_qso & ew_cut & this_mag_cut & nb_mask & where_hiL]
    #     goodh_puri_qso_loL = L_Arr[nice_lya & nice_z & is_qso &
    #                                ew_cut & this_mag_cut & nb_mask & ~where_hiL]
    #     goodh_comp_sf = L_lya[nice_lya & nice_z & is_sf & totals_mask]
    #     goodh_comp_qso_hiL = L_lya[nice_lya &
    #                                nice_z & is_qso & totals_mask & where_hiL]
    #     goodh_comp_qso_loL = L_lya[nice_lya &
    #                                nice_z & is_qso & totals_mask & ~where_hiL]
    #     badh_qso_hiL = L_Arr[nice_lya & ~nice_z & is_qso &
    #                          is_LAE & nb_mask & this_mag_cut & where_hiL]
    #     badh_qso_loL = L_Arr[nice_lya & ~nice_z & is_qso &
    #                          is_LAE & nb_mask & this_mag_cut & ~where_hiL]
    #     badh_normal = L_Arr[nice_lya & ~nice_z & (
    #         is_sf | (is_qso & ~is_LAE)) & nb_mask & this_mag_cut]
    #     badh_gal = L_Arr[nice_lya & ~nice_z & is_gal & nb_mask & this_mag_cut]

    #     hg_puri_sf, _ = np.histogram(goodh_puri_sf, bins=bins2)
    #     hg_puri_qso_hiL, _ = np.histogram(goodh_puri_qso_hiL, bins=bins2)
    #     hg_puri_qso_loL, _ = np.histogram(goodh_puri_qso_loL, bins=bins2)
    #     hg_comp_sf, _ = np.histogram(goodh_comp_sf, bins=bins2)
    #     hg_comp_qso_hiL, _ = np.histogram(goodh_comp_qso_hiL, bins=bins2)
    #     hg_comp_qso_loL, _ = np.histogram(goodh_comp_qso_loL, bins=bins2)
    #     hb_qso_hiL, _ = np.histogram(badh_qso_hiL, bins=bins2)
    #     hb_qso_loL, _ = np.histogram(badh_qso_loL, bins=bins2)
    #     hb_normal, _ = np.histogram(badh_normal, bins=bins2)
    #     hb_gal, _ = np.histogram(badh_gal, bins=bins2)

    #     hg_puri = (
    #         hg_puri_sf
    #         + hg_puri_qso_loL * good_qso_factor
    #         + hg_puri_qso_hiL * hiL_factor
    #     )
    #     hg_comp = (
    #         hg_comp_sf
    #         + hg_comp_qso_loL * good_qso_factor
    #         + hg_comp_qso_hiL * hiL_factor
    #     )
    #     hb = (
    #         hb_normal
    #         + hb_qso_loL * good_qso_factor
    #         + hb_qso_hiL * hiL_factor
    #         + hb_gal * gal_factor
    #     )
    #     totals_sf, _ = np.histogram(L_lya[totals_mask & is_sf], bins=bins2)
    #     totals_qso_loL, _ = np.histogram(
    #         L_lya[totals_mask & is_qso & ~where_hiL], bins=bins2)
    #     totals_qso_hiL, _ = np.histogram(
    #         L_lya[totals_mask & is_qso & where_hiL], bins=bins2)
    #     totals = (
    #         totals_sf
    #         + totals_qso_loL * good_qso_factor
    #         + totals_qso_hiL * hiL_factor
    #     )

    # if which_one == 'Completeness':
    #     ax.plot(
    #         b_c, hg_comp / totals, marker='s',
    #         label=filter_tags[nb], zorder=99, alpha=0.5
    #     )
    # if which_one == 'Purity':
    #     ax.plot(
    #         b_c, hg_puri / (hg_puri + hb), marker='s',
    #         label=filter_tags[nb], zorder=99, alpha=0.5
    #     )

    nb_min = nbs_to_consider[0]
    nb_max = nbs_to_consider[-1]
    nb_mask = (lya_lines >= nb_min) & (lya_lines <= nb_max)
    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1
    this_zspec_cut = (z_min < zspec) & (zspec < z_max)

    totals_mask = this_zspec_cut & this_mag_cut & ew_cut

    goodh_puri_sf = L_Arr[nice_lya & nice_z &
                          is_sf & ew_cut & this_mag_cut & nb_mask]
    goodh_puri_qso = L_Arr[nice_lya & nice_z &
                           is_qso & ew_cut & this_mag_cut & nb_mask]
    goodh_comp_sf = L_lya[nice_lya & nice_z & is_sf & totals_mask]
    goodh_comp_qso = L_lya[nice_lya & nice_z & is_qso & totals_mask]
    badh_to_corr = L_Arr[nice_lya & ~nice_z & (
        is_qso & is_LAE) & nb_mask & this_mag_cut]
    badh_normal = L_Arr[nice_lya & ~nice_z & (
        is_sf | (is_qso & ~is_LAE)) & nb_mask & this_mag_cut]
    badh_gal = L_Arr[nice_lya & ~nice_z & is_gal & nb_mask & this_mag_cut]

    hg_puri_sf, _ = np.histogram(goodh_puri_sf, bins=bins2)
    hg_puri_qso, _ = np.histogram(goodh_puri_qso, bins=bins2)
    hg_comp_sf, _ = np.histogram(goodh_comp_sf, bins=bins2)
    hg_comp_qso, _ = np.histogram(goodh_comp_qso, bins=bins2)
    hb_to_corr, _ = np.histogram(badh_to_corr, bins=bins2)
    hb_normal, _ = np.histogram(badh_normal, bins=bins2)
    hb_gal, _ = np.histogram(badh_gal, bins=bins2)

    hg_puri = hg_puri_sf + hg_puri_qso * good_qso_factor
    hg_comp = hg_comp_sf + hg_comp_qso * good_qso_factor
    hb = hb_normal + hb_to_corr * good_qso_factor + hb_gal * gal_factor
    totals_sf, _ = np.histogram(L_lya[totals_mask & is_sf], bins=bins2)
    totals_qso, _ = np.histogram(L_lya[totals_mask & is_qso], bins=bins2)
    totals = totals_sf + totals_qso * good_qso_factor

    completeness = hg_comp / totals
    purity = hg_puri / (hg_puri + hb)
    F1score = 2 * purity * completeness / (purity + completeness)

    ax.plot(b_c, completeness, marker='s', label='Completeness', c='C5')
    ax.plot(b_c, purity, marker='^', label='Purity', c='C6')
    # ax.plot(b_c, F1score, marker='^', label='F1 score', zorder=-99, c='dimgray')

    ax.set_xlabel(r'$\log L$ (erg$\,$s$^{-1}$)')

    ax.set_xlim((42, 45.5))
    ax.set_ylim((0, 1))
    ax.legend(fontsize=10)
    ax.set_title(
        f'r{mag_min}-{mag_max}, EW0_cut = {ew0_cut}, z{z_min:0.2f}-{z_max:0.2f}',
        fontsize=12)

    plt.savefig(f'{dirname}/puricomp1d_{survey_name}.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()


def plot_puricomp_grids(puri, comp, L_bins, r_bins, dirname, survey_name):
    fig = plt.figure(figsize=(7, 6))

    width = 1
    height = 1
    spacing = 0.1
    cbar_width = 0.08

    # ADD AXES
    ax0 = fig.add_axes([0, 0, width, height])
    ax1 = fig.add_axes([width + spacing, 0, width, height])
    axc = fig.add_axes([width * 2 + spacing * 2, 0, cbar_width, height])

    # Mask puri and comp where at least one of them is zero or nan
    mask_puricomp = ~(np.isfinite(puri) & np.isfinite(comp)
                     & (puri > 0) & (comp > 0))
    puri[mask_puricomp] = np.nan
    comp[mask_puricomp] = np.nan

    # PLOT STUFF
    cmap = 'Spectral'
    sns.heatmap(puri.T, ax=ax0, vmin=0, vmax=1, cbar_ax=axc, cmap=cmap)
    sns.heatmap(comp.T, ax=ax1, vmin=0, vmax=1, cbar=False, cmap=cmap)

    # TICKS
    xticks = range(len(L_bins))
    yticks = range(len(r_bins))
    xtick_labels = ['{0:0.1f}'.format(n) for n in L_bins]
    ytick_labels = ['{0:0.1f}'.format(n) for n in r_bins]

    ax0.set_yticks(yticks)
    ax0.set_yticklabels(ytick_labels, rotation='horizontal')
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xtick_labels, rotation='vertical')
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(axis='y', direction='in', labelsize=14)
    ax0.tick_params(axis='x', direction='in', labelsize=14)

    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels, rotation='horizontal')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels, rotation='vertical')
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(axis='y', direction='in', labelsize=14)
    ax1.tick_params(axis='x', direction='in', labelsize=14)

    # SPINES
    ax0.spines[:].set_visible(True)
    ax1.spines[:].set_visible(True)

    # TITLES
    ax0.set_title('Purity', fontsize=25)
    ax1.set_title('Completeness', fontsize=25)

    # AXES LABELS
    ax0.set_xlabel(r'$\logL_{\mathrm{Ly}\alpha}$ (erg s$^{-1}$)')
    ax1.set_xlabel(r'$\logL_{\mathrm{Ly}\alpha}$')
    ax0.set_ylabel('$r$ (magAB)')

    plt.savefig(f'{dirname}/PuriComp2D_{survey_name}.pdf',
                bbox_inches='tight', facecolor='white',)
    plt.close()


def puricomp_corrections(mag_min, mag_max, L_Arr, L_e_Arr, nice_lya, nice_z,
                         mag, zspec_cut, z_cut, mag_cut, ew_cut, L_bins, L_lya,
                         is_gal, is_sf, is_qso, is_LAE, where_hiL, hiL_factor,
                         good_qso_factor, gal_factor):
    r_bins = np.linspace(mag_min, mag_max, 10 + 1)

    # Perturb L
    N_iter = 1000
    h2d_nice_qso_loL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_nice_qso_hiL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_nice_sf_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_normal_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_hiL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_loL_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_gal_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))

    for k in range(N_iter):
        L_perturbed = L_Arr + L_e_Arr * np.random.randn(len(L_e_Arr))
        L_perturbed[np.isnan(L_perturbed)] = 0.

        h2d_nice_sf_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & nice_z & zspec_cut & is_sf],
            mag[nice_lya & nice_z & zspec_cut & is_sf],
            bins=[L_bins, r_bins]
        )

        h2d_nice_qso_loL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & nice_z & zspec_cut & is_qso & ~where_hiL],
            mag[nice_lya & nice_z & zspec_cut & is_qso & ~where_hiL],
            bins=[L_bins, r_bins]
        )

        h2d_nice_qso_hiL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & nice_z & zspec_cut & is_qso & where_hiL],
            mag[nice_lya & nice_z & zspec_cut & is_qso & where_hiL],
            bins=[L_bins, r_bins]
        )

        h2d_sel_normal_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & ~is_gal & z_cut &
                        (is_sf | (is_qso & ~is_LAE))],
            mag[nice_lya & ~is_gal & z_cut & (is_sf | (is_qso & ~is_LAE))],
            bins=[L_bins, r_bins]
        )

        h2d_sel_loL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & ~is_gal & z_cut &
                        is_qso & is_LAE & ~where_hiL],
            mag[nice_lya & ~is_gal & z_cut & is_qso & is_LAE & ~where_hiL],
            bins=[L_bins, r_bins]
        )

        h2d_sel_hiL_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & ~is_gal & z_cut &
                        is_qso & is_LAE & where_hiL],
            mag[nice_lya & ~is_gal & z_cut & is_qso & is_LAE & where_hiL],
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
        h2d_parent_sf
        + h2d_parent_qso_loL * good_qso_factor
        + h2d_parent_qso_hiL * hiL_factor
    )
    h2d_nice = (
        h2d_nice_qso_hiL * hiL_factor
        + h2d_nice_qso_loL * good_qso_factor
        + h2d_nice_sf
    )
    h2d_sel = (
        h2d_sel_normal
        + h2d_sel_qso_hiL * hiL_factor
        + h2d_sel_qso_loL * good_qso_factor
        + h2d_sel_gal * gal_factor
    )

    puri2d = h2d_nice / h2d_sel
    comp2d = h2d_nice / h2d_parent

    return puri2d, comp2d, L_bins, r_bins


def all_corrections(params, pm_flx, pm_err, zspec, EW_lya, L_lya, is_gal,
                    is_qso, is_sf, is_LAE, where_hiL, survey_name,
                    hiL_factor, good_qso_factor, gal_factor, plot_it=True):
    mag_min, mag_max, nb_min, nb_max, ew0_cut, ew_oth, cont_est_m = params

    # Vector of magnitudes in r band
    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mag[np.isnan(mag)] = 99.

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    # Make the directory if it doesn't exist
    folder_name = (
        f'LF_r{mag_min}-{mag_max}_z{z_min:0.1f}-{z_max:0.1f}_ew{ew0_cut}_ewoth{ew_oth}'
        f'_{cont_est_m}'
    )
    dirname = f'/home/alberto/cosmos/LAEs/Luminosity_functions/{folder_name}'
    os.makedirs(dirname, exist_ok=True)

    # Estimate continuum, search lines
    cont_est_lya, cont_err_lya, lya_lines, other_lines, z_Arr, nice_z =\
        search_lines(pm_flx, pm_err, ew0_cut, zspec, cont_est_m)

    z_cut_nice = (z_min - 0.2 < z_Arr) & (z_Arr < z_max + 0.2)
    z_cut = (z_min < z_Arr) & (z_Arr < z_max)
    zspec_cut = (z_min < zspec) & (zspec < z_max)
    ew_cut = EW_lya > ew0_cut
    mag_cut = (mag > mag_min) & (mag < mag_max)

    # Nice lya selection
    nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont_est_lya, z_Arr
    )
    nice_lya = (nice_lya & z_cut_nice & mag_cut)

    # Estimate Luminosity
    _, _, L_Arr, _, _, _ = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    # Compute and save L corrections and errors
    L_binning = np.logspace(40, 47, 25 + 1)
    L_bin_c = [L_binning[i: i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]
    L_Lbin_err, median_L = compute_L_Lbin_err(
        L_Arr[nice_lya & nice_z], L_lya[nice_z & nice_lya], L_binning
    )
    np.save('npy/L_nb_err.npy', L_Lbin_err)
    np.save('npy/L_bias.npy', median_L)
    np.save('npy/L_nb_err_binning.npy', L_binning)

    # Correct L_Arr with the median
    mask_median_L = (median_L < 10)
    L_Arr = L_Arr - np.interp(L_Arr, np.log10(L_bin_c)
                              [mask_median_L], median_L[mask_median_L])

    # Apply bin err
    L_binning_position = binned_statistic(
        10 ** L_Arr, None, 'count', bins=L_binning
    ).binnumber
    L_binning_position[L_binning_position > len(
        L_binning) - 2] = len(L_binning) - 2
    L_e_Arr = L_Lbin_err[L_binning_position]

    bins = np.log10(L_binning)

    # Compute puri/comp 2D
    puri2d, comp2d, L_bins, r_bins = puricomp_corrections(
        mag_min, mag_max, L_Arr, L_e_Arr, nice_lya,
        nice_z, mag, zspec_cut, z_cut, mag_cut, ew_cut, bins,
        L_lya, is_gal, is_sf, is_qso, is_LAE, where_hiL, hiL_factor,
        good_qso_factor, gal_factor
    )

    np.save(f'npy/puri2d_{survey_name}.npy', puri2d)
    np.save(f'npy/comp2d_{survey_name}.npy', comp2d)
    np.save('npy/puricomp2d_L_bins.npy', L_bins)
    np.save('npy/puricomp2d_r_bins.npy', r_bins)

    if not plot_it:
        return

    plot_puricomp_grids(puri2d, comp2d, L_bins, r_bins, dirname, survey_name)

    nbs_to_consider = np.arange(nb_min, nb_max + 1)

    purity_or_completeness_plot(mag, nbs_to_consider, lya_lines, nice_lya,
                                nice_z, L_Arr, mag_max, mag_min, ew0_cut, is_gal,
                                is_sf, is_qso, is_LAE, zspec, L_lya, dirname,
                                ew_cut, where_hiL, survey_name)


def make_corrections(params):
    for survey_name in ['minijpas', 'jnep']:
        pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal, is_LAE, where_hiL =\
            load_mocks('train', survey_name)
        all_corrections(
            params, pm_flx, pm_err, zspec, EW_lya, L_lya, is_gal,
            is_qso, is_sf, is_LAE, where_hiL, survey_name,
            hiL_factor, good_qso_factor, gal_factor
        )


def effective_volume(nb_min, nb_max, survey_name):
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
            raise ValueError('Survey name not known')

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


def make_the_LF(params, cat_list=['minijpas', 'jnep'], return_hist=False):
    mag_min, mag_max, nb_min, nb_max, ew0_cut, ew_oth, cont_est_m = params

    pm_flx, pm_err, tile_id, pmra_sn, pmdec_sn, parallax_sn, starprob, _, _,\
        _, _, _, _, N_minijpas, x_im, y_im = load_minijpas_jnep(cat_list)
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
                               ew_oth, obs=True, mask=mask)
    other_lines = identify_lines(line_other, pm_flx, cont_est_other)

    N_sources = pm_flx.shape[1]

    mag_cut = (mag > mag_min) & (mag < mag_max)

    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    mask = (lya_lines >= nb_min) & (lya_lines <= nb_max) & mag_cut
    nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont_est_lya, z_Arr, mask=mask
    )
    # Save the selection
    selection = {
        'src': np.where(nice_lya)[0],
        'tile_id': tile_id[nice_lya],
        'x_im': x_im[nice_lya],
        'y_im': y_im[nice_lya],
        'nb_sel': lya_lines[nice_lya]
    }
    with open('npy/selection.npy', 'wb') as f:
        pickle.dump(selection, f)

    # Estimate Luminosity
    _, _, L_Arr, _, _, _ = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    L_Lbin_err = np.load('npy/L_nb_err.npy')
    median_L = np.load('npy/L_bias.npy')
    L_binning = np.load('npy/L_nb_err_binning.npy')
    L_bin_c = [L_binning[i: i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]

    # Correct L_Arr with the median
    mask_median_L = (median_L < 10)
    L_Arr = L_Arr - np.interp(L_Arr, np.log10(L_bin_c)
                              [mask_median_L], median_L[mask_median_L])

    # Apply bin err
    L_binning_position = binned_statistic(
        10 ** L_Arr, None, 'count', bins=L_binning
    ).binnumber
    L_binning_position[L_binning_position > len(
        L_binning) - 2] = len(L_binning) - 2
    L_e_Arr = L_Lbin_err[L_binning_position]

    L_bins = np.load('npy/puricomp2d_L_bins.npy')
    r_bins = np.load('npy/puricomp2d_r_bins.npy')
    puri2d_minijpas = np.load('npy/puri2d_minijpas.npy')
    comp2d_minijpas = np.load('npy/comp2d_minijpas.npy')
    puri2d_jnep = np.load('npy/puri2d_jnep.npy')
    comp2d_jnep = np.load('npy/comp2d_jnep.npy')

    bins = np.log10(L_binning)

    N_sources = pm_flx.shape[1]
    is_minijpas_source = np.ones(N_sources).astype(bool)
    is_minijpas_source[N_minijpas:] = False

    print(f'nice miniJPAS = {count_true(nice_lya & is_minijpas_source)}')
    print(f'nice J-NEP = {count_true(nice_lya & ~is_minijpas_source)}')

    volume = effective_volume(nb_min, nb_max, 'both')
    volume_mj = effective_volume(nb_min, nb_max, 'minijpas')
    volume_jn = effective_volume(nb_min, nb_max, 'jnep')

    b = bins

    LF_bins = np.array([(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)])

    bin_width = np.array([b[i + 1] - b[i] for i in range(len(b) - 1)])

    L_LF_err_percentiles = LF_perturb_err(
        L_Arr[is_minijpas_source], L_e_Arr[is_minijpas_source], nice_lya[is_minijpas_source],
        mag[is_minijpas_source], z_Arr[is_minijpas_source], starprob[is_minijpas_source],
        bins, puri2d_minijpas, comp2d_minijpas, L_bins, r_bins, 'minijpas',
        tile_id[is_minijpas_source]
    )
    L_LF_err_plus_mj = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    L_LF_err_minus_mj = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]
    hist_median_mj = L_LF_err_percentiles[1]

    L_LF_err_percentiles = LF_perturb_err(
        L_Arr[~is_minijpas_source], L_e_Arr[~is_minijpas_source], nice_lya[~is_minijpas_source],
        mag[~is_minijpas_source], z_Arr[~is_minijpas_source], starprob[~is_minijpas_source],
        bins, puri2d_jnep, comp2d_jnep, L_bins, r_bins, 'jnep',
        tile_id[~is_minijpas_source]
    )
    L_LF_err_plus_jn = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    L_LF_err_minus_jn = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]
    hist_median_jn = L_LF_err_percentiles[1]

    hist_median = hist_median_jn + hist_median_mj
    L_LF_err_plus = L_LF_err_plus_jn + L_LF_err_plus_mj
    L_LF_err_minus = L_LF_err_minus_jn + L_LF_err_minus_mj

    ###### RAW LF ######
    LF_raw = np.histogram(L_Arr[nice_lya], bins=bins)[0] / bin_width / volume
    ####################

    # Initialize dict to save the LFs
    LFs_dict = {'LF_bins': LF_bins}

    fig, ax = plt.subplots(figsize=(7, 5))
    # fig, ax = plt.subplots(figsize=(4, 4))

    # Plot the corrected total LF
    yerr_cor_plus = (hist_median + L_LF_err_plus **
                     2) ** 0.5 / bin_width / volume
    yerr_cor_minus = (hist_median + L_LF_err_minus **
                      2) ** 0.5 / bin_width / volume
    xerr = bin_width / 2
    LF_values = hist_median / bin_width / volume
    ax.errorbar(LF_bins, LF_values,
                yerr=[yerr_cor_minus, yerr_cor_plus], xerr=xerr,
                marker='s', linestyle='', color='k', capsize=4,
                label='miniJPAS + J-NEP', zorder=99)
    LFs_dict['LF_total'] = LF_values
    LFs_dict['LF_total_err'] = [yerr_cor_minus, yerr_cor_plus, xerr]

    # Plot the total raw LF
    ax.plot(LF_bins, LF_raw, ls='', markerfacecolor='none', markeredgecolor='dimgray',
            marker='^', markersize=11, zorder=4, label='Raw LF (miniJPAS + J-NEP)')
    LFs_dict['LF_total_raw'] = LF_raw

    # Plot the corrected J-NEP LF
    yerr_cor_plus = (hist_median_jn + L_LF_err_plus_jn **
                     2) ** 0.5 / bin_width / volume_jn
    yerr_cor_minus = (hist_median_jn + L_LF_err_minus_jn **
                      2) ** 0.5 / bin_width / volume_jn
    xerr = bin_width / 2
    LF_values = hist_median_jn / bin_width / volume_jn
    ax.errorbar(LF_bins + 0.028, LF_values,
                yerr=[yerr_cor_minus, yerr_cor_plus], xerr=xerr,
                marker='^', linestyle='', markersize=10, color='g',
                label='J-NEP', zorder=2)
    LFs_dict['LF_jnep'] = LF_values
    LFs_dict['LF_jnep_err'] = [yerr_cor_minus, yerr_cor_plus, xerr]

    # Plot the corrected miniJPAS LF
    yerr_cor_plus = (hist_median_mj + L_LF_err_plus_mj **
                     2) ** 0.5 / bin_width / volume_mj
    yerr_cor_minus = (hist_median_mj + L_LF_err_minus_mj **
                      2) ** 0.5 / bin_width / volume_mj
    xerr = bin_width / 2
    LF_values = hist_median_mj / bin_width / volume_mj
    ax.errorbar(LF_bins + 0.014, LF_values,
                yerr=[yerr_cor_minus, yerr_cor_plus], xerr=xerr,
                marker='^', linestyle='', markersize=10, color='m',
                label='miniJPAS', zorder=3)
    LFs_dict['LF_minijpas'] = LF_values
    LFs_dict['LF_minijpas_err'] = [yerr_cor_minus, yerr_cor_plus, xerr]

    # Save the dict
    folder_name = (
        f'LF_r{mag_min}-{mag_max}_z{z_min:0.1f}-{z_max:0.1f}_ew{ew0_cut}_ewoth{ew_oth}'
        f'_{cont_est_m}'
    )
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
    ax.legend(fontsize=7)

    ax.set_title(
        fr'r{mag_min}-{mag_max}, z {z_min:0.2f}-{z_max:0.2f}'
    )

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
        (17, 23.5, 6, 20, 30, 400, 'nb'),
        (17, 23, 6, 20, 30, 400, 'nb'),
        # (17, 24, 6, 20, 0, 400, 'nb'),
        # (17, 24, 6, 20, 15, 400, 'nb'),

        # (17, 24, 15, 22, 30, 400, 'nb'),
        # (17, 24, 5, 14, 30, 400, 'nb'),
    ]

    for params in LF_parameters:
        print(
            'mag{0}-{1}, nb{2}-{3}, ew0_lya={4}, ew_oth={5}, cont_est_method={6}'
            .format(*params))
        make_corrections(params)
        make_the_LF(params)