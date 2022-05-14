#!/home/alberto/miniconda3/bin/python3

import numpy as np
np.seterr(all='ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10})

import seaborn as sns

import pandas as pd

from my_functions import *
from LF_puricomp_corrections import weights_LF

import glob
import os

from scipy.stats import binned_statistic

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# Useful definitions
w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67
filter_tags = load_filter_tags()
gal_factor = 12.57
z_nb_Arr = w_central[:-4] / w_lya - 1

def load_mocks():
    ## Load my QSO catalog

    filename = '/home/alberto/almacen/Source_cats/QSO_100000_0/'
    files = glob.glob(filename +'data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    data_qso = pd.concat(fi, axis=0, ignore_index=True)

    qso_flx = data_qso.to_numpy()[:, 1 : 60 + 1].T
    qso_err = data_qso.to_numpy()[:, 60 + 1 : 120 + 1].T

    qso_flx += qso_err * np.random.normal(size=qso_err.shape)

    EW_qso = data_qso['EW0'].to_numpy()
    qso_zspec = data_qso['z'].to_numpy()

    ## Load my GAL catalog

    filename = '/home/alberto/almacen/Source_cats/GAL_100000_0/'
    files = glob.glob(filename +'data*')
    files.sort()
    fi = []

    for name in files:
        fi.append(pd.read_csv(name))

    data_gal = pd.concat(fi, axis=0, ignore_index=True)

    gal_flx = data_gal.to_numpy()[:, 1 : 60 + 1].T
    gal_err = data_gal.to_numpy()[:, 60 + 1 : 120 + 1].T

    gal_flx += gal_err * np.random.normal(size=gal_err.shape)

    EW_gal = np.zeros(data_gal['z'].to_numpy().shape)
    gal_zspec = data_gal['z'].to_numpy()

    ## Load SF catalog

    filename = '/home/alberto/almacen/Source_cats/LAE_10deg_z2-4.25_0/'
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

    sf_flx += sf_err * np.random.normal(size=sf_err.shape)

    EW_sf = data['EW0'].to_numpy()
    sf_zspec = data['z'].to_numpy()

    pm_flx = np.hstack((qso_flx, sf_flx, gal_flx))
    pm_err = np.hstack((qso_err, sf_err, gal_err))
    zspec = np.concatenate((qso_zspec, sf_zspec, gal_zspec))
    EW_lya = np.concatenate((EW_qso, EW_sf, EW_gal))

    N_sf = sf_flx.shape[1]
    N_qso = qso_flx.shape[1]
    N_gal = gal_flx.shape[1]

    sf_dL = cosmo.luminosity_distance(sf_zspec).to(u.cm).value

    sf_L = data['L_lya'].to_numpy()
    qso_L = data_qso['L_lya'].to_numpy()
    gal_L = np.zeros(EW_gal.shape)

    sf_flambda = 10 ** sf_L / (4*np.pi * sf_dL **2)
    qso_flambda = data_qso['F_line']
    gal_flambda = np.zeros(EW_gal.shape)

    L_lya = np.concatenate((qso_L, sf_L, gal_L))
    fline = np.concatenate((qso_flambda, sf_flambda, gal_flambda))

    is_qso = np.concatenate((np.ones(N_qso), np.zeros(N_sf + N_gal))).astype(bool)
    is_sf = np.concatenate((np.zeros(N_qso), np.ones(N_sf), np.zeros(N_gal))).astype(bool)
    is_gal = np.concatenate((np.zeros(N_qso), np.zeros(N_sf), np.ones(N_gal))).astype(bool)

    return pm_flx, pm_err, zspec, EW_lya, L_lya, fline, is_qso, is_sf, is_gal

def search_lines(pm_flx, pm_err, ew0_cut, zspec):
    # Lya search
    cont_est_lya, cont_err_lya = estimate_continuum(pm_flx, pm_err, IGM_T_correct=True)
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut)
    lya_lines, lya_cont_lines, line_widths = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    cont_est_other, cont_err_other = estimate_continuum(pm_flx, pm_err, IGM_T_correct=False)
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
        in_bin = (10 ** L_Arr >= L_binning[i]) & (10 ** L_Arr < L_binning[i + 1])
        if count_true(in_bin) == 0:
            L_Lbin_err_plus[i] = last[0]
            L_Lbin_err_minus[i] = last[1]
            continue
        perc = np.nanpercentile((10 ** L_Arr - 10 ** L_lya)[in_bin], [16, 50, 84])
        L_Lbin_err_plus[i] = perc[2] - perc[1]
        
        last = [L_Lbin_err_plus[i], L_Lbin_err_minus[i]]
        median[i] = perc[1]

    return L_Lbin_err_plus, median

def purity_or_completeness_plot(which_one, mag, nbs_to_consider, lya_lines,
                                z_Arr, nice_lya, nice_z, L_Arr, mag_max,
                                mag_min, ew0_cut, is_gal, is_sf, is_qso, zspec,
                                L_lya, dirname, ew_cut):
    fig, ax = plt.subplots(figsize=(4, 4))

    bins2 = np.linspace(42, 45.5, 15)

    b_c = [0.5 * (bins2[i] + bins2[i + 1]) for i in range(len(bins2) - 1)]

    this_mag_cut = (mag < mag_max) & (mag > mag_min)

    for nb in nbs_to_consider:
        nb_mask = (lya_lines == nb)

        z_min = (w_central[nb] - nb_fwhm_Arr[nb] * 0.5) / w_lya - 1
        z_max = (w_central[nb] + nb_fwhm_Arr[nb] * 0.5) / w_lya - 1

        this_zspec_cut = (z_min < zspec) & (zspec < z_max)
        totals_mask = this_zspec_cut & this_mag_cut & ew_cut

        goodh_puri = L_Arr[nice_lya & nice_z & totals_mask]
        goodh_comp = L_lya[nice_lya & nice_z & totals_mask]
        badh = L_Arr[nice_lya & ~nice_z & (is_qso | is_sf) & nb_mask & this_mag_cut]
        badh_gal = L_Arr[nice_lya & ~nice_z & is_gal & nb_mask & this_mag_cut]

        hg_puri, _ = np.histogram(goodh_puri, bins=bins2)
        hg_comp, _ = np.histogram(goodh_comp, bins=bins2)
        hb, _ = np.histogram(badh, bins=bins2)
        hb_gal, _ = np.histogram(badh_gal, bins=bins2)
        hb_gal = hb_gal * gal_factor
        totals, _ = np.histogram(L_lya[totals_mask], bins=bins2)

        if which_one == 'Completeness':
            ax.plot(
                b_c, hg_comp / totals, marker='s',
                label=filter_tags[nb], zorder=99, alpha=0.5
            )
        if which_one == 'Purity':
            ax.plot(
                b_c, hg_puri / (hg_puri + hb + hb_gal), marker='s',
                label=filter_tags[nb], zorder=99, alpha=0.5
            )

    nb_min = nbs_to_consider[0]
    nb_max = nbs_to_consider[-1]
    nb_mask = (lya_lines >= nb_min) & (lya_lines <= nb_max)
    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1
    this_zspec_cut = (z_min < zspec) & (zspec < z_max)

    totals_mask = this_zspec_cut & this_mag_cut & ew_cut

    goodh_puri = L_Arr[nice_lya & nice_z & totals_mask]
    goodh_comp = L_lya[nice_lya & nice_z & totals_mask]
    badh = L_Arr[nice_lya & ~nice_z & (is_qso | is_sf) & this_mag_cut & nb_mask]
    badh_gal = L_Arr[nice_lya & ~nice_z & is_gal & this_mag_cut & nb_mask]

    hg_puri, _ = np.histogram(goodh_puri, bins=bins2)
    hg_comp, _ = np.histogram(goodh_comp, bins=bins2)
    hb, _ = np.histogram(badh, bins=bins2)
    hb_gal, _ = np.histogram(badh_gal, bins=bins2)
    hb_gal = hb_gal * gal_factor
    totals, _ = np.histogram(L_lya[totals_mask], bins=bins2)

    if which_one == 'Completeness':
        ax.plot(b_c, hg_comp / totals, marker='s', label='All', zorder=99, c='k')
    if which_one == 'Purity':
        ax.plot(b_c, hg_puri / (hg_puri + hb + hb_gal), marker='s', label='All', zorder=99, c='k')

    ax.set_xlabel(r'$\log L$ (erg$\,$s$^{-1}$)')
    ax.set_ylabel(which_one.lower())

    ax.set_xlim((42, 45.5))
    ax.set_ylim((0, 1))
    ax.legend()
    ax.set_title(f'r{mag_min}-{mag_max}, EW0_cut = {ew0_cut}, z{z_min:0.2f}-{z_max:0.2f}')

    plt.savefig(f'{dirname}/{which_one}', bbox_inches='tight', facecolor='white')
    plt.close()

def plot_puricomp_grids(puri, comp, L_bins, r_bins, dirname):
    fig = plt.figure(figsize=(7, 6))

    width = 1
    height = 1
    spacing = 0.1
    cbar_width = 0.3

    ### ADD AXES
    ax0 = fig.add_axes([0, 0, width, height])
    ax1 = fig.add_axes([width + spacing, 0, width, height])
    axc = fig.add_axes([width * 2 + spacing * 2, 0, cbar_width, height])

    ### PLOT STUFF
    cmap = 'Spectral'
    sns.heatmap(puri.T, ax=ax0, vmin=0, vmax=1, cbar_ax=axc, cmap=cmap)
    sns.heatmap(comp.T, ax=ax1, vmin=0, vmax=1, cbar=False, cmap=cmap)

    ### TICKS
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

    ### SPINES
    ax0.spines[:].set_visible(True)
    ax1.spines[:].set_visible(True)

    ### TITLES
    ax0.set_title('Purity', fontsize=25)
    ax1.set_title('Completeness', fontsize=25)

    plt.savefig(f'{dirname}/PuriComp2D', bbox_inches='tight', facecolor='white')
    plt.close()

def puricomp_corrections(mag_min, mag_max, L_Arr, L_e_Arr, nice_lya, nice_z,
                         mag, zspec_cut, z_cut, mag_cut, ew_cut, L_bins, L_lya, is_gal):
    r_bins = np.linspace(mag_min, mag_max, 10 + 1)

    # Perturb L
    N_iter = 1000
    h2d_nice_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))
    h2d_sel_gal_i = np.empty((len(L_bins) - 1, len(r_bins) - 1, N_iter))

    for k in range(N_iter):
        L_perturbed = np.log10(
            10 ** L_Arr + L_e_Arr * np.random.randn(len(L_e_Arr))
        )
        L_perturbed[np.isnan(L_perturbed)] = 0.

        h2d_nice_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & nice_z & zspec_cut],
            mag[nice_lya & nice_z & zspec_cut],
            bins=[L_bins, r_bins]
        )

        h2d_sel_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & ~is_gal & z_cut],
            mag[nice_lya & ~is_gal & z_cut],
            bins=[L_bins, r_bins]
        )

        h2d_sel_gal_i[..., k], _, _ = np.histogram2d(
            L_perturbed[nice_lya & is_gal & z_cut],
            mag[nice_lya & is_gal & z_cut],
            bins=[L_bins, r_bins]
        )

    # Take the median
    h2d_nice = np.median(h2d_nice_i, axis=2)
    h2d_sel = np.median(h2d_sel_i, axis=2)
    h2d_sel_gal = np.median(h2d_sel_gal_i, axis=2)
    h2d_parent, _, _ = np.histogram2d(
        L_lya[zspec_cut & mag_cut & ew_cut],
        mag[zspec_cut & mag_cut & ew_cut],
        bins=[L_bins, r_bins]
    )

    puri2d = h2d_nice / (h2d_sel + h2d_sel_gal * gal_factor)
    puri2d_err = (
        h2d_nice / (h2d_sel + h2d_sel_gal * gal_factor) ** 2
        + (h2d_nice / (h2d_sel + h2d_sel_gal * gal_factor)**2) ** 2
        * (h2d_sel + gal_factor**2 * h2d_sel_gal)
    ) ** 0.5
    comp2d = h2d_nice / h2d_parent
    comp2d_err = h2d_nice ** 0.5 / h2d_parent

    return puri2d, comp2d, puri2d_err, comp2d_err, L_bins, r_bins

def all_corrections(params, pm_flx, pm_err, zspec, EW_lya, L_lya, is_gal,
                    is_qso, is_sf):
    mag_min, mag_max, nb_min, nb_max, ew0_cut, ew_oth = params

    # Vector of magnitudes in r band
    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mag[np.isnan(mag)] = 99.

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1
    print(f'z interval: ({z_min:0.2f}, {z_max:0.2f})')

    # Make the directory if it doesn't exist
    folder_name = f'LF_r{mag_min}-{mag_max}_z{z_min:0.1f}-{z_max:0.1f}_ew{ew0_cut}_ewoth{ew_oth}'
    dirname = f'/home/alberto/cosmos/LAEs/Luminosity_functions/{folder_name}'
    os.makedirs(dirname, exist_ok=True)

    # Estimate continuum, search lines
    cont_est_lya, cont_err_lya, lya_lines, other_lines, z_Arr, nice_z =\
        search_lines(pm_flx, pm_err, ew0_cut, zspec)

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

    ### Estimate Luminosity
    _, _, L_Arr, _, _, _ = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    ML_predict_mask = (mag < 23) & (L_Arr > 0)
    L_Arr[ML_predict_mask] = ML_predict_L(
        pm_flx[:, ML_predict_mask], pm_err[:, ML_predict_mask],
        z_Arr[ML_predict_mask], L_Arr[ML_predict_mask], 'RFmag15-23'
    )

    ML_predict_mask = (mag > 23) & (L_Arr > 0)
    L_Arr[ML_predict_mask] = ML_predict_L(
        pm_flx[:, ML_predict_mask], pm_err[:, ML_predict_mask],
        z_Arr[ML_predict_mask], L_Arr[ML_predict_mask], 'RFmag23-23.5'
    )

    ## Compute and save L corrections and errors
    L_binning = np.logspace(41, 46, 20 + 1)
    L_Lbin_err, median_L = compute_L_Lbin_err(
        L_Arr[nice_lya & nice_z], L_lya[nice_z & nice_lya], L_binning
    )
    np.save('npy/L_nb_err.npy', L_Lbin_err)
    np.save('npy/L_bias.npy', median_L)
    np.save('npy/L_nb_err_binning.npy', L_binning)

    # Apply bin err
    L_binning_position = binned_statistic(
            10 ** L_Arr, None, 'count', bins=L_binning
    ).binnumber
    L_binning_position[L_binning_position > len(L_binning) - 2] = len(L_binning) - 2
    L_e_Arr = L_Lbin_err[L_binning_position]

    bins = np.log10(L_binning)

    # Compute puri/comp 2D
    puri2d, comp2d, puri2d_err, comp2d_err, L_bins, r_bins = puricomp_corrections(
        mag_min, mag_max, L_Arr, L_e_Arr, nice_lya,
        nice_z, mag, zspec_cut, z_cut, mag_cut, ew_cut, bins,
        L_lya, is_gal
    )

    plot_puricomp_grids(puri2d, comp2d, L_bins, r_bins, dirname)

    np.save('npy/puri2d.npy', puri2d)
    np.save('npy/comp2d.npy', comp2d)
    np.save('npy/puri2d_err.npy', puri2d_err)
    np.save('npy/comp2d_err.npy', comp2d_err)
    np.save('npy/puricomp2d_L_bins.npy', L_bins)
    np.save('npy/puricomp2d_r_bins.npy', r_bins)

    nbs_to_consider = np.arange(nb_min, nb_max + 1)

    for which_one in ['Purity', 'Completeness']:
        purity_or_completeness_plot(
            which_one, mag, nbs_to_consider, lya_lines, z_Arr,
            nice_lya, nice_z, L_Arr, mag_max, mag_min, ew0_cut,
            is_gal, is_sf, is_qso, zspec, L_lya, dirname, ew_cut,
        )

def make_corrections(params):
    pm_flx, pm_err, zspec, EW_lya, L_lya,\
        _, is_qso, is_sf, is_gal = load_mocks()
    all_corrections(
        params, pm_flx, pm_err, zspec, EW_lya, L_lya, is_gal,
        is_qso, is_sf
    )

def Zero_point_error(tile_id_Arr, catname):
    ## Load Zero Point magnitudes
    zpt_cat = pd.read_csv(f'csv/{catname}.CalibTileImage.csv', sep=',', header=1)

    zpt_mag = zpt_cat['ZPT'].to_numpy()
    zpt_err = zpt_cat['ERRZPT'].to_numpy()

    ones = np.ones((len(w_central), len(zpt_mag)))

    zpt_err = (
        mag_to_flux(ones * zpt_mag, w_central.reshape(-1, 1))
        - mag_to_flux(ones * (zpt_mag + zpt_err), w_central.reshape(-1, 1))
    )

    # Duplicate rows to match the tile_ID of each source
    idx = np.empty(tile_id_Arr.shape).astype(int)

    zpt_id = zpt_cat['TILE_ID'].to_numpy()
    for src in range(len(tile_id_Arr)):
        idx[src] = np.where(
            (zpt_id == tile_id_Arr[src]) & (zpt_cat['IS_REFERENCE_METHOD'] == 1)
        )[0][0]
    
    zpt_err = zpt_err[:, idx]

    return zpt_err

def load_minijpas_jnep():
    pm_flx = np.array([]).reshape(60, 0)
    pm_err = np.array([]).reshape(60, 0)
    tile_id = np.array([])
    parallax_sn = np.array([])
    pmra_sn = np.array([])
    pmdec_sn = np.array([])
    starprob = np.array([])
    spCl = np.array([])
    zsp = np.array([])

    N_minijpas = 0
    split_converter = lambda s: np.array(s.split()).astype(float)
    sum_flags = lambda s: np.sum(np.array(s.split()).astype(float))

    for name in ['minijpas', 'jnep']:
        cat = pd.read_csv(f'csv/{name}.Flambda_aper3_photoz_gaia_3.csv', sep=',', header=1,
            converters={0: int, 1: int, 2: split_converter, 3: split_converter, 4: sum_flags,
            5: sum_flags})

        cat = cat[np.array([len(x) for x in cat['FLUX_APER_3_0']]) != 0] # Drop bad rows due to bad query
        cat = cat[(cat.FLAGS == 0) & (cat.MASK_FLAGS == 0)] # Drop flagged
        cat = cat.reset_index()

        tile_id_i = cat['TILE_ID'].to_numpy()

        parallax_i = cat['parallax'].to_numpy() / cat['parallax_error'].to_numpy()
        pmra_i = cat['pmra'].to_numpy() / cat['pmra_error'].to_numpy()
        pmdec_i = cat['pmdec'].to_numpy() / cat['pmdec_error'].to_numpy()

        pm_flx_i = np.stack(cat['FLUX_APER_3_0'].to_numpy()).T * 1e-19
        pm_err_i = np.stack(cat['FLUX_RELERR_APER_3_0'].to_numpy()).T * pm_flx_i

        if name == 'minijpas':
            N_minijpas = pm_flx_i.shape[1]

        starprob_i = cat['morph_prob_star']

        pm_err_i = (pm_err_i ** 2 + Zero_point_error(cat['TILE_ID'], name) ** 2) ** 0.5

        spCl_i = cat['spCl']
        zsp_i = cat['zsp']

        pm_flx = np.hstack((pm_flx, pm_flx_i))
        pm_err = np.hstack((pm_err, pm_err_i))
        tile_id = np.concatenate((tile_id, tile_id_i))
        pmra_sn = np.concatenate((pmra_sn, pmra_i))
        pmdec_sn = np.concatenate((pmdec_sn, pmdec_i))
        parallax_sn = np.concatenate((parallax_sn, parallax_i))
        starprob = np.concatenate((starprob, starprob_i))
        spCl = np.concatenate((spCl, spCl_i))
        zsp = np.concatenate((zsp, zsp_i))

    return pm_flx, pm_err, tile_id, pmra_sn, pmdec_sn, parallax_sn, starprob,\
        spCl, zsp, N_minijpas

def LF_perturb_err(L_Arr, L_e_Arr, nice_lya, mag, z_Arr, starprob,
                   bins, puri2d, comp2d, puri2d_err, comp2d_err, L_bins,
                   r_bins, tile_id):
    which_w = [0, 2]
    N_bins = len(bins) - 1

    N_iter = 200
    hist_i_mat = np.zeros((N_iter, N_bins))

    for k in range(N_iter):
        L_perturbed = np.log10(
            10 ** L_Arr + L_e_Arr * np.random.randn(len(L_e_Arr))
        )
        L_perturbed[np.isnan(L_perturbed)] = 0.


        puri, comp = weights_LF(
            L_perturbed[nice_lya], mag[nice_lya], puri2d, comp2d, puri2d_err, comp2d_err,
            L_bins, r_bins, z_Arr[nice_lya], starprob[nice_lya], tile_id, which_w, True
        )

        w = np.random.rand(len(puri))
        include_mask = (w < puri)
        w[:] = 1.
        w[~include_mask] = 0.
        w[include_mask] = 1. / comp[include_mask]
        w[np.isnan(w) | np.isinf(w)] = 0.

        hist_i_mat[k], _ = np.histogram(L_perturbed[nice_lya], bins=bins, weights=w)

    L_LF_err_percentiles = np.percentile(hist_i_mat, [16, 50, 84], axis=0)
    return L_LF_err_percentiles

def effective_volume(nb_min, nb_max, area):
    '''
    Due to NB overlap, specially when considering single filters, the volume probed by one
    NB has to be corrected because some sources could be detected in that NB or in either
    of the adjacent ones.
    '''
    z_min_overlap = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max_overlap = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    z_min_abs = (w_central[nb_min - 1] + nb_fwhm_Arr[nb_min - 1] * 0.5) / w_lya - 1
    z_max_abs = (w_central[nb_max + 1] - nb_fwhm_Arr[nb_min + 1] * 0.5) / w_lya - 1

    volume_abs = z_volume(z_min_abs, z_max_abs, area)
    volume_overlap = (
        z_volume(z_min_overlap, z_min_abs, area)
        + z_volume(z_max_abs, z_max_overlap, area)
    )

    return volume_abs + volume_overlap * 0.5

def make_the_LF(params):
    mag_min, mag_max, nb_min, nb_max, ew0_cut, ew_oth = params

    pm_flx, pm_err, tile_id, pmra_sn, pmdec_sn, parallax_sn, starprob, _, _,\
    N_minijpas = load_minijpas_jnep()
    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mask = mask_proper_motion(parallax_sn, pmra_sn, pmdec_sn)

    # Lya search
    cont_est_lya, cont_err_lya = estimate_continuum(pm_flx, pm_err, IGM_T_correct=True)
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut, mask=mask)
    lya_lines, lya_cont_lines, _ = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    cont_est_other, cont_err_other = estimate_continuum(pm_flx, pm_err, IGM_T_correct=False)
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

    _, _, L_Arr, L_e_Arr, _, _ = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    ML_predict_mask = (mag < 23) & (L_Arr > 0)
    L_Arr[ML_predict_mask] = ML_predict_L(
        pm_flx[:, ML_predict_mask], pm_err[:, ML_predict_mask],
        z_Arr[ML_predict_mask], L_Arr[ML_predict_mask], 'RFmag15-23'
    )

    ML_predict_mask = (mag > 23) & (L_Arr > 0)
    L_Arr[ML_predict_mask] = ML_predict_L(
        pm_flx[:, ML_predict_mask], pm_err[:, ML_predict_mask],
        z_Arr[ML_predict_mask], L_Arr[ML_predict_mask], 'RFmag23-23.5'
    )

    L_binning = np.load('npy/L_nb_err_binning.npy')
    L_Lbin_err = np.load('npy/L_nb_err.npy')

    # Apply bin err
    L_binning_position = binned_statistic(
            10 ** L_Arr, None, 'count', bins=L_binning
    ).binnumber
    L_binning_position[L_binning_position > len(L_binning) - 2] = len(L_binning) - 2
    L_e_Arr = L_Lbin_err[L_binning_position]

    L_bins = np.load('npy/puricomp2d_L_bins.npy')
    r_bins = np.load('npy/puricomp2d_r_bins.npy')
    puri2d = np.load('npy/puri2d.npy')
    comp2d = np.load('npy/comp2d.npy')
    puri2d_err = np.load('npy/puri2d_err.npy')
    comp2d_err = np.load('npy/comp2d_err.npy')

    bins = np.log10(L_binning)

    N_sources = pm_flx.shape[1]
    is_minijpas_source = np.ones(N_sources).astype(bool)
    is_minijpas_source[N_minijpas:] = False

    print(f'nice miniJPAS = {count_true(nice_lya & is_minijpas_source)}')
    print(f'nice J-NEP = {count_true(nice_lya & ~is_minijpas_source)}')

    _, b = np.histogram(L_Arr[nice_lya], bins=bins)

    LF_bins = np.array([(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)])

    bin_width = np.array([b[i + 1] - b[i] for i in range(len(b) - 1)])

    volume = effective_volume(nb_min, nb_max, 0.895 + 0.24)
    volume_mj = effective_volume(nb_min, nb_max, 0.895)
    volume_jn = effective_volume(nb_min, nb_max, 0.24)

    L_LF_err_percentiles = LF_perturb_err(
        L_Arr, L_e_Arr, nice_lya, mag, z_Arr, starprob, bins,
        puri2d, comp2d, puri2d_err, comp2d_err, L_bins, r_bins, tile_id
    )
    L_LF_err_plus = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    L_LF_err_minus = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]
    hist_median = L_LF_err_percentiles[1]

    L_LF_err_percentiles = LF_perturb_err(
        L_Arr[is_minijpas_source], L_e_Arr[is_minijpas_source], nice_lya[is_minijpas_source],
        mag[is_minijpas_source], z_Arr[is_minijpas_source], starprob[is_minijpas_source],
        bins, puri2d, comp2d, puri2d_err, comp2d_err, L_bins, r_bins, tile_id
    )
    L_LF_err_plus_mj = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    L_LF_err_minus_mj = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]
    hist_median_mj = L_LF_err_percentiles[1]

    L_LF_err_percentiles = LF_perturb_err(
        L_Arr[~is_minijpas_source], L_e_Arr[~is_minijpas_source], nice_lya[~is_minijpas_source],
        mag[~is_minijpas_source], z_Arr[~is_minijpas_source], starprob[~is_minijpas_source],
        bins, puri2d, comp2d, puri2d_err, comp2d_err, L_bins, r_bins, tile_id
    )
    L_LF_err_plus_jn = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    L_LF_err_minus_jn = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]
    hist_median_jn = L_LF_err_percentiles[1]

    fig, ax = plt.subplots(figsize=(7, 5))

    yerr_cor_plus = (hist_median + L_LF_err_plus ** 2) ** 0.5\
        / volume / bin_width
    yerr_cor_minus = (hist_median + L_LF_err_minus ** 2) ** 0.5\
        / volume / bin_width
    xerr = bin_width / 2
    ax.errorbar(LF_bins, hist_median / volume / bin_width,
        yerr= [yerr_cor_minus, yerr_cor_plus], xerr=xerr,
        marker='s', linestyle='', color='k', capsize=4,
        label='miniJPAS + J-NEP', zorder=99)

    yerr_cor_plus = (hist_median_jn + L_LF_err_plus_jn ** 2) ** 0.5\
        / volume_jn / bin_width
    yerr_cor_minus = (hist_median_jn + L_LF_err_minus_jn ** 2) ** 0.5\
        / volume_jn / bin_width
    xerr = bin_width / 2
    ax.errorbar(LF_bins + 0.024, hist_median_jn / volume_jn / bin_width,
        yerr= [yerr_cor_minus, yerr_cor_plus], xerr=xerr,
        marker='^', linestyle='', markersize=10, label='J-NEP')

    yerr_cor_plus = (hist_median_mj + L_LF_err_plus_mj ** 2) ** 0.5\
        / volume_mj / bin_width
    yerr_cor_minus = (hist_median_mj + L_LF_err_minus_mj ** 2) ** 0.5\
        / volume_mj / bin_width
    xerr = bin_width / 2
    ax.errorbar(LF_bins + 0.012, hist_median_mj / volume_mj / bin_width,
        yerr= [yerr_cor_minus, yerr_cor_plus], xerr=xerr,
        marker='^', linestyle='', markersize=10, label='miniJPAS')

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
        label='Spinoso2020 (2.2 < z < 3.25)'
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
        label='Zhang2021 (2 < z < 3.2)'
    )

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log L_{\mathrm{Ly}\alpha}$ (erg$\,$s$^{-1}$)')
    ax.set_ylabel(r'$\Phi$ (Mpc$^{-3}\,\Delta\logL^{-1}$)')
    ax.set_ylim(1e-8, 1e-2)
    ax.set_xlim(42, 46)
    ax.legend()

    ax.set_title(
        f'r{mag_min}-{mag_max}, EW0_cut = {ew0_cut}, z{z_min:0.2f}-{z_max:0.2f}'
    )

    folder_name = f'LF_r{mag_min}-{mag_max}_z{z_min:0.1f}-{z_max:0.1f}_ew{ew0_cut}_ewoth{ew_oth}'
    dirname = f'/home/alberto/cosmos/LAEs/Luminosity_functions/{folder_name}'
    os.makedirs(dirname, exist_ok=True)

    plt.savefig(f'{dirname}/LumFunc', bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    # Parameters of the LF:
    # (min_mag, max_mag, nb_min, nb_max, ew0_cut)
    
    LF_parameters = [
        (17, 24, 5, 15, 30, 400),
        (17, 24, 6, 6, 30, 400),
        (17, 24, 7, 7, 30, 400),
        (17, 24, 8, 8, 30, 400),
        (17, 24, 9, 9, 30, 400),
        (17, 24, 10, 10, 30, 400),
        (17, 24, 11, 11, 30, 400),
        (17, 24, 12, 12, 30, 400),
        (17, 24, 13, 13, 30, 400),
        (17, 24, 14, 14, 30, 400),
        (17, 24, 15, 15, 30, 400)
    ]

    for params in LF_parameters:
        make_corrections(params)
        make_the_LF(params)