from minijpas_LF_and_puricomp import add_errors, search_lines
from load_mocks import ensemble_mock
from my_functions import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='ignore')

matplotlib.rcParams.update({'font.size': 13})
matplotlib.use('TkAgg')

qso_LAEs_frac = 1.

gal_area = 3
bad_qso_area = 400
good_qso_area = 400 / qso_LAEs_frac

# the proportional factors are made in relation to bad_qso
# so bad_qso_factor = 1
gal_factor = bad_qso_area / gal_area
good_qso_factor = bad_qso_area / good_qso_area

w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67
filter_tags = load_filter_tags()


def make_hist_plot(nice_lya, z_cut, lya_lines, ew0_cut, nb_min, nb_max):
    fig, ax = plt.subplots(figsize=(8, 4))

    mask = nice_lya & is_gal & z_cut
    w0 = w_central[lya_lines[mask]] / (1 + zspec[mask])

    bin_min = 1400
    bin_max = 6000
    bins = np.linspace(bin_min, bin_max, 70)
    bins_c = [bins[i: i + 2].sum() * 0.5 for i in range(len(bins) - 1)]

    hist, _ = np.histogram(w0, bins)
    ax.fill_between(bins_c, hist / gal_area,
                    step='pre', color='dimgray')

    gal_line_w = [2799, 4861, 3727, 5008, 2326]
    gal_line_name = ['MgII', r'H$\beta$', 'OII', 'OIII', 'CII']
    for w, name in zip(gal_line_w, gal_line_name):
        ax.axvline(w, color='orange', linestyle='--', zorder=-99)
        if name != 'OIII':
            ax.text(w - 70, 340, name, fontsize=13)
        else:
            ax.text(w - 20, 340, name, fontsize=13)

    ax.set_xlim(bin_min, bin_max)
    ax.set_ylim(1e-2, 300)
    ax.set_yscale('log')
    ax.set_xlabel('$\lambda_0$ ($\AA$)', fontsize=14)
    ax.set_ylabel('Sources density (deg$^{-2})$', fontsize=14)

    fig.tight_layout()
    fig.savefig(f'figures/GAL_contaminants_w0_hist_ew0min{ew0_cut}_nb{nb_min}-{nb_max}.pdf',
                bbox_inches='tight', facecolor='w', edgecolor='w')

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = np.linspace(890, 3700, 70)
    bins_c = [bins[i: i + 2].sum() * 0.5 for i in range(len(bins) - 1)]

    mask = nice_lya & is_qso & (zspec < 2)
    w0 = w_central[lya_lines[mask]] / (1 + zspec[mask])
    hbad, _ = np.histogram(w0, bins)
    mask = nice_lya & is_qso & (zspec > 2)
    w0 = w_central[lya_lines[mask]] / (1 + zspec[mask])
    hgood, _ = np.histogram(w0, bins)

    ax.fill_between(bins_c, (hbad + hgood * 0.5) /
                    bad_qso_area, step='pre', color='dimgray')

    qso_line_w = [1549.48, 1908.73, 2799.12, 2326.00, 1215.67, 1025, 1399.8]
    qso_line_name = ['CIV', 'CIII', 'MgII', 'CII',
                     r'Ly$\alpha$\n+NV', 'OVI', r'SiIV\n+OIV']

    for w, name in zip(qso_line_w, qso_line_name):
        ax.axvline(w, color='orange', linestyle='--', zorder=-99)
        if name == r'SiIV\n+OIV':
            ax.text(w - 120, 70, name, fontsize=13)
        elif name == r'Ly$\alpha$\n+NV':
            ax.text(w - 50, 70, name, fontsize=13)
        else:
            ax.text(w - 70, 340, name, fontsize=13)

    ax.set_yscale('log')
    ax.set_xlabel('$\lambda_0$ ($\AA$)', fontsize=14)
    ax.set_ylabel('Sources density (deg$^{-2})$', fontsize=14)
    ax.set_ylim(1e-2, 300)

    # fig.tight_layout()
    fig.savefig(f'figures/QSO_contaminants_w0_hist_ew0min{ew0_cut}_nb{nb_min}-{nb_max}.pdf',
                bbox_inches='tight', facecolor='w', edgecolor='w')


if __name__ == '__main__':
    name_qso = 'QSO_contaminants'
    name_qso_bad = 'QSO_LAES'
    name_gal = f'GAL_LC_lines_0'
    name_sf = f'LAE_12.5deg_z2-4.25_train_minijpas_VUDS_0'

    pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal,\
        is_LAE, where_hiL, _, L_NV, EW_NV, hiL_qso_area, good_qso_area  =\
             ensemble_mock(name_qso, name_gal, name_sf,
                           name_qso_bad, add_errs=False, mag_min=17, mag_max=24)

    pm_flx, pm_err = add_errors(pm_flx, apply_err=True,
                                survey_name='minijpasAEGIS003')

    where_bad_flx = ~np.isfinite(pm_flx)
    pm_flx[where_bad_flx] = 0.
    pm_err[where_bad_flx] = 9999999999.

    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mag[np.isnan(mag)] = 99.

    N_sources = pm_flx.shape[1]

    params = [
        (30, 100, 1, 24),
    ]
    
    for params_set in params:
        (ew0_cut, ew_other, nb_min, nb_max) = params_set
        print(params_set)

        cont_est_lya, cont_err_lya, lya_lines, other_lines, z_Arr, nice_z =\
            search_lines(pm_flx, pm_err, ew0_cut, ew_other, zspec, 'nb')

        z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
        z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

        z_cut = (lya_lines >= nb_min) & (lya_lines <= nb_max)
        zspec_cut = (z_min < zspec) & (zspec < z_max)
        mag_cut = (mag > 17) & (mag < 24)

        N_sources = len(mag_cut)
        snr = np.empty(N_sources)
        for src in range(N_sources):
            l = lya_lines[src]
            snr[src] = pm_flx[l, src] / pm_err[l, src]
        nice_lya_mask = (lya_lines >= nb_min) & (lya_lines <= nb_max) & mag_cut & (snr > 6)

        # Nice lya selection
        nice_lya = nice_lya_select(lya_lines, other_lines, pm_flx, pm_err,
                                cont_est_lya, z_Arr, mask=nice_lya_mask)

        z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
        z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

        z_cut = (z_min < z_Arr) & (z_Arr < z_max)
        make_hist_plot(nice_lya, z_cut, lya_lines, ew0_cut, nb_min, nb_max)
