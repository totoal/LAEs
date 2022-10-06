from minijpas_LF_and_puricomp import add_errors
from load_mocks import ensemble_mock
from my_functions import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='ignore')

matplotlib.rcParams.update({'font.size': 13})
matplotlib.use('TkAgg')

qso_LAEs_frac = 0.3

gal_area = 5.54
bad_qso_area = 200
good_qso_area = 400 / qso_LAEs_frac

# the proportional factors are made in relation to bad_qso
# so bad_qso_factor = 1
gal_factor = bad_qso_area / gal_area
good_qso_factor = bad_qso_area / good_qso_area

w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67
filter_tags = load_filter_tags()


def search_LAEs(ew0_cut, ew_other, nb_min, nb_max, pm_flx, pm_err, zspec):
    # Lya search
    cont_est_lya, cont_err_lya = estimate_continuum(
        pm_flx, pm_err, IGM_T_correct=True)
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut)
    lya_lines, lya_cont_lines, line_widths = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    cont_est_other, cont_err_other = estimate_continuum(
        pm_flx, pm_err, IGM_T_correct=False)
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
                               ew_other, obs=True)
    other_lines = identify_lines(line_other, cont_est_other, pm_err)

    # Compute z
    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    mag_min = 17
    mag_max = 24

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1
    print(f'z interval: ({z_min:0.2f}, {z_max:0.2f})')

    z_cut = (z_min < z_Arr) & (z_Arr < z_max)
    mag_cut = (mag > mag_min) & (mag < mag_max)

    snr = np.empty(N_sources)
    for src in range(N_sources):
        l = lya_lines[src]
        snr[src] = pm_flx[l, src] / pm_err[l, src]

    nice_lya_mask = z_cut & mag_cut & (snr > 6)
    nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont_est_lya, z_Arr, mask=nice_lya_mask
    )

    return nice_lya, z_Arr, lya_lines


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
                     r'Ly$\alpha$', r'Ly$\beta$', 'SiIV\n+OIV']

    for w, name in zip(qso_line_w, qso_line_name):
        ax.axvline(w, color='orange', linestyle='--', zorder=-99)
        if name != 'SiIV\n+OIV':
            ax.text(w - 70, 340, name, fontsize=13)
        else:
            ax.text(w - 120, 70, name, fontsize=13)

    ax.set_yscale('log')
    ax.set_xlabel('$\lambda_0$ ($\AA$)', fontsize=14)
    ax.set_ylabel('Sources density (deg$^{-2})$', fontsize=14)
    ax.set_ylim(1e-2, 300)

    fig.tight_layout()
    fig.savefig(f'figures/QSO_contaminants_w0_hist_ew0min{ew0_cut}_nb{nb_min}-{nb_max}.pdf',
                bbox_inches='tight', facecolor='w', edgecolor='w')


if __name__ == '__main__':
    field_name = 'minijpasAEGIS001'
    name_qso = 'QSO_100000_0'
    name_qso_bad = f'QSO_double_train_minijpas_DR16_D_0'
    name_gal = f'GAL_LC_lines_0'
    name_sf = f'LAE_12.5deg_z2-4.25_train_minijpas_0'

    pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal, is_LAE, where_hiL, _ =\
        ensemble_mock(name_qso, name_gal, name_sf, name_qso_bad, add_errs=False)

    pm_flx, pm_err = add_errors(pm_flx, apply_err=True,
                                survey_name='minijpasAEGIS001')

    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mag[np.isnan(mag)] = 99.

    N_sources = pm_flx.shape[1]

    params = [
        (30, 400, 2, 5),
        (30, 400, 5, 8),
        (30, 400, 8, 11),
        (30, 400, 11, 14),
        (30, 400, 14, 17),
        (30, 400, 17, 20),
    ]
    
    for params_set in params:
        (ew0_cut, ew_other, nb_min, nb_max) = params_set
        print(params_set)
        nice_lya, z_Arr, lya_lines =\
            search_LAEs(ew0_cut, ew_other, nb_min, nb_max, pm_flx, pm_err, zspec)

        z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
        z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

        z_cut = (z_min < z_Arr) & (z_Arr < z_max)
        make_hist_plot(nice_lya, z_cut, lya_lines, ew0_cut, nb_min, nb_max)
