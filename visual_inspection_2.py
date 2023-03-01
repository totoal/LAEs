import os

import pickle

from LAEs.my_functions import *
from LAEs.load_mocks import ensemble_mock
from LAEs.minijpas_LF_and_puricomp import add_errors, nb_or_3fm_cont, compute_L_Lbin_err
from LAEs.load_jpas_catalogs import load_minijpas_jnep
from visual_inspection import plot_paper

w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67
filter_tags = load_filter_tags()

def mock_selection():
    sf_frac = 0.1
    qso_LAE_frac = 1

    gal_area = 3
    bad_qso_area = 400
    good_qso_area = 400 * qso_LAE_frac
    sf_area = 200 * sf_frac

    name_qso_bad = 'QSO_LAES_2'
    name_qso = 'QSO_contaminants_2'
    name_gal = f'GAL_LC_lines_0'
    name_sf = f'LAE_12.5deg_z2-4.25_train_minijpas_VUDS_0'

    pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal,\
        is_LAE, where_hiL, _, L_NV, EW_NV, _, _ =\
        ensemble_mock(name_qso, name_gal, name_sf, name_qso_bad,
                    add_errs=False, sf_frac=sf_frac, qso_LAE_frac=qso_LAE_frac,
                    mag_min=17, mag_max=24)
    print(len(is_qso))

    L_lya_NV = np.log10(10**L_lya + 10**L_NV)

    pm_flx, pm_err = add_errors(pm_flx, apply_err=True,
                                survey_name='minijpasAEGIS004')

    where_bad_flx = ~np.isfinite(pm_flx)
    pm_flx[where_bad_flx] = 0.
    pm_err[where_bad_flx] = 9999999999.

    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mag[np.isnan(mag)] = 99.

    N_sources = pm_flx.shape[1]
    N_sources
    print(f'qso {count_true(is_qso)}')
    print(f'sf {count_true(is_sf)}')
    print(f'gal {count_true(is_gal)}')

    ew0_cut = 30
    ew_other = 100

    # Cont est
    cont_est_lya, cont_err_lya, cont_est_other, cont_err_other =\
            nb_or_3fm_cont(pm_flx, pm_err, 'nb')

    # Lya search
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut, sigma=3)
    lya_lines, lya_cont_lines, line_widths = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
        ew_other, obs=True, sigma=5)
    other_lines = identify_lines(line_other, cont_est_other, pm_err)

    # Compute z
    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    nice_z = np.abs(z_Arr - zspec) < 0.16

    mag_min = 17
    mag_max = 23.5

    nb_min = 1
    nb_max = 20

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1
    print(f'z interval: ({z_min:0.2f}, {z_max:0.2f})')

    z_cut = (z_min < z_Arr) & (z_Arr < z_max)
    zspec_cut = (z_min < zspec) & (zspec < z_max)
    mag_cut = (mag > mag_min) & (mag < mag_max)

    snr = np.empty(N_sources)
    for src in range(N_sources):
        l = lya_lines[src]
        snr[src] = pm_flx[l, src] / pm_err[l, src]

    nice_lya_mask = (lya_lines >= nb_min) & (lya_lines <= nb_max) & mag_cut & (snr > 6)
    nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont_est_lya, z_Arr, mask=nice_lya_mask
    )
    print(sum(nice_lya))

    EW_nb_Arr, EW_nb_e, L_Arr_uncorr, L_e_Arr, flambda, flambda_e = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    # Compute and save L corrections and errors
    L_binning = np.logspace(40, 47, 25 + 1)
    L_bin_c = [L_binning[i: i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]
    Lmask = nice_z & nice_lya & (L_lya > 42) & is_qso
    L_Lbin_err_plus, L_Lbin_err_minus, median_L = compute_L_Lbin_err(
        L_Arr_uncorr[Lmask], L_lya[Lmask], L_binning
    )

    mask_median_L = (median_L < 10)
    corr_L = np.interp(L_Arr_uncorr, np.log10(L_bin_c)
                    [mask_median_L], median_L[mask_median_L])
    # Correct L_Arr with the median
    L_Arr = L_Arr_uncorr - corr_L

    sel_class = np.zeros_like(nice_lya, dtype='<U32')
    sel_class[is_sf] = 'SFG'
    sel_class[is_qso & is_LAE] = 'LAE_QSO'
    sel_class[is_qso & ~is_LAE] = 'CONT_QSO'
    sel_class[is_gal] = 'GAL'
    prior_ratios = [1,
                    bad_qso_area/good_qso_area,
                    gal_area/good_qso_area,
                    sf_area/good_qso_area]

    other_lines_sel = []
    for ii, ll in enumerate(other_lines):
        if nice_lya[ii]:
            other_lines_sel.append(ll)

    mock_sel = {
        'flx': pm_flx[:, nice_lya],
        'flx_err': pm_err[:, nice_lya],
        'cont_est': cont_est_lya[:, nice_lya],
        'cont_err': cont_err_lya[:, nice_lya],
        'L_lya': L_Arr[nice_lya],
        'EW_lya': EW_nb_Arr[nice_lya],
        'z_NB': z_Arr[nice_lya],
        'nb_sel': lya_lines[nice_lya],
        'other_lines': other_lines_sel,
        'class': sel_class[nice_lya],
        'prior_ratios': prior_ratios,
    }
    return mock_sel

if __name__ == '__main__':
    ###########
    print('Mock candidate selection...')
    mock_sel = mock_selection()
    # selecting 130 random objects from the selection
    N_plot_mock = 130
    N_good_qso = sum(mock_sel['class'] == 'LAE_QSO')
    sel_ratios = [
        1,
        sum(mock_sel['class'] == 'CONT_QSO') / N_good_qso,
        sum(mock_sel['class'] == 'GAL') / N_good_qso,
        sum(mock_sel['class'] == 'SFG') / N_good_qso,
    ]
    real_ratios = [selr / mockr for selr, mockr in zip(sel_ratios, mock_sel['prior_ratios'])]
    N_plot_each_class = np.array(real_ratios) / sum(real_ratios) * N_plot_mock
    N_plot_each_class = N_plot_each_class.astype(int)

    # Array of indices of the selected sources to plot in the mock
    id_arr_to_plot = []
    for k, cl in enumerate(['LAE_QSO', 'CONT_QSO', 'GAL', 'SFG']):
        id_arr_to_plot.append(
            np.random.choice(
                np.where(mock_sel['class'] == cl)[0],
                N_plot_each_class[k], replace=False,
            )
        )
    id_arr_to_plot = np.concatenate(id_arr_to_plot)
    ###########

    ###########
    print('Loading miniJPAS&J-NEP candidate selection...')
    selection = pd.read_csv('csv/selection.csv')
    pm_flx, pm_err = load_minijpas_jnep(selection=True)[:2]
    cont_est_lya, cont_err_lya = estimate_continuum(pm_flx, pm_err, IGM_T_correct=False)
    ###########

    # Dir to save the images
    dirname = '/home/alberto/almacen/Selected_LAEs/obs_mock_vi'
    os.makedirs(dirname, exist_ok=True)

    N_total_to_plot = len(id_arr_to_plot) + len(selection)
    shuffle_indices = np.random.permutation(N_total_to_plot)

    # Counter of sources plotted
    iii = 0

    # Plot mock sources
    for src_mock in id_arr_to_plot:
        text_str = (
            f'z = {mock_sel["z_NB"][src_mock]:0.2f}'
            f'\nr = {flux_to_mag(mock_sel["flx"][-2, src_mock], w_central[-2])[0]:0.2f}'
            f'\nL_Lya = {mock_sel["L_lya"][src_mock]:0.2f}'
            f'\nEW_Lya = {mock_sel["EW_lya"][src_mock]:0.2f}'
        )

        args = (mock_sel['flx'][:, src_mock], mock_sel['flx_err'][:, src_mock],
                mock_sel['cont_est'][:, src_mock], mock_sel['cont_err'][:, src_mock],
                0, None, None, mock_sel['nb_sel'][src_mock],
                mock_sel['other_lines'][src_mock], shuffle_indices[iii],
                dirname, mock_sel['z_NB'][src_mock],
                None, None, text_str)
        plot_paper(*args)

        iii += 1

    # Plot real sources
    N_src_mjj = len(selection)
    for jjj in range(N_src_mjj):
        src = selection['src'][jjj]
        other_lines = selection['other_lines'][jjj]
        oth_raw_list = other_lines[1:-1].split()
        if len(oth_raw_list) == 0:
            oth_list = []
        else:
            oth_list = [int(item[:-1]) for item in oth_raw_list[:-1]] + [int(oth_raw_list[-1])]

        text_str = (
            f'z = {z_NB(selection["nb_sel"][jjj])[0]:0.2f}'
            f'\nr = {selection["r"][jjj]:0.2f}'
            f'\nL_Lya = {selection["L_lya"][jjj]:0.2f}'
            f'\nEW_Lya = {mock_sel["EW_lya"][jjj]:0.2f}'
        )

        args = (pm_flx[:, src], pm_err[:, src],
                cont_est_lya[:, src], cont_err_lya[:, src],
                0, None, None, selection['nb_sel'][jjj], oth_list,
                shuffle_indices[iii], dirname, z_NB(selection['nb_sel'][jjj])[0],
                None, None, text_str)
        plot_paper(*args)

        iii += 1

    # Save stuff to reproduce results
    np.save(f'{dirname}/shuffle_indices', shuffle_indices)
    np.save(f'{dirname}/id_arr_to_plot', id_arr_to_plot)
    with open(f'{dirname}/mock_sel_dict.pkl', 'wb') as f:
        pickle.dump(mock_sel, f)