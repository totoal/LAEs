from minijpas_LF_and_puricomp import *
from my_functions import *

import time

import numpy as np

def retrieve_mock_LF(pm_flx, pm_err, ew0_cut, ew_oth, mag,
                     mag_min, mag_max, nb_min, nb_max):
    cont_est_lya, cont_err_lya, cont_est_other, cont_err_other =\
        nb_or_3fm_cont(pm_flx, pm_err, 'nb') 

    # Lya search
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut)
    lya_lines, lya_cont_lines, _ = identify_lines(
        line, pm_flx, cont_est_lya, first=True, return_line_width=True
    )
    lya_lines = np.array(lya_lines)

    # Other lines
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
        ew_oth, obs=True)
    other_lines = identify_lines(line_other, pm_flx, cont_est_other)

    N_sources = pm_flx.shape[1]

    mag_cut = (mag > mag_min) & (mag < mag_max)

    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    mask = (lya_lines >= nb_min) & (lya_lines <= nb_max) & mag_cut
    nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont_est_lya, z_Arr, mask=mask
    )
    
    ### Estimate Luminosity
    _, _, L_Arr, _, _, _ = EW_L_NB(
        pm_flx, pm_err, cont_est_lya, cont_err_lya, z_Arr, lya_lines, N_nb=0
    )

    L_Lbin_err = np.load('npy/L_nb_err.npy')
    median_L = np.load('npy/L_bias.npy')
    L_binning = np.load('npy/L_nb_err_binning.npy')
    L_bin_c = [L_binning[i : i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]

    # Correct L_Arr with the median
    L_Arr =  np.log10(10 ** L_Arr - np.interp(10 ** L_Arr, L_bin_c, median_L))

    # Apply bin err
    L_binning_position = binned_statistic(
            10 ** L_Arr, None, 'count', bins=L_binning
    ).binnumber
    L_binning_position[L_binning_position > len(L_binning) - 2] = len(L_binning) - 2
    L_e_Arr = L_Lbin_err[L_binning_position]

    L_bins = np.load('npy/puricomp2d_L_bins.npy')
    r_bins = np.load('npy/puricomp2d_r_bins.npy')
    puri2d_minijpas = np.load('npy/puri2d_minijpas.npy')
    comp2d_minijpas = np.load('npy/comp2d_minijpas.npy')
    bins = np.log10(L_binning)

    starprob = np.ones(mag.shape)
    tile_id = np.ones(mag.shape) * 2241

    LF_percentiles = LF_perturb_err(L_Arr, L_e_Arr, nice_lya, mag, z_Arr,
                                    starprob, bins, puri2d_minijpas,
                                    comp2d_minijpas, L_bins, r_bins,
                                    'minijpas', tile_id, [0])
    
    return LF_percentiles[1], bins

def LF_R_squared(qso_fraction, sf_fraction):
    minijpas_area = 0.895
    gal_area = 5.54
    bad_qso_area = 100
    good_qso_area = 200
    # sf_area would be = 100 too, not necessary to specify

    # the proportional factors are made in relation to bad_qso
    # so bad_qso_factor = 1
    gal_factor = bad_qso_area / gal_area
    good_qso_factor = bad_qso_area / good_qso_area

    # Load full mock
    survey_name = 'minijpas'
    train_or_test = 'train'

    name_qso = 'QSO_100000_0'
    name_qso_bad = f'QSO_double_{train_or_test}_{survey_name}_DR16_D_0'
    name_qso_hiL = ''
    name_gal = f'GAL_LC_{survey_name}_0'
    name_sf = f'LAE_12.5deg_z2-4.25_{train_or_test}_{survey_name}_0'

    add_errs = True
    print('Loading mock...')
    pm_flx, pm_err, zspec, EW_lya, L_lya, is_qso, is_sf, is_gal,\
        is_LAE, where_hiL, _ = ensemble_mock(name_qso, name_gal, name_sf,
                                             name_qso_bad, name_qso_hiL, add_errs,
                                             qso_fraction, sf_fraction)

    # Make 2D puricomp
    print('2D puricomp...')
    ew0lya = 30
    ewoth = 400
    mag = flux_to_mag(pm_flx[-2], w_central[-2])
    mag_min = 17
    mag_max = 24
    nb_min = 5
    nb_max = 20

    make_the_LF_params = mag_min, mag_max, nb_min, nb_max, ew0lya, ewoth, 'nb'
    all_corrections(make_the_LF_params, pm_flx, pm_err, zspec, EW_lya, L_lya, is_gal,
                    is_qso, is_sf, is_LAE, where_hiL, 'minijpas', hiL_factor,
                    good_qso_factor, gal_factor)
    
    # Retrieve mock LF
    print('Mock LF')
    mock_hist, b = retrieve_mock_LF(pm_flx, pm_err, ew0lya, ewoth, mag,
                                    mag_min, mag_max, nb_min, nb_max)

    # Make minijpas LF
    print('miniJPAS LF')
    minijpas_hist, b = make_the_LF(make_the_LF_params, ['minijpas'], True)

    bin_width = np.array([b[i + 1] - b[i] for i in range(len(b) - 1)])

    # z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    # z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1
    mock_vol = effective_volume(nb_min, nb_max, bad_qso_area)
    minijpas_vol = effective_volume(nb_min, nb_max, minijpas_area)

    mock_LF = mock_hist / bin_width / mock_vol
    minijpas_LF = minijpas_hist / bin_width / minijpas_vol

    where_not_zero = ((mock_LF > 0) & (minijpas_LF > 0))
    SS_res = ((np.log10((mock_LF / minijpas_LF)[where_not_zero])) ** 2).sum()
    SS_tot = ((np.log10(mock_LF[where_not_zero])
        - np.nanmean(np.log10(mock_LF[where_not_zero]))) ** 2).sum()
    R_squared = 1 - (SS_res / SS_tot)

    print(f'R^2 = {R_squared}')

    return mock_LF, minijpas_LF, R_squared

if __name__ == '__main__':
    t0 = time.time()

    frac_list = [1., 0.75, 0.5, 0.25]
    out_list = []

    for X in frac_list:
        for Y in frac_list:
            print(f'fracs = {X}, {Y}')
            out = LF_R_squared(X, Y)
            out_list.append(out)

    np.save('npy/method_val_out.npy', out_list)

    print('Elapsed: {0:0.0f} m {1:0.1f} s'.format(*divmod(time.time() - t0, 60)))