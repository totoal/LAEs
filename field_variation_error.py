import numpy as np
import pickle

import os

from LAEs.my_functions import central_wavelength, nb_fwhm
from LAEs.minijpas_LF_and_puricomp import effective_volume
from LAEs.plot_puricomp2d import load_puricomp1d

w_central = central_wavelength()
w_lya = 1215.67
nb_fwhm_Arr = nb_fwhm(np.arange(60))
L_binning = np.load('npy/L_nb_err_binning.npy')
b = np.log10(L_binning)
LF_bins = np.array([(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)])
bin_width = np.array([b[i + 1] - b[i] for i in range(len(b) - 1)])

survey_list_0 = np.array([f'minijpasAEGIS00{i}' for i in range(1, 4 + 1)] + ['jnep'])

def mask_puricomp(puri, comp, min_puri=0.0, min_comp=0.0):
    return (puri > min_puri) & (comp > min_comp)

min_N_bin = 0


def bootstrapped_LFs(nb1, nb2, survey_list_indices):
    '''
    Returns a matrix of N iterations of the LF using the fields given by
    survey_list_indices.
    '''
    survey_list = survey_list_0[survey_list_indices]

    this_hist = None

    how_many_jnep = sum(survey_list_indices == 4)
    how_many_minijpas = 5 - how_many_jnep

    vol_minijpas = effective_volume(nb1, nb2, 'minijpas')
    vol_jnep = effective_volume(nb1, nb2, 'jnep')
    this_volume = vol_minijpas / 4 * how_many_minijpas + vol_jnep * how_many_jnep

    for survey_name in survey_list:
        pathname = f'Luminosity_functions/LF_r17-24_nb{nb1}-{nb2}_ew30_ewoth100_nb_1.0'
        filename_hist = f'{pathname}/hist_i_mat_{survey_name}_boots.npy'
        hist_i_mat = np.load(filename_hist)

        this_field_LF = hist_i_mat / effective_volume(nb1, nb2, survey_name) / bin_width

        if this_hist is None:
            this_hist = hist_i_mat
            field_LF_mat = this_field_LF
        else:
            this_hist += hist_i_mat
            field_LF_mat = np.vstack([field_LF_mat, this_field_LF])

    return this_hist / bin_width / this_volume

def bootstrapped_combined_LF(survey_list_indices):
    survey_list = survey_list_0[survey_list_indices]

    comb_nbs_list = [[1, 5], [4, 8], [7, 11], [10, 14], [13, 17], [16, 20]]

    total_volume = 0.
    for [this_nb_min, this_nb_max] in comb_nbs_list:
        total_volume += effective_volume(this_nb_min, this_nb_max, 'both')
    masked_volume = None
    hist_mat = None
    LF_raw = None
    total_puri_list = np.load('npy/total_puri_list.npy')
    total_comp_list = np.load('npy/total_comp_list.npy')

    for i, [nb1, nb2] in enumerate(comb_nbs_list):
        pathname = f'Luminosity_functions/LF_r17-24_nb{nb1}-{nb2}_ew30_ewoth100_nb_1.0'
        puricomp_bins = load_puricomp1d(pathname)[-1]

        # Bin centers
        bc = [puricomp_bins[i: i + 2].sum() * 0.5 for i in range(len(puricomp_bins) - 1)]

        this_puri = np.interp(LF_bins, bc, total_puri_list[i])
        this_comp = np.interp(LF_bins, bc, total_comp_list[i])
        this_hist = None
        for survey_name in survey_list:
            filename_hist = f'{pathname}/hist_i_mat_{survey_name}.npy'
            hist_i_mat = np.load(filename_hist)

            this_field_LF = hist_i_mat / effective_volume(nb1, nb2, survey_name) / bin_width

            if this_hist is None:
                this_hist = hist_i_mat
                field_LF_mat = this_field_LF
            else:
                this_hist += hist_i_mat
                field_LF_mat = np.vstack([field_LF_mat, this_field_LF])

        this_hist = this_hist / total_volume / bin_width

        how_many_jnep = sum(survey_list_indices == 4)
        how_many_minijpas = 5 - how_many_jnep

        vol_minijpas = effective_volume(nb1, nb2, 'minijpas')
        vol_jnep = effective_volume(nb1, nb2, 'jnep')
        vol = vol_minijpas / 4 * how_many_minijpas + vol_jnep * how_many_jnep
        this_volume = np.ones_like(LF_bins) * vol

        filename_dict = f'{pathname}/LFs.pkl'
        with open(filename_dict, 'rb') as file:
            this_LF_raw = pickle.load(file)['LF_total_raw'] * this_volume
            if LF_raw is None:
                LF_raw = this_LF_raw
            else:
                LF_raw += this_LF_raw
            
        # Set masked bins by puricomp_mask to 0
        puricomp_mask = mask_puricomp(this_puri, this_comp) & (this_LF_raw >= min_N_bin)
        this_hist[:, ~puricomp_mask] = 0.
        this_volume[~puricomp_mask] = 0.

        if masked_volume is None:
            masked_volume = this_volume
        else:
            masked_volume += this_volume

        if hist_mat is None:
            hist_mat = this_hist
        else:
            hist_mat = hist_mat + this_hist

    hist_mat = hist_mat * total_volume / masked_volume

    return hist_mat

if __name__ == '__main__':
    nbs_list = [[1, 5], [4, 8], [7, 11], [10, 14], [13, 17], [16, 20]]

    N_realizations = 1000

    hist_mat = None
    print('Making combined LF')
    for iter_i in range(N_realizations):
        print(f'{iter_i + 1} / {N_realizations}', end='\r')
        boots = np.random.choice(np.arange(5), 5, replace=True)
        this_hist_mat = bootstrapped_combined_LF(boots)

        if hist_mat is None:
            hist_mat = this_hist_mat
        else:
            hist_mat = np.vstack([hist_mat, this_hist_mat])
    print('\n')
    L_LF_err_percentiles = np.percentile(hist_mat, [16, 50, 84], axis=0)
    LF_err_plus = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
    LF_err_minus = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]

    save_LF_name = '/home/alberto/cosmos/LAEs/Luminosity_functions/Total_LF'
    os.makedirs(save_LF_name, exist_ok=True)

    np.save(f'{save_LF_name}/LF_err_plus', LF_err_plus)
    np.save(f'{save_LF_name}/LF_err_minus', LF_err_minus)
    np.save(f'{save_LF_name}/median_LF_boots', L_LF_err_percentiles[1])
    np.save(f'{save_LF_name}/hist_mat', hist_mat)

    for [nb1, nb2] in nbs_list:
        print(f'NBs = {nb1}, {nb2}')
        pathname = f'Luminosity_functions/LF_r17-24_nb{nb1}-{nb2}_ew30_ewoth100_nb_1.0'
        hist_mat = None
        for iter_i in range(N_realizations):
            print(f'{iter_i + 1} / {N_realizations}', end='\r')

            boots = np.random.choice(np.arange(5), 5, replace=True)

            this_hist_mat = bootstrapped_LFs(nb1, nb2, boots)

            if hist_mat is None:
                hist_mat = this_hist_mat
            else:
                hist_mat = np.vstack([hist_mat, this_hist_mat])
        print('\n')

        L_LF_err_percentiles = np.percentile(hist_mat, [16, 50, 84], axis=0)
        LF_err_plus = L_LF_err_percentiles[2] - L_LF_err_percentiles[1]
        LF_err_minus = L_LF_err_percentiles[1] - L_LF_err_percentiles[0]

        np.save(f'{pathname}/LF_err_plus', LF_err_plus)
        np.save(f'{pathname}/LF_err_minus', LF_err_minus)
        np.save(f'{pathname}/median_LF_boots', L_LF_err_percentiles[1])