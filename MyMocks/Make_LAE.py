#!/home/alberto/miniconda3/bin/python3

import numpy as np

from my_utilities import *

import csv
import pandas as pd

from scipy.integrate import simpson

from time import time
import os
import sys

def main(part):
    t0 = time()

    ####    Line wavelengths
    # w_lya = 1215.67

    ####    Mock parameters.
    z_lya = [2, 4]
    obs_area = 10 # deg**2

    # Wavelength array where to evaluate the spectrum

    w_min  = 2500   # Minimum wavelength
    w_max  = 10000  # Maximum wavelegnth
    N_bins = 10000  # Number of bins

    w_Arr = np.linspace(w_min, w_max, N_bins)

    ####    Specific LAE parameters
    w_in  = [5, 5.1] # Line width interval
    s_in = [-31., -30.] # Logarithmic uncertainty in flux density # 
    L_in = [41.75, 45]
    LINE = 'Lya'

    ####    Load LAE LF (Sobral 2016)

    phistar = 10 ** -3.45
    Lstar = 10 ** 42.93
    alpha = -1.93

    LAE_LF = np.empty((1000, 2))
    Lx = np.linspace(10 ** L_in[0], 10 ** L_in[1], 1000)

    LAE_LF[:, 1] = schechter(Lx, phistar, Lstar, alpha) * Lx * np.log(10)
    LAE_LF[:, 0] = np.log10(Lx)

    ####    Compute the number of sources and L_line distribution 

    Volume_LAE = z_volume(z_lya[0], z_lya[1], obs_area)
    LF_p_cum_x = np.linspace(L_in[0], L_in[1], 1000)
    N_sources_LAE = int(
        simpson(
            np.interp(LF_p_cum_x, LAE_LF[:, 0], LAE_LF[:, 1]), LF_p_cum_x
        ) * Volume_LAE
    )
    LF_p_cum = np.cumsum(np.interp(
        LF_p_cum_x, LAE_LF[:,0], LAE_LF[:,1])
    )
    LF_p_cum /= np.max(LF_p_cum)
    L_Arr = np.interp(np.random.rand(N_sources_LAE), LF_p_cum, LF_p_cum_x)

    ## Define z, widths and s Array

    z_Arr = np.random.rand(N_sources_LAE) * (z_lya[1] - z_lya[0]) + z_lya[0]
    widths_Arr = np.random.rand(N_sources_LAE) * (w_in[1] - w_in[0]) + w_in[0]
    s_Arr = 10**(np.random.rand(N_sources_LAE) * (s_in[1] - s_in[0]) + s_in[0])

    # Define EW arr
    ew_x = np.linspace(10, 500, 10000)
    w_0 = 75
    ew_dist_cum = np.cumsum(np.exp(-ew_x / w_0))
    ew_dist_cum /= np.max(ew_dist_cum)
    e_Arr = np.interp(np.random.rand(N_sources_LAE), ew_dist_cum, ew_x)

    # Dependece of noise with wavelength
    Noise_w_Arr = np.linspace(3000, 9000, 10)
    Noise_Arr   = np.ones(len(Noise_w_Arr)) # Now it is flat.

    # Compute g_Arr
    g_Arr = L_flux_to_g(L_Arr, z_Arr, e_Arr)

    # Intergalactic medium mean absortion parameters: (From Faucher et al)
    T_A = -0.001845
    T_B =  3.924

    #### Grid dictionary load
    Grid_Dictionary = Load_BC03_grid_data()

    #### AGE, MET and EXT parameters
    mcmc = np.load('./mcmc_chains/mcmc_chains_Nw_800_Nd_4_Ns_'
                '400_Nb_100_z_1.9_3.0_sn_7.0_g_23.5_p_0.9_pp_50.0.npy',
                    allow_pickle=True).item()

    #### Let's load the data of the gSDSS filter
    gSDSS_lambda_Arr_f, gSDSS_Transmission_Arr_f = Load_Filter('gSDSS')
    gSDSS_lambda_pivot, gSDSS_FWHM = FWHM_lambda_pivot_filter('gSDSS')

    gSDSS_data = {}

    gSDSS_data['lambda_Arr_f'      ] = np.copy(gSDSS_lambda_Arr_f      )
    gSDSS_data['Transmission_Arr_f'] = np.copy(gSDSS_Transmission_Arr_f)
    gSDSS_data['lambda_pivot'      ] = np.copy(gSDSS_lambda_pivot      )
    gSDSS_data['FWHM'              ] = np.copy(gSDSS_FWHM              )

    ####################################################################

    filename =\
        f'/home/alberto/cosmos/LAEs/MyMocks/LAE_{obs_area}deg_z{z_lya[0]}-{z_lya[1]}'

    if not os.path.exists(filename):
        os.mkdir(filename)

    SED_file = open(filename + f'/SEDs{part}.csv', 'w')
    SED_no_line_file = open(filename + f'/SEDs_no_line{part}.csv', 'w')

    SED_writer = csv.writer(SED_file)
    SED_no_line_writer = csv.writer(SED_no_line_file)

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()
    # define a different tcurves only with r and i
    tcurves_sampling = {}
    tcurves_sampling['tag'] = [tcurves['tag'][-3], tcurves['tag'][-1], tcurves['tag'][-2]]
    tcurves_sampling['w'] = [tcurves['w'][-3], tcurves['w'][-1], tcurves['w'][-2]]
    tcurves_sampling['t'] = [tcurves['t'][-3], tcurves['t'][-1], tcurves['t'][-2]]

    w_Arr_reduced = np.interp(
        np.linspace(0, len(w_Arr), 1000), np.arange(len(w_Arr)), w_Arr
    )

    err_fit_params = np.load('../npy/err_fit_params_minijpas.npy')

    z_out_Arr = []
    EW_out_Arr = []

    good = np.where(g_Arr > 2e-19)[0]
    N_good_sources = len(good)

    pm_SEDs = np.zeros((60, N_good_sources))
    pm_SEDs_no_line = np.copy(pm_SEDs)

    # Initialize mask for the second cut. Used later
    good2 = np.ones(good.shape).astype(bool)

    print(f'N_sources = {N_good_sources}\n')

    for j, i in enumerate(good):
        print('Generating spectrum {}/{}'.format(j+1, N_good_sources), end='\r')

        my_z = z_Arr[i]
        my_e = e_Arr[i]
        my_g = g_Arr[i]
        my_width = widths_Arr[i]
        my_s = s_Arr[i]

        ## Select AGE, MET, EXT so they don't produce a bad source, i. e.: i << g
        count = 0
        while True:
            count += 1
            chain_step = np.random.randint(0, 32000)

            my_AGE = 10 ** mcmc['chains'][-chain_step, 0]
            my_MET = mcmc['chains'][-chain_step, 1]
            my_EXT = mcmc['chains'][-chain_step, 2]
            SEDs, _, SEDs_no_line\
                    = generate_spectrum(
                    LINE, my_z, my_e, my_g,
                    my_width, my_s, my_MET,
                    my_AGE, my_EXT, w_Arr, Grid_Dictionary,
                    Noise_w_Arr, Noise_Arr, T_A, T_B,
                    gSDSS_data
                    )
            aux_pm = JPAS_synth_phot(SEDs_no_line, w_Arr, tcurves_sampling)

            if aux_pm[1] - aux_pm[0] < 0.5e-18:
                count = 0
                break
            if count == 10:
                break
            
        # mag r < 24 cut
        if aux_pm[2] < 6e-19:
            good2[j] = False
            continue

        pm_SEDs[:, j] = JPAS_synth_phot(SEDs, w_Arr, tcurves)
        pm_SEDs_no_line[:, j] = JPAS_synth_phot(SEDs_no_line, w_Arr, tcurves)

        SED_writer.writerow(np.interp(w_Arr_reduced, w_Arr, SEDs))
        SED_no_line_writer.writerow(np.interp(w_Arr_reduced, w_Arr, SEDs_no_line))

        EW_out_Arr.append(my_e)
        z_out_Arr.append(my_z)

    # Load limit mags
    detec_lim = np.vstack(
        (
            pd.read_csv('csv/5sigma_depths_NB.csv', header=None),
            pd.read_csv('csv/5sigma_depths_BB.csv', header=None)
        )
    )[:, 1].reshape(-1, 1)

    # Add errors
    a = err_fit_params[:, 0].reshape(-1, 1)
    b = err_fit_params[:, 1].reshape(-1, 1)
    c = err_fit_params[:, 2].reshape(-1, 1)
    expfit = lambda x: a * np.exp(b * x + c)

    w_central = central_wavelength().reshape(-1, 1)

    mags = flux_to_mag(pm_SEDs, w_central)
    mags[np.isnan(mags) | np.isinf(mags)] = 99.

    mag_err = expfit(mags)
    where_himag = np.where(mags > detec_lim)

    mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)
    mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

    pm_SEDs_err = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    # Perturb according to the error
    pm_SEDs += np.random.normal(size=mags.shape) * pm_SEDs_err

    # Now recompute the error
    # mags = flux_to_mag(pm_SEDs, w_central)
    # mags[np.isnan(mags) | np.isinf(mags) | (mags > 26)] = 99.
    # mag_err = expfit(mags)
    # where_himag = np.where(mags > detec_lim)

    # mag_err[where_himag] = expfit(detec_lim)[where_himag[0]].reshape(-1,)
    # mags[where_himag] = detec_lim[where_himag[0]].reshape(-1,)

    # pm_SEDs_err = mag_to_flux(mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    np.save(filename + '/w_Arr.npy', w_Arr_reduced)

    hdr = tcurves['tag'] + [s + '_e' for s in tcurves['tag']] + ['z', 'EW0', 'L_lya']

    pd.DataFrame(
        data=np.hstack((pm_SEDs.T[good2], pm_SEDs_err.T[good2],
        np.array(z_out_Arr).reshape(-1, 1),
        np.array(EW_out_Arr).reshape(-1, 1), L_Arr[good][good2].reshape(-1, 1)))
    ).to_csv(filename + f'/data{part}.csv', header=hdr)

    SED_file.close()
    SED_no_line_file.close()

    print()
    m, s = divmod(int(time() - t0), 60)
    print('Elapsed: {}m {}s'.format(m, s))

if __name__ == '__main__':
    main(sys.argv[1])