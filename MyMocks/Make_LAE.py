import numpy as np
from my_utilities import *
import csv
import pandas as pd
from scipy.integrate import simpson
from time import time
import os

t0 = time()

####    Line wavelengths
w_lya = 1215.67

####    Mock parameters. MUST BE THE SAME AS IN 'Make_OII.py'   ####
z_lya = [2, 5]
obs_area = 1 # deg**2

# Wavelength array where to evaluate the spectrum

w_min  = 2500   # Minimum wavelength
w_max  = 10000  # Maximum wavelegnth
N_bins = 10000  # Number of bins

w_Arr = np.linspace(w_min, w_max, N_bins)

####    Specific LAE parameters
w_in  = [5, 5.1] # Line width interval
s_in = [-31., -30.] # Logarithmic uncertainty in flux density # 
L_in = [41.75, 43]
LINE = 'Lya'

####    Load LAE LF

filepath = '../csv/Konno_LF.csv' # From Konno et al. 2016
LAE_LF = []
with open(filepath, mode='r') as csvfile:
    rdlns = csv.reader(csvfile, delimiter=',')
    for line in rdlns:
        LAE_LF.append(line)
LAE_LF = np.array(LAE_LF).astype(float)

####    Compute the number of sources and L_line distribution 

Volume_LAE = z_volume(z_lya[0], z_lya[1], obs_area)
LF_p_cum_x = np.linspace(L_in[0], L_in[1], 1000)
N_sources_LAE = int(
    simpson(
        np.interp(LF_p_cum_x, LAE_LF[:, 0], LAE_LF[:, 1]), LF_p_cum_x
    ) * Volume_LAE
)
print(f'N_sources = {N_sources_LAE}')
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


AGE_Arr = np.zeros(N_sources_LAE)
MET_Arr = np.zeros(N_sources_LAE)
EXT_Arr = np.zeros(N_sources_LAE)
n_steps_mcmc = 32000
n, r = divmod(N_sources_LAE, n_steps_mcmc)
for k in range(n):
    idx_slice = slice(k * n_steps_mcmc, (k+1) * n_steps_mcmc)
    AGE_Arr[idx_slice] = 10 ** mcmc['chains'][-n_steps_mcmc:, 0]
    MET_Arr[idx_slice] = mcmc['chains'][-n_steps_mcmc:, 1]
    EXT_Arr[idx_slice] = mcmc['chains'][-n_steps_mcmc:, 2]
AGE_Arr[-r:] = 10 ** mcmc['chains'][-r:, 0]
MET_Arr[-r:] = mcmc['chains'][-r:, 1]
EXT_Arr[-r:] = mcmc['chains'][-r:, 2]

#### Let's load the data of the gSDSS filter
gSDSS_lambda_Arr_f, gSDSS_Transmission_Arr_f = Load_Filter('gSDSS')
gSDSS_lambda_pivot, gSDSS_FWHM = FWHM_lambda_pivot_filter('gSDSS')

gSDSS_data = {}

gSDSS_data['lambda_Arr_f'      ] = np.copy(gSDSS_lambda_Arr_f      )
gSDSS_data['Transmission_Arr_f'] = np.copy(gSDSS_Transmission_Arr_f)
gSDSS_data['lambda_pivot'      ] = np.copy(gSDSS_lambda_pivot      )
gSDSS_data['FWHM'              ] = np.copy(gSDSS_FWHM              )

####################################################################

filename = f'LAE_{obs_area}deg_z{z_lya[0]}-{z_lya[1]}'

os.mkdir(filename)

SED_file = open(filename + '/SEDs.csv', 'w')
SED_no_line_file = open(filename + '/SEDs_no_line.csv', 'w')

SED_writer = csv.writer(SED_file)
SED_no_line_writer = csv.writer(SED_no_line_file)

tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

pm_SEDs = np.zeros((60, N_sources_LAE))
pm_SEDs_no_line = np.copy(pm_SEDs)

w_Arr_reduced = np.interp(
    np.linspace(0, len(w_Arr), 1000), np.arange(len(w_Arr)), w_Arr
)

err_fit_params = np.load('../npy/err_fit_params_minijpas.npy')

z_out_Arr = []
EW_out_Arr = []

for i in range(N_sources_LAE):
    print('Generating spectrum {}/{}'.format(i+1, N_sources_LAE), end='\r')

    my_z = z_Arr[i]
    my_e = e_Arr[i]
    my_g = g_Arr[i]
    my_width = widths_Arr[i]
    my_s = s_Arr[i]
    my_MET = MET_Arr[i]
    my_AGE = AGE_Arr[i]
    my_EXT = EXT_Arr[i]

    SEDs, _, SEDs_no_line\
            = generate_spectrum(
            LINE, my_z, my_e, my_g,
            my_width, my_s, my_MET,
            my_AGE, my_EXT, w_Arr, Grid_Dictionary,
            Noise_w_Arr, Noise_Arr, T_A, T_B,
            gSDSS_data
            )
    pm_SEDs[:, i] = JPAS_synth_phot(SEDs, w_Arr, tcurves)
    pm_SEDs_no_line[:, i] = JPAS_synth_phot(SEDs_no_line, w_Arr, tcurves)

    SED_writer.writerow(np.interp(w_Arr_reduced, w_Arr, SEDs))
    SED_no_line_writer.writerow(np.interp(w_Arr_reduced, w_Arr, SEDs_no_line))

    EW_out_Arr.append(my_e)
    z_out_Arr.append(my_z)

# Add errors
m = err_fit_params[:, 0].reshape(-1, 1)
b = err_fit_params[:, 1].reshape(-1, 1)
pm_SEDs_err = pm_SEDs * 10 ** (b + m * np.log10(np.abs(pm_SEDs)))

detec_lim = np.vstack(
    (
        pd.read_csv('csv/5sigma_depths_NB.csv', header=None),
        pd.read_csv('csv/5sigma_depths_BB.csv', header=None),
    )
)
detec_lim = mag_to_flux(detec_lim[:, 1], detec_lim[:, 0]).reshape(-1, 1)

lim_flx = np.ones(pm_SEDs.shape) * detec_lim
err_lim = lim_flx * 10 ** (b + m * np.log10(np.abs(lim_flx)))
where_low_flx = np.where(pm_SEDs < detec_lim)
pm_SEDs_err[where_low_flx] = err_lim[where_low_flx]

np.save(filename + '/w_Arr.npy', w_Arr_reduced)

hdr = tcurves['tag'] + [s + '_e' for s in tcurves['tag']] + ['z', 'EW0', 'L_lya']

pd.DataFrame(
    data=np.hstack((pm_SEDs.T, pm_SEDs_err.T, np.array(z_out_Arr).reshape(-1, 1),
    np.array(EW_out_Arr).reshape(-1, 1), L_Arr.reshape(-1, 1)))
).to_csv(filename + '/data.csv', header=hdr)

SED_file.close()
SED_no_line_file.close()

print()
m, s = divmod(int(time() - t0), 60)
print('Elapsed: {}m {}s'.format(m, s))