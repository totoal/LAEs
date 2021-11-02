import numpy as np
from astropy.cosmology import Planck18 as cosmo
from my_utilities import *
import csv
from scipy.integrate import simpson
from time import time
import os
import pandas as pd

####    Line wavelengths
w_lya = 1215.67

####    Mock parameters. MUST BE THE SAME AS IN 'Make_OII.py'   ####
# z_lya = [3.05619946, 3.17876562] # LAE z interval
z_lya = [2.5, 3.2]
obs_area = 400 # deg**2

filename = 'LAE_' + str(obs_area) + 'deg_z' + str(z_lya[0]) + '-' + str(z_lya[1])

# Wavelength array where to evaluate the spectrum

w_min  = 2500   # Minimum wavelength
w_max  = 10000  # Maximum wavelegnth
N_bins = 10000  # Number of bins

w_Arr = np.linspace(w_min , w_max , N_bins)

####    Specific LAE parameters
w_in  = [5, 5.1] # Line width interval
s_in = [-31., -30.] # Logarithmic uncertainty in flux density # 
LINE = 'Lya'

####################################################################

#####   Load LAE LF

filepath = '../csv/HETDEX_LumFunc.csv'
LAE_LF = []
with open(filepath, mode='r') as csvfile:
    rdlns = csv.reader(csvfile, delimiter=',')
    for line in rdlns:
        LAE_LF.append(line)
LAE_LF = np.array(LAE_LF).astype(float)

####    Compute the number of sources and L_line distribution 

Volume_LAE = z_volume(z_lya[0], z_lya[1], obs_area)
N_sources_LAE = int(simpson(LAE_LF[:,1], LAE_LF[:,0], dx=0.1) * Volume_LAE)
LF_p_cum_x = np.linspace(LAE_LF[0,0], LAE_LF[-1,0], 1000)
LF_p_cum = np.cumsum(np.interp(
    LF_p_cum_x, LAE_LF[:,0], LAE_LF[:,1])
)
LF_p_cum /= np.max(LF_p_cum)
L_Arr = np.interp(np.random.rand(N_sources_LAE), LF_p_cum, LF_p_cum_x)

# Define z, widths and s Array
z_Arr = np.random.rand(N_sources_LAE) * (z_lya[1] - z_lya[0]) + z_lya[0]
widths_Arr = np.random.rand(N_sources_LAE) * (w_in[1] - w_in[0]) + w_in[0]
s_Arr = 10**(np.random.rand(N_sources_LAE) * (s_in[1] - s_in[0]) + s_in[0])

# Define EW arr
ew_x = np.linspace(10, 500, 10000)
w_0 = 75
ew_dist_cum = np.cumsum(np.exp(-ew_x / w_0))
ew_dist_cum /= np.max(ew_dist_cum)
e_Arr = np.interp(np.random.rand(N_sources_LAE), ew_dist_cum, ew_x)

# Define g flux array
g_Arr = L_flux_to_g(L_Arr, z_Arr, e_Arr)
'''
g_Arr = 10 ** (np.random.rand(N_sources_LAE) * (-17 - -17.9) + -17.9)
e_Arr = np.random.rand(N_sources_LAE) * (150 - 10) + 10
'''

# Dependece of noise with wavelength
Noise_w_Arr = np.linspace(3000, 9000, 10)
Noise_Arr   = np.ones(len(Noise_w_Arr)) # Now it is flat.

# Intergalactic medium mean absortion parameters : (From Faucher et al)
T_A = -0.001845
T_B =  3.924

#### Grid dictionary load
Grid_Dictionary = Load_BC03_grid_data()

#### AGE, MET and EXT parameters
mcmc = np.load('./mcmc_chains/mcmc_chains_Nw_800_Nd_4_Ns_'
               '400_Nb_100_z_1.9_3.0_sn_7.0_g_23.5_p_0.9_pp_50.0.npy',
                allow_pickle=True).item()

AGE_Arr = 10 ** mcmc['chains'][-N_sources_LAE:, 0]
MET_Arr = mcmc['chains'][-N_sources_LAE:, 1]
EXT_Arr = mcmc['chains'][-N_sources_LAE:, 2]

#### Let's load the data of the gSDSS filter
gSDSS_lambda_Arr_f, gSDSS_Transmission_Arr_f = Load_Filter('gSDSS')
gSDSS_lambda_pivot, gSDSS_FWHM = FWHM_lambda_pivot_filter('gSDSS')

gSDSS_data = {}

gSDSS_data['lambda_Arr_f'      ] = np.copy(gSDSS_lambda_Arr_f      )
gSDSS_data['Transmission_Arr_f'] = np.copy(gSDSS_Transmission_Arr_f)
gSDSS_data['lambda_pivot'      ] = np.copy(gSDSS_lambda_pivot      )
gSDSS_data['FWHM'              ] = np.copy(gSDSS_FWHM              )

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

t0 = time()

for i in range(N_sources_LAE):
    print('{}/{}'.format(i+1, N_sources_LAE), end='\r')

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

#Add errors
m = err_fit_params[:, 0].reshape(-1, 1)
b = err_fit_params[:, 1].reshape(-1, 1)
pm_SEDs_err = pm_SEDs * 10 ** (b + m * np.log10(np.abs(pm_SEDs)))

lim_flx = (np.ones(pm_SEDs.shape) * 1e-19)
err_lim = lim_flx * 10 ** (b + m * np.log10(np.abs(lim_flx)))
where_low_flx = np.where(pm_SEDs < 1e-19)
pm_SEDs_err[where_low_flx] = err_lim[where_low_flx]

pm_SEDs += pm_SEDs_err * np.random.randn(pm_SEDs.shape[0], pm_SEDs.shape[1])

utils = {
    'z_Arr': np.array(z_out_Arr),
    'w_Arr': w_Arr_reduced,
    'EW_Arr': np.array(EW_out_Arr)
}

np.save(filename + '/pm_flx.npy', pm_SEDs)
np.save(filename + '/pm_flx_err.npy', pm_SEDs_err)
np.save(filename + '/pm_flx_no_line_no_err.npy', pm_SEDs_no_line)
np.save(filename + '/utils.npy', utils)

print()
m, s = divmod(int(time() - t0), 60)
print('Elapsed: {}m {}s'.format(m, s))
