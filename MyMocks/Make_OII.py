import numpy as np
from astropy.cosmology import Planck18 as cosmo
from my_utilities import *
import csv
import os
from time import time

####    Line wavelengths
w_lya = 1215.67
w_oii = 0.5 * ( 3727.092 + 3729.875 )

####    Mock parameters. MUST BE THE SAME AS IN 'Make_LAE.py'   ####
z_lya = [2.5, 3.2] # LAE z interval
z_oii = [ w_lya*(z + 1)/w_oii - 1 for z in z_lya] # OII z interval computed from z_lya
obs_area = 1 # deg**2

filename = 'OII_' + str(obs_area) + 'deg_z' + str(z_lya[0]) + '-' + str(z_lya[1])
os.mkdir(filename)

# Wavelength array where to evaluate the spectrum

w_min  = 2500   # Minimum wavelength
w_max  = 10000  # Maximum wavelgnth
N_bins = 10000  # Number of bins

w_Arr = np.linspace( w_min , w_max , N_bins )

####    Specific OII parameters
ew_in = [5, 100] # Rest frame EW interval 
w_in  = [5, 5.1] # Line width interval
s_in = [-31., -30.] # Logarithmic uncertainty in flux density # 
LINE = 'OII'
OII_MET = 36.97118006667763
OII_AGE = 2.8748148018996997
OII_EXT = 6.913769672799271e-14

####################################################################

#####   Load OII LF (Gilbank 2010)

filepath = '../csv/Gilbank2010-OII_LF.csv'
OII_LF = []
with open(filepath, mode='r') as csvfile:
    rdlns = csv.reader(csvfile, delimiter=',')
    
    for line in rdlns:
        OII_LF.append(line)
OII_LF = np.array(OII_LF).astype(float)

####    Compute the number of sources and L_line distribution 
bin_width = OII_LF[1, 0] - OII_LF[0, 0]
Volume_OII = z_volume(z_oii[0], z_oii[1], obs_area)
N_sources_OII = int(np.sum(OII_LF[:, 1]) * bin_width * Volume_OII)
LF_p_cum = np.cumsum(OII_LF[:, 1])
LF_p_cum /= np.max(LF_p_cum)
L_Arr = np.interp(np.random.rand(N_sources_OII), LF_p_cum, OII_LF[:, 0])

# Define EW and z Arrays
z_Arr = np.random.rand(N_sources_OII) * (z_lya[1] - z_lya[0]) + z_lya[0]
z_Arr_OII = w_lya * (z_Arr + 1) / w_oii - 1
# e_Arr = np.random.rand(N_sources_OII) * (ew_in[1] - ew_in[0]) + ew_in[0]
widths_Arr = np.random.rand(N_sources_OII) * (w_in[1] - w_in[0]) + w_in[0]
s_Arr = 10 ** (np.random.rand(N_sources_OII) * (s_in[1] - s_in[0]) + s_in[0])

# Define g flux array
# g_Arr = L_flux_to_g(L_Arr, z_Arr_OII, e_Arr)

e_Arr = np.random.rand(N_sources_OII) * (ew_in[1] - ew_in[0]) + ew_in[0]
g_Arr = 10 ** (np.random.rand(N_sources_OII) * (-17 - -18) + -18)

# Dependece of noise with wavelength
Noise_w_Arr = np.linspace(3000, 9000, 10)
Noise_Arr   = np.ones(len(Noise_w_Arr)) # Now it is flat.

# AGE, MET & EXT Arrs
MET_Arr = [OII_MET] * N_sources_OII
AGE_Arr = [OII_AGE] * N_sources_OII
EXT_Arr = [OII_EXT] * N_sources_OII

# Intergalactic medium mean absortion parameters: (From Faucher et al)
T_A = -0.001845
T_B =  3.924

#### Grid dictionary load
Grid_Dictionary = Load_BC03_grid_data()

# Let's load the data of the gSDSS filter
gSDSS_lambda_Arr_f , gSDSS_Transmission_Arr_f = Load_Filter('gSDSS')
gSDSS_lambda_pivot , gSDSS_FWHM = FWHM_lambda_pivot_filter('gSDSS')

gSDSS_data = {}

gSDSS_data['lambda_Arr_f'      ] = np.copy(gSDSS_lambda_Arr_f      )
gSDSS_data['Transmission_Arr_f'] = np.copy(gSDSS_Transmission_Arr_f)
gSDSS_data['lambda_pivot'      ] = np.copy(gSDSS_lambda_pivot      )
gSDSS_data['FWHM'              ] = np.copy(gSDSS_FWHM              )

##########################################################################
SED_file = open(filename + '/SEDs.csv', 'w')
SED_no_line_file = open(filename + '/SEDs_no_line.csv', 'w')

SED_writer = csv.writer(SED_file)
SED_no_line_writer = csv.writer(SED_no_line_file)

tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

pm_SEDs = np.zeros((60, N_sources_OII))
pm_SEDs_no_line = np.copy(pm_SEDs)

w_Arr_reduced = np.interp(
    np.linspace(0, len(w_Arr), 1000), np.arange(len(w_Arr)), w_Arr
)

err_fit_params = np.load('../npy/err_fit_params_minijpas.npy')

z_out_Arr = []
EW_out_Arr = []

t0 = time()

for i in range(N_sources_OII):
    print('{}/{}'.format(i+1, N_sources_OII), end='\r')

    my_z = z_Arr_OII[i]
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
    z_out_Arr.append(z_Arr[i])
#Add errors
m = err_fit_params[:, 0].reshape(-1, 1)
b = err_fit_params[:, 1].reshape(-1, 1)
pm_SEDs *= 1 + 10 ** (b + m * np.log10(np.abs(pm_SEDs)))\
    * np.random.randn(pm_SEDs.shape[0], pm_SEDs.shape[1])

utils = {
    'z_Arr': np.array(z_out_Arr),
    'w_Arr': w_Arr_reduced,
    'EW_Arr': np.array(EW_out_Arr)
}

np.save(filename + '/pm_SEDs.npy', pm_SEDs)
np.save(filename + '/pm_SEDs_no_line.npy', pm_SEDs_no_line)
np.save(filename + '/utils.npy', utils)

print()
m, s = divmod(int(time() - t0), 60)
print('Elapsed: {}m {}s'.format(m, s))
