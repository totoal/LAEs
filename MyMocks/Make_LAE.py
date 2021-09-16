import numpy as np
from astropy.cosmology import Planck18 as cosmo
from my_utilities import *
import csv

####    Line wavelengths
w_lya = 1215.67

####    Mock parameters. MUST BE THE SAME AS IN 'Make_OII.py'   ####
z_lya = [2.25, 3.2] # LAE z interval
obs_area = 10 # deg**2

# Wavelength array where to evaluate the spectrum

w_min  = 2500   # Minimum wavelength
w_max  = 10000  # Maximum wavelegnth
N_bins = 50000  # Number of bins

w_Arr = np.linspace( w_min , w_max , N_bins )

####    Specific LAE parameters
ew_in = [30, 100] # Rest frame EW interval 
w_in  = [5, 5.1] # Line width interval
s_in = [ -31.  , -30.] # Logarithmic uncertainty in flux density # 
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

bin_width = LAE_LF[1,0] - LAE_LF[0,0]
Volume_LAE = z_volume(z_lya[0], z_lya[1], obs_area)
N_sources_LAE = int(np.sum(LAE_LF[:,1]) * bin_width * Volume_LAE)
LF_p_cum = np.cumsum(LAE_LF[:,1])
LF_p_cum /= np.max(LF_p_cum)
L_Arr = np.interp(np.random.rand(N_sources_LAE), LF_p_cum, LAE_LF[:,0])

# Define EW and z Arrays
z_Arr = np.random.rand(N_sources_LAE) * (z_lya[1] - z_lya[0]) + z_lya[0]
e_Arr = np.random.rand(N_sources_LAE) * (ew_in[1] - ew_in[0]) + ew_in[0]
widths_Arr = np.random.rand(N_sources_LAE) * (w_in[1] - w_in[0]) + w_in[0]
s_Arr = 10**(np.random.rand(N_sources_LAE) * (s_in[1] - s_in[0]) + s_in[0])

# Define g flux array
g_Arr = L_flux_to_g(L_Arr, z_Arr, e_Arr)

# Dependece of noise with wavelength
Noise_w_Arr = np.linspace( 3000 , 9000 , 10 )
Noise_Arr   = np.ones( len( Noise_w_Arr ) ) # Now it is flat.

# Intergalactic medium mean absortion parameters : (From Faucher et al)
T_A = -0.001845
T_B =  3.924

#### Grid dictionary load
Grid_Dictionary = Load_BC03_grid_data()

#### AGE, MET and EXT parameters
DD = 0.00001
'''
MIN_MET = np.amin(Grid_Dictionary['met_Arr']) # Minimum metallicity
MAX_MET = np.amax(Grid_Dictionary['met_Arr']) # Maximum metallicity

MAX_AGE = cosmo.age(z_Arr).value              # Maximum Age
MIN_AGE = np.amin(Grid_Dictionary['age_Arr']) # Minimum Age

MIN_EXT = np.amin(Grid_Dictionary['ext_Arr']) # Minimum extintion
MAX_EXT = 0.05                                # Maximum extintion
'''
# Defining parameters according to HETDEX spectra fits (Provided by Sid 16/09/21)
MIN_MET = 26.850313
MAX_MET = 35.3161076
SIGMA_MET = (MAX_MET - MIN_MET) * 0.5
MU_MET = (MAX_MET + MIN_MET) * 0.5

MIN_AGE = 10 ** -2.31549077
MAX_AGE = 10 ** -1.94981165
SIGMA_AGE = (MAX_AGE - MIN_AGE) * 0.5
MU_AGE = (MAX_AGE + MIN_AGE) * 0.5

MIN_EXT = 5.43513275e-2
MAX_EXT = 0.8218874
SIGMA_EXT = (MAX_EXT - MIN_EXT) * 0.5
MU_EXT = (MAX_EXT + MIN_EXT) * 0.5
#####

MET_Arr = np.random.randn(N_sources_LAE) * SIGMA_MET + MU_MET
AGE_Arr = np.random.randn(N_sources_LAE) * SIGMA_AGE + MU_AGE
EXT_Arr = np.random.randn(N_sources_LAE) * SIGMA_EXT + MU_EXT

#### Let's load the data of the gSDSS filter
gSDSS_lambda_Arr_f, gSDSS_Transmission_Arr_f = Load_Filter('gSDSS')
gSDSS_lambda_pivot, gSDSS_FWHM = FWHM_lambda_pivot_filter('gSDSS')

gSDSS_data = {}

gSDSS_data[ 'lambda_Arr_f'       ] = np.copy( gSDSS_lambda_Arr_f       )
gSDSS_data[ 'Transmission_Arr_f' ] = np.copy( gSDSS_Transmission_Arr_f )
gSDSS_data[ 'lambda_pivot'       ] = np.copy( gSDSS_lambda_pivot       )
gSDSS_data[ 'FWHM'               ] = np.copy( gSDSS_FWHM               )

# Initialize cat
cat = {}
cat['SEDs'] = np.zeros((N_sources_LAE, len(w_Arr)))
cat['SEDs_no_IGM'] = np.zeros((N_sources_LAE, len(w_Arr)))
cat['SEDs_no_line'] = np.zeros((N_sources_LAE, len(w_Arr)))
cat['w_Arr'] = w_Arr
cat['LAE'] = np.ones(N_sources_LAE, dtype=bool)
cat['EW_Arr'] = e_Arr
cat['redshift_Lya_Arr'] = z_Arr
cat['AGE'] = np.zeros(N_sources_LAE)
cat['MET'] = np.zeros(N_sources_LAE)
cat['EXT'] = np.zeros(N_sources_LAE)

for i in range(N_sources_LAE):

    print('{}/{}'.format(i+1, N_sources_LAE))

    my_z = z_Arr[i]
    my_e = e_Arr[i]
    my_g = g_Arr[i]
    my_width = widths_Arr[i]
    my_s = s_Arr[i]
    my_MET = MET_Arr[i]
    my_AGE = AGE_Arr[i]
    my_EXT = EXT_Arr[i]

    cat['SEDs'][i,:], cat['SEDs_no_IGM'][i,:], cat['SEDs_no_line'][i,:]\
            = generate_spectrum(
            LINE, my_z, my_e, my_g,
            my_width, my_s, my_MET,
            my_AGE, my_EXT, w_Arr, Grid_Dictionary,
            Noise_w_Arr, Noise_Arr, T_A, T_B,
            gSDSS_data
            )
    cat['AGE'][i] = my_AGE
    cat['MET'][i] = my_MET
    cat['EXT'][i] = my_EXT

filename = 'LAE_1deg'
np.save('Source_cat_' + filename + '.npy', cat)
