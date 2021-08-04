import numpy as np
from astropy.cosmology import Planck18 as cosmo
from my_utilities import *
import csv

####    Line wavelengths
w_lya = 1215.67
w_oii = 0.5 * ( 3727.092 + 3729.875 )

####    Mock parameters. MUST BE THE SAME AS IN 'Make_LAE.py'   ####
z_lya = [2, 3.5] # LAE z interval
z_oii = [ w_lya*(z + 1)/w_oii - 1 for z in z_lya] # OII z interval computed from z_lya
obs_area = 1 # deg**2

# Wavelength array where to evaluate the spectrum

w_min  = 2500   # Minimum wavelength
w_max  = 12000  # Maximum wavelgnth
N_bins = 10000  # Number of bins

w_Arr = np.linspace( w_min , w_max , N_bins )

####    Specific OII parameters
ew_in = [5, 50] 
w_in  = [5, 5.1]
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

bin_width = OII_LF[1,0] - OII_LF[0,0]
Volume_OII = z_volume(z_oii[0], z_oii[1], obs_area)
N_sources_OII = int(np.sum(OII_LF[:,1]) * bin_width * Volume_OII)
LF_p_cum = np.cumsum(OII_LF[:,1])
LF_p_cum /= np.max(LF_p_cum)
L_Arr = 10 ** np.interp(np.random.rand(100000), LF_p_cum, OII_LF[:,0])

# Define EW and z Arrays
z_Arr = np.random.rand(N_sources_OII) * (ew_in[1] - ew_in[0]) + ew_in[0]
z_Arr_OII = w_lya * (z_Arr + 1) / w_oii - 1
e_Arr = np.random.rand(N_sources_OII) * (ew_in[1] - ew_in[0]) + ew_in[0]\
    * (1 + z_Arr_OII[N_sources_OII]) / (1 + z_Arr[N_sources_OII])
widths_Arr = np.random.rand(N_sources_OII) * (w_in[1] - w_in[0]) + w_in[0]

# Define g flux array
g_Arr = L_flux_to_g(L_Arr, z_Arr_OII, e_Arr)

#### Grid dictionary load
Grid_Dictionary = Load_BC03_grid_data()
