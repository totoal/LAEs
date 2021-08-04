import numpy as np
import csv
from pylab import *
from my_utilities import *
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

filename = 'aaaa' #A: Identifier to save catalog to 'Source_cat_$filename$.npy

# Wavelength of emission lines: (Don't change)

w_Lya = 1215.67
w_OII = 0.5 * ( 3727.092 + 3729.875 )

#### Some parameters
z_in =  [   2,   3.3    ] # Redshift interval for Lya
z_oii = [ w_Lya*(z + 1)/w_OII - 1 for z in z_in] # Redshift interval for OII
area = 10 ## deg**2

#### Function to compute a volume from z interval

def z_volume(z_min, z_max):
    dc_max = cosmo.comoving_distance(z_max).to(u.Mpc).value
    dc_min = cosmo.comoving_distance(z_min).to(u.Mpc).value
    d_side = cosmo.kpc_comoving_per_arcmin((z_max - z_min)*0.5)\
            .to(u.Mpc/u.deg).value * area**0.5
    print('Volume: {} Mpc2'.format(str((dc_max - dc_min) * d_side**2)))
    return (dc_max - dc_min) * d_side**2

#### Function to calculate EW from line flux

def L_flux_to_g(L_Arr, rand_z_Arr, rand_EW_Arr):
    dL_Arr = cosmo.luminosity_distance(rand_z_Arr).to(u.cm).value
    g_Arr = 10**L_Arr / ((1 + rand_z_Arr) * rand_EW_Arr * 4*np.pi * dL_Arr**2) 
    return g_Arr

##########################################################################################

##########################################################################################
#####   Load HETDEX LumFunc for reference

filepath = '../csv/HETDEX_LumFunc.csv'
HETDEX_LF = []
with open(filepath, mode='r') as csvfile:
    rdlns = csv.reader(csvfile, delimiter=',')
    
    for line in rdlns:
        HETDEX_LF.append(line)
HETDEX_LF = np.array(HETDEX_LF).astype(float)

#### Calculate L_Arr distribution
bin_width = HETDEX_LF[1,0] - HETDEX_LF[0,0]
Volume_lya = z_volume(z_in[0], z_in[1])
N_sources_lya = int(np.sum(HETDEX_LF[:,1]) * bin_width * Volume_lya)
LF_p = HETDEX_LF[:,1] / np.sum(HETDEX_LF[:,1])
L_Arr_lya = np.random.choice(HETDEX_LF[:,0], N_sources_lya, True, LF_p)
L_Arr_lya += (2*np.random.rand(N_sources_lya) - 1) * bin_width

##########################################################################################
#####   Load OII LF (Gilbank 2010)

filepath = '../csv/Gilbank2010-OII_LF.csv'
OII_LF = []
with open(filepath, mode='r') as csvfile:
    rdlns = csv.reader(csvfile, delimiter=',')
    
    for line in rdlns:
        OII_LF.append(line)
OII_LF = np.array(OII_LF).astype(float)

#### Calculate L_Arr distribution
bin_width = OII_LF[1,0] - OII_LF[0,0]
Volume_OII = z_volume(z_oii[0], z_oii[1])
N_sources_OII = int(np.sum(OII_LF[:,1]) * bin_width * Volume_OII)
LF_p = OII_LF[:,1] / np.sum(OII_LF[:,1])
L_Arr_OII = np.random.choice(OII_LF[:,0], N_sources_OII, True, LF_p)
L_Arr_OII += (2*np.random.rand(N_sources_OII) - 1) * bin_width

#### Function to calculate EW from line flux

def L_flux_to_g(L_Arr, rand_z_Arr, rand_EW_Arr):
    dL_Arr = cosmo.luminosity_distance(rand_z_Arr).to(u.cm).value
    F_line_Arr = 10**L_Arr / (4*np.pi * dL_Arr**2) 
    g_Arr = F_line_Arr / rand_EW_Arr / 145 
    return g_Arr

##########################################################################################

# Define some of the parameter space to cover:

e_in = [  10    ,  60   ] #A: Rest frame equivalent Width of Lya
s_in = [ -31.  , -30.  ] # Logarithmic uncertainty in flux density # 
w_in = [   5.  ,   5.1 ] # Width of the Lya line.


# Define random propeties for the sample:

N_sources = N_sources_lya + N_sources_OII
print('N LAE: {}'.format(str(N_sources_lya)))
print('N OII: {}'.format(str(N_sources_OII)))

e_Arr = np.zeros(N_sources)

z_Arr = np.random.rand(N_sources) * (z_in[1] - z_in[0]) + z_in[0]
z_Arr_OII = w_Lya * (z_Arr + 1) / w_OII - 1
e_Arr[:N_sources_lya] = (np.random.rand(N_sources_lya) * (e_in[1] - e_in[0]) + e_in[0])
# OII emitters follow a lognormal dist with mean 10 sigma 0.77 (Blanton&Lin 2000)
e_Arr[N_sources_lya:] = np.random.lognormal(
        np.log(10**2 / (10**2 + 0.77**2)**0.5),
        np.log(1 + 0.77**2 / 10**2),
        N_sources_OII) * (1 + z_Arr_OII[N_sources_lya:]) / (1 + z_Arr[N_sources_lya:])
s_Arr = 10 ** ( np.random.rand( N_sources ) * ( s_in[1] - s_in[0] ) + s_in[0] )
W_Arr =         np.random.rand( N_sources ) * ( w_in[1] - w_in[0] ) + w_in[0]

g_Arr = np.zeros(N_sources)
g_Arr[:N_sources_lya] = L_flux_to_g(
        L_Arr_lya, z_Arr[:N_sources_lya], e_Arr[:N_sources_lya]
        )
g_Arr[N_sources_lya:] = L_flux_to_g(
        L_Arr_OII, z_Arr_OII[N_sources_lya:], e_Arr[N_sources_lya:]
        )

# OII parameters for Bruzual and charlot. This defines the continuum. 

OII_MET = 36.97118006667763
OII_AGE = 2.8748148018996997
OII_EXT = 6.913769672799271e-14

# Wavelength array where to evaluate the spectrum

w_min  = 2500   # Minimum wavelength
w_max  = 12000  # Maximum wavelgnth
N_bins = 10000  # Number of bins

w_Arr = np.linspace( w_min , w_max , N_bins )

# Intergalactic medium mean absortion parameters : (From Faucher et al)

T_A = -0.001845
T_B =  3.924

# Dependece of noise with wavelength

Noise_w_Arr = np.linspace( 3000 , 9000 , 10 )
Noise_Arr   = np.ones( len( Noise_w_Arr ) ) # Now it is flat.

##########################################################################################
##########################################################################################
##########################################################################################

Grid_Dictionary = Load_BC03_grid_data()

print( Grid_Dictionary.keys() )

print( 'ext' , Grid_Dictionary['ext_Arr'] )
print( 'age' , Grid_Dictionary['age_Arr'] )
print( 'met' , Grid_Dictionary['met_Arr'] )

##########################################################################################
##########################################################################################
##########################################################################################

# Let's load the data of the gSDSS filter

gSDSS_lambda_Arr_f , gSDSS_Transmission_Arr_f = Load_Filter( 'gSDSS' )

gSDSS_lambda_pivot , gSDSS_FWHM = FWHM_lambda_pivot_filter( 'gSDSS' )

gSDSS_data = {}

gSDSS_data[ 'lambda_Arr_f'       ] = np.copy( gSDSS_lambda_Arr_f       )
gSDSS_data[ 'Transmission_Arr_f' ] = np.copy( gSDSS_Transmission_Arr_f )
gSDSS_data[ 'lambda_pivot'       ] = np.copy( gSDSS_lambda_pivot       )
gSDSS_data[ 'FWHM'               ] = np.copy( gSDSS_FWHM               )

##########################################################################################
##########################################################################################
##########################################################################################

# Let's define the LAE properties for the continum

MASK_LAEs = np.ones(N_sources, dtype=bool)
MASK_LAEs[N_sources_lya:] = False

catalog_MET_Arr = np.zeros( N_sources )
catalog_AGE_Arr = np.zeros( N_sources )
catalog_EXT_Arr = np.zeros( N_sources )

z_True_Arr = np.copy( z_Arr )
z_True_Arr[ ~MASK_LAEs ] = w_Lya * ( 1 + z_True_Arr[ ~MASK_LAEs ] ) / w_OII - 1.

DD = 0.00001

MIN_MET = np.amin( Grid_Dictionary['met_Arr'] ) # Minimum metallicity
MAX_MET = np.amax( Grid_Dictionary['met_Arr'] ) # Maximum metallicity

MIN_AGE = np.amin( Grid_Dictionary['age_Arr'] ) # Minimum Age
MAX_AGE = 3.25                                  # Maximum Age

MIN_EXT = np.amin( Grid_Dictionary['ext_Arr'] ) # Minium extintion
MAX_EXT = 0.05                                  # Maximum extintion

## ALBERTO: defined this to naively avoid extremely deep absortions
new_MIN_AGE = (MAX_AGE + MIN_AGE) * 0.5

for i in range( 0 , N_sources ):

    if MASK_LAEs[i] : 
        MET = np.random.rand( ) * ( MAX_MET - MIN_MET ) + MIN_MET
        AGE = np.random.rand( ) * ( MAX_AGE - new_MIN_AGE ) + new_MIN_AGE
        EXT = np.random.rand( ) * ( MAX_EXT - MIN_EXT ) + MIN_EXT 

    else : 
        MET = OII_MET    
        AGE = OII_AGE  
        EXT = OII_EXT    

    catalog_MET_Arr[i] = ( 1 + np.random.rand( ) * DD  - 0.5*DD ) * MET
    catalog_AGE_Arr[i] = ( 1 + np.random.rand( ) * DD  - 0.5*DD ) * AGE
    catalog_EXT_Arr[i] = ( 1 + np.random.rand( ) * DD  - 0.5*DD ) * EXT

##########################################################################################
##########################################################################################
##########################################################################################

catalogs_SEDs    = np.zeros( N_sources * len(w_Arr) ).reshape( N_sources , len(w_Arr) )
catalogs_SEDs_No = np.zeros( N_sources * len(w_Arr) ).reshape( N_sources , len(w_Arr) )

for i in range( 0 , N_sources ):

    print( '     ' , i+1 , '/'  , N_sources, end='\r')

    my_z = z_True_Arr[i]

    my_ew     = e_Arr[i]
    my_flux_g = g_Arr[i]
    my_widths = W_Arr[i]
    my_noises = s_Arr[i]

    my_MET = catalog_MET_Arr[i]
    my_AGE = catalog_AGE_Arr[i]
    my_EXT = catalog_EXT_Arr[i]

    if     MASK_LAEs[i] :
        LINE = 'Lya'
        my_ew = (  1 + my_z  ) * my_ew
    if not MASK_LAEs[i] :
        LINE = 'OII'
        my_ew = (1 + z_Arr[i]) * my_ew

    tmp_spec , tmp_spec_no_IGM = generate_spectrum(
                                                 LINE, my_z, my_ew, my_flux_g,
                                                 my_widths, my_noises, my_MET,
                                                 my_AGE, my_EXT, w_Arr, Grid_Dictionary,
                                                 Noise_w_Arr, Noise_Arr, T_A, T_B,
                                                 gSDSS_data
                                                  )

    catalogs_SEDs[i] = tmp_spec

    catalogs_SEDs_No[i] = tmp_spec_no_IGM
print()

##########################################################################################
##########################################################################################
##########################################################################################

my_perfect_new_cat = {}

my_perfect_new_cat['LAE'] = MASK_LAEs

my_perfect_new_cat['SEDs'       ] = catalogs_SEDs
my_perfect_new_cat['SEDs_No_IGM'] = catalogs_SEDs_No

my_perfect_new_cat['w_Arr'] = w_Arr

my_perfect_new_cat['MET_Arr'] = catalog_MET_Arr
my_perfect_new_cat['AGE_Arr'] = catalog_AGE_Arr
my_perfect_new_cat['EXT_Arr'] = catalog_EXT_Arr

my_perfect_new_cat['redshift_Lya_Arr' ] = z_Arr
my_perfect_new_cat['redshift_Arr'     ] = z_True_Arr

#my_perfect_new_cat['flux_l_Arr'] = f_Arr
my_perfect_new_cat['flux_g_Arr'] = g_Arr
my_perfect_new_cat['widths_Arr'] = W_Arr
my_perfect_new_cat['Noises_Arr'] = s_Arr
my_perfect_new_cat['EW_Arr']     = e_Arr

##########################################################################################
##########################################################################################
##########################################################################################

np.save( 'Source_cat_' + filename + '.npy' , my_perfect_new_cat )
