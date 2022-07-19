import sys

import numpy as np

import matplotlib
# see http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
matplotlib.use('Svg')

from pylab import *

import Keith as Ke

from astropy.cosmology import Planck15 as cosmo

from scipy.stats import norm

import scipy

import emcee 
###########

# IMPORTANT !!!

# scipy/0.17.0-foss-2016a-Python-2.7.11

###########

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

# OII stack properties!!
#[-1.36344111 41.99792174  0.34791187]

#======================================================#
#======================================================#
#======================================================#
#file_name = 'CATALOG_TO_COMPUTE_T'
#
#data = np.load(  '../EXTRACT_T/' + file_name + '.npy' , allow_pickle=True ).item()
#
#SEDs     = data['SEDs']
#SEDs_err = data['SEDs_err']
#
##redshift_Arr = data['wave'].data / 1215.67 - 1.
#
#w_Arr = data['SEDs_w'] 
#======================================================#
#======================================================#
#======================================================#

#file_name = 'CUT_BY_HAND_TNSE_OUTPUT_p_0.9_pp_10.0_EW0_15.0_LR_1.0'
file_name = 'X_matched_HETDEX_r_10.0'

DATA_0 = np.load( '../TSNE/' + file_name + '.npy' , allow_pickle=True ).item()

#Out[5]: dict_keys(['detectid', 'fwhm', 'throughput', 'fluxlimit_4540', 'shotid', 'field', 'n_ifu', 'gmag_err', 'gmag', 'ra', 'dec', 'date', 'obsid', 'wave', 'wave_err', 'flux',     'flux_err', 'linewidth', 'linewidth_err', 'continuum', 'continuum_err', 'sn', 'sn_err', 'chi2', 'chi2_err', 'multiframe', 'fibnum', 'x_raw', 'y_raw', 'amp', 'chi2fib', 'detectnam    e', 'expnum', 'fiber_id', 'ifuid', 'ifuslot', 'inputid', 'noise_ratio', 'specid', 'weight', 'x_ifu', 'y_ifu', 'combined_continuum', 'combined_continuum_err', 'combined_plae', 'co    mbined_plae_err', 'mag_sdss_g', 'mag_sdss_g_err', 'plae_classification', 'plae_sdss_g', 'plae_sdss_g_max', 'plae_sdss_g_min', 'SEDs', 'SEDs_w'])

#print( data.keys() )
#=========================================#
#=========================================#

z_min=2.0
z_max=3.0

sn_min = float( sys.argv[1] ) #5.5
sn_max = None

mag_sdss_g_min = float( sys.argv[2] )
mag_sdss_g_max = None

plae_classification_min = 0.85
combined_plae_min       = 70.

EW_0_min = None
EW_0_max = None

EW_ex_min = None

L_Lya_max = None # 10**(43.5)
L_Lya_min = None # 10**(43.8)
#=========================================#
#=========================================#

my_str_SAVE = '_z_' + str(z_min) + '_' + str(z_max) + '_sn_' + str(sn_min) + '_g_' + str(mag_sdss_g_min) + '_p_' + str(plae_classification_min) + '_pp_' + str(combined_plae_min)

data = Ke.mask_data( DATA_0 , sn_min=sn_min, sn_max=sn_max , plae_classification_min=plae_classification_min , combined_plae_min=combined_plae_min , mag_sdss_g_min=mag_sdss_g_min , mag_sdss_g_max=mag_sdss_g_max, EW_0_min=EW_0_min, EW_0_max=EW_0_max, EW_ex_min=EW_ex_min, L_Lya_max=L_Lya_max, L_Lya_min=L_Lya_min, z_min=z_min, z_max=z_max )

#======================================================#
#======================================================#
#======================================================#
SEDs     = data['SEDs']
SEDs_err = data['SEDs_err']

#redshift_Arr = data['wave'].data / 1215.67 - 1.

w_Arr = data['SEDs_w'] 
#======================================================#
#======================================================#
#======================================================#
w_Lya = 1215.67

w_rest_Arr = np.linspace( 850 , 2000 , 10000 )

redshift_Lya_Arr = data['wave'].data / w_Lya - 1.

w_lya_match_min = 1250
w_lya_match_max = 1400

####z_min_match_lya = 1.9
####z_max_match_lya = 3.5
#======================================================#
#======================================================#
#======================================================#
###Dz_Lya_Arr = [ 0.1 , 0.2 , 0.3 ]
###Nz_Lya = 20
###
###cm = get_cmap( 'rainbow' )
###
###z_list = []
###f_list = []
###
###for i , Dz_Lya in enumerate( Dz_Lya_Arr ) :
###
###    cte = i * 1. / ( len( Dz_Lya_Arr ) - 0.99 )
###
###    z_centers , f_stack_Arr , s_stack_Arr = Ke.COMPUTE_STACK_corrections( SEDs , SEDs_err , w_Arr , redshift_Lya_Arr , w_rest_Arr , w_lya_match_min , w_lya_match_max , Dz_Lya , Nz_Lya )
###
###    plot( z_centers , f_stack_Arr , 'o'  , color=cm(cte) , label='Dz='+str(Dz_Lya) )
###
###    z_list +=   z_centers.tolist()
###    f_list += f_stack_Arr.tolist()
###
###z_list = np.array( z_list )
###f_list = np.array( f_list )
###
###mask_ok = np.isfinite( f_list )
###
###z_list = z_list[ mask_ok ]
###f_list = f_list[ mask_ok ]
###
###print( 'z_list' , z_list )
###print( 'f_list' , f_list )
####======================================================#
####======================================================#
####======================================================#
###deg    =  2
###N_iter = 10
###
###P = Ke.make_fit_corrections( N_iter , z_list , f_list , deg )
###
###X_Arr = np.linspace( 1.9 , 3.5 , 1000 )
###
###Y_Arr = np.zeros( len(X_Arr) )
###
###for i in range( 0 , deg+1 ):
###
###    Y_Arr += P[i]*X_Arr**(deg-i)
###
###plot( X_Arr , Y_Arr , 'k' )
###
###legend(loc=0)
###
###savefig( 'fig_F_stack_evolution_Lya_'+my_str_SAVE+'.pdf' )
###clf()
#======================================================#
#======================================================#
#======================================================#

mask_zz = ( redshift_Lya_Arr > 2.05 ) * ( redshift_Lya_Arr < 2.85 )

#stack_rest_Arr , stack_rest_err_Arr = Ke.COMPUTE_STACK_corrected( SEDs[ mask_zz ] , SEDs_err[ mask_zz ] , w_Arr , redshift_Lya_Arr[ mask_zz ] , w_rest_Arr , X_Arr , Y_Arr )
stack_rest_Arr , stack_rest_err_Arr = Ke.COMPUTE_STACK_normalized( SEDs[ mask_zz ] , SEDs_err[ mask_zz ] , w_Arr , redshift_Lya_Arr[ mask_zz ] , w_rest_Arr )

#======================================================#
#======================================================#
#======================================================#

#names = [ 'w_Arr' , 'f_Arr' , 's_Arr' ]
#
#formats = [ np.float64 ] * len( names )

data_to_save = {} #np.zeros( len(w_rest_Arr) , dtype={'names':names,'formats':formats} )

data_to_save[ 'w_Arr' ] =     w_rest_Arr
data_to_save[ 'f_Arr' ] = stack_rest_Arr
data_to_save[ 's_Arr' ] = stack_rest_err_Arr

data_to_save[ 'z_Cor_Arr' ] = X_Arr
data_to_save[ 'f_Cor_Arr' ] = Y_Arr

data_to_save[ 'z_STACK_Arr' ] = z_list
data_to_save[ 'f_STACK_Arr' ] = f_list

np.save( 'OBSERVED_STACKS/OBSERVED_STACK'+my_str_SAVE+'.npy' , data_to_save )

#======================================================#
#======================================================#
#======================================================#
old_stack_rest_Arr , old_stack_rest_err_Arr = Ke.COMPUTE_STACK( SEDs , SEDs_err , w_Arr , redshift_Lya_Arr , w_rest_Arr )

old_mask_w = ( w_rest_Arr > w_lya_match_min ) * ( w_rest_Arr < w_lya_match_max )

II_new = np.trapz(     stack_rest_Arr[ old_mask_w ] , w_rest_Arr[ old_mask_w ] )
II_old = np.trapz( old_stack_rest_Arr[ old_mask_w ] , w_rest_Arr[ old_mask_w ] )

old_stack_rest_Arr = old_stack_rest_Arr * II_new / II_old

figure( figsize=(25,5) )

plot( w_rest_Arr , old_stack_rest_Arr , label='old' , color='orangered')

plot( w_rest_Arr , stack_rest_Arr , color='cornflowerblue' , label='new' )

fill_between( w_rest_Arr , -1.*stack_rest_err_Arr , 1.*stack_rest_err_Arr , color='cornflowerblue' , alpha=0.5 )
fill_between( w_rest_Arr , -2.*stack_rest_err_Arr , 2.*stack_rest_err_Arr , color='cornflowerblue' , alpha=0.5 )

ylim( -1. , 5. )

legend(loc=0)

savefig( 'fig_F_stack_Lya_MCMC_test'+my_str_SAVE+'.pdf' )
clf()
#sys.exit()
#======================================================#
#======================================================#
#======================================================#

figure( figsize=(25,15) )

ax1 = subplot(211)
ax2 = subplot(212)

Nz = 6

z_edges = np.linspace( 1.9 , 2.9 , Nz+1 )

mask_w = ( w_rest_Arr > w_lya_match_min ) * ( w_rest_Arr < w_lya_match_max )

cm = get_cmap('rainbow')

for i in range( 0 , Nz ):

    cte = i * 1. / ( Nz - 0.9999 )

    color = cm( cte )

    z_bin_min = z_edges[ i   ]
    z_bin_max = z_edges[ i+1 ]

    mask_z = ( redshift_Lya_Arr > z_bin_min ) * ( redshift_Lya_Arr < z_bin_max )

    tmp_stack , _ = Ke.COMPUTE_STACK( SEDs[ mask_z ] , SEDs_err[ mask_z ] , w_Arr , redshift_Lya_Arr[ mask_z ] , w_rest_Arr )

    label = str( np.round( z_bin_min , 2 ) ) + '<z<' + str( np.round( z_bin_max , 2 ) ) 

    ax1.plot( w_rest_Arr , tmp_stack , color=color , label=label )

    II = np.trapz( tmp_stack[ mask_w ] , w_rest_Arr[ mask_w ] )

    ax2.plot( w_rest_Arr , tmp_stack / II , color=color , label=label )

ax1.set_ylim( -0.1 , 0.4 )

ax2.set_ylim( -0.01 , 0.02 )

ax2.legend(loc=0)

savefig( 'fig_LAE_STACLS_COMPARISON'+my_str_SAVE+'.pdf' )















