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
def Compute_log_like( model_Arr , stack_Arr , stack_err_Arr ):

    sigma2 = stack_err_Arr**2

    cc = 1.0

    #xi = np.sum( cc *( stack_Arr - model_Arr ) ** 2 / sigma2 )

    log_like = -0.5 * np.sum( cc *( model_Arr - stack_Arr ) ** 2 / sigma2 + np.log(sigma2))
    #log_like = -0.5 * np.sum( cc *( stack_Arr - model_Arr ) ** 2 / sigma2 )

   #log_like = -1. * np.log10( xi )

    return log_like
#======================================================#
#======================================================#
#======================================================#
def prior_f( theta ):

    log_AGE , MET , EXT , log_CTE = theta[0] , theta[1] , theta[2] , theta[3]

    AGE = 10 ** log_AGE
    CTE = 10 ** log_CTE

    my_bool = True 

    bool_age = ( AGE     >  1e-3 ) * ( AGE     <  2.00 )
    bool_met = ( MET     >  22.  ) * ( MET     < 72    )
    bool_ext = ( EXT     >   0.  ) * ( EXT     <  1.0  )
    bool_cte = ( CTE     >  1e-5 ) * ( CTE     <  1.0  )

    return_bool = my_bool * bool_age * bool_met * bool_ext * bool_cte

    return return_bool
#======================================================#
#======================================================#
#======================================================#
def get_spec( w_stack_Arr , MET , AGE , EXT , CTE , stack_OII_Arr , Grid_Dictionary , w_int_min , w_int_max ):

    w_grid_Arr , flux_Arr = Ke.Interpolate_Lines_Arrays_3D_grid_MCMC( MET , AGE , EXT , Grid_Dictionary )

    tmp_model_Arr = np.interp( w_stack_Arr , w_grid_Arr , flux_Arr )

    mask_w = ( w_stack_Arr > w_int_min ) * ( w_stack_Arr < w_int_max )

    II_model = np.trapz( tmp_model_Arr[ mask_w ] , w_stack_Arr[ mask_w ] )
    II_OII   = np.trapz( stack_OII_Arr[ mask_w ] , w_stack_Arr[ mask_w ] )

    tmp_model_Arr = tmp_model_Arr * 1. / II_model
    stack_OII_Arr = stack_OII_Arr * 1. / II_OII

    model_Arr = CTE * tmp_model_Arr + ( 1. - tmp_model_Arr ) * stack_OII_Arr 

    return model_Arr
#======================================================#
#======================================================#
#======================================================#
def main_f( theta , w_stack_Arr , stack_Arr , stack_err_Arr , stack_OII_Arr , Grid_Dictionary , w_int_min , w_int_max ):

    if not prior_f( theta ) : return -np.inf

    log_AGE , MET , EXT , log_CTE = theta[0] , theta[1] , theta[2] , theta[3] 

    AGE = 10 ** log_AGE
    CTE = 10 ** log_CTE

    model_Arr = get_spec( w_stack_Arr , MET , AGE , EXT , CTE , stack_OII_Arr , Grid_Dictionary , w_int_min , w_int_max ) 

    w_min_fit = 1230
    w_max_fit = 1725

    mask_line = Ke.mask_spectrum_from_emission_lines( w_stack_Arr , Dw=20 , w_min=w_min_fit , w_max=w_max_fit )

    II_stack = np.trapz( stack_Arr[ mask_line ] , w_stack_Arr[ mask_line ] )
    II_model = np.trapz( model_Arr[ mask_line ] , w_stack_Arr[ mask_line ] )

    my_model_Arr = model_Arr * II_stack / II_model

    ln_like = Compute_log_like( my_model_Arr[ mask_line ] , stack_Arr[ mask_line ] , stack_err_Arr[ mask_line ] )

    return ln_like
#======================================================#
#======================================================#
#======================================================#

DATA = np.load( 'OBSERVED_STACK.npy' )

w_rest_Arr  = DATA['w_Arr']

f_stack_Arr = DATA['f_Arr']
s_stack_Arr = DATA['s_Arr']

#======================================================#
#======================================================#
#======================================================#

Grid_Dictionary = Ke.Load_BC03_grid_data()

print( Grid_Dictionary.keys() )
print( Grid_Dictionary['w_Arr'] )

#======================================================#
#======================================================#
#======================================================#
AGE_OII = 10**-1.36344111 
MET_OII = 41.99792174  
EXT_OII = 0.34791187

w_Lya = 1215.67
w_OII = 0.5 * ( 3727.092 + 3729.875 )

w_grid_OII_Arr , flux_OII_Arr = Ke.Interpolate_Lines_Arrays_3D_grid_MCMC( MET_OII , AGE_OII , EXT_OII , Grid_Dictionary )

w_grid_Lya_Arr = w_grid_OII_Arr * w_Lya / w_OII

stack_OII_Arr = np.interp( w_rest_Arr , w_grid_Lya_Arr , flux_OII_Arr )
#======================================================#
#======================================================#
#======================================================#
w_int_min = 1350.
w_int_max = 1375.
#======================================================#
#======================================================#
#======================================================#
N_walkers = 20
N_dim     =  4
N_steps   = 20
N_burn    = 20 
#======================================================#
#======================================================#
#======================================================#
theta_0 = np.zeros( N_walkers * N_dim ).reshape( N_walkers , N_dim )

log_AGE_min =  -2.
log_AGE_max =   0.0

MET_min = 22.
MET_max = 72.

EXT_min = 0.0
EXT_max = 1.0

log_CTE_min =  -5.
log_CTE_max =   0.

theta_0[ : , 0 ] = np.random.rand( N_walkers ) * ( log_AGE_max - log_AGE_min ) + log_AGE_min
theta_0[ : , 1 ] = np.random.rand( N_walkers ) * (     MET_max -     MET_min ) +     MET_min
theta_0[ : , 2 ] = np.random.rand( N_walkers ) * (     EXT_max -     EXT_min ) +     EXT_min
theta_0[ : , 3 ] = np.random.rand( N_walkers ) * ( log_CTE_max - log_CTE_min ) + log_CTE_min

#======================================================#
#======================================================#
#======================================================#
#ln_p = main_f( theta , w_rest_Arr , stack_rest_Arr , stack_rest_Arr , Grid_Dictionary )

args = (  w_rest_Arr , f_stack_Arr , s_stack_Arr , stack_OII_Arr , Grid_Dictionary , w_int_min , w_int_max )

sampler = emcee.EnsembleSampler( N_walkers , N_dim, main_f , args=args )

###################################################################
###################################################################
###################################################################
state = sampler.run_mcmc( theta_0 , N_burn , progress=True )

sampler.reset()

sampler.run_mcmc( state , N_steps , progress=True )

chains = sampler.get_chain()

flat_samples = np.zeros( N_walkers * N_steps * N_dim ).reshape( N_walkers * N_steps , N_dim )

for i in range( 0 , N_dim ):
    flat_samples[ : , i ] = chains[ : , : , i ].ravel()

matrix_sol = np.zeros( N_dim )

for i in range( 0 , N_dim ):
    matrix_sol[i] = np.mean( flat_samples[ : , i ] )

my_chains_matrix = np.copy( flat_samples )
###################################################################
###################################################################
###################################################################
ax_list = []

label_list = [ r'$\log \; Age \; [Gyr]$' , r'$\rm Metallicity$' , r'$\rm Extintion$' , 'CTE']

MAIN_VALUE_mean   = np.zeros(N_dim)
MAIN_VALUE_median = np.zeros(N_dim)

for i in range( 0 , N_dim ):

    x_prop = my_chains_matrix[ : , i ]

    x_prop_min = np.percentile( x_prop , 5  )
    x_prop_50  = np.percentile( x_prop , 50 )
    x_prop_max = np.percentile( x_prop , 95 )

    x_min = x_prop_50 - ( x_prop_max - x_prop_min ) * 1.00
    x_max = x_prop_50 + ( x_prop_max - x_prop_min ) * 1.00

    mamamask = ( x_prop > x_min ) * ( x_prop < x_max ) 
    
    MAIN_VALUE_mean[  i] = np.mean(       x_prop[ mamamask ] )
    MAIN_VALUE_median[i] = np.percentile( x_prop[ mamamask ] , 50 )

figure( figsize=(15,15) )

for i in range( 0 , N_dim ):

    y_prop = my_chains_matrix[ : , i ]

    y_prop_min = np.percentile( y_prop , 5  )
    y_prop_50  = np.percentile( y_prop , 50 )
    y_prop_max = np.percentile( y_prop , 95 )

    y_min = y_prop_50 - ( y_prop_max - y_prop_min ) * 1.00
    y_max = y_prop_50 + ( y_prop_max - y_prop_min ) * 1.00

    for j in range( 0 , N_dim ):

        if i < j : continue

        x_prop = my_chains_matrix[ : , j ]

        x_prop_min = np.percentile( x_prop , 5  )
        x_prop_50  = np.percentile( x_prop , 50 )
        x_prop_max = np.percentile( x_prop , 95 )

        x_min = x_prop_50 - ( x_prop_max - x_prop_min ) * 1.00 
        x_max = x_prop_50 + ( x_prop_max - x_prop_min ) * 1.00

        ax = plt.subplot2grid( ( N_dim , N_dim ) , (i, j)  )

        ax_list += [ ax ]

        DDX = x_max - x_min
        DDY = y_max - y_min

        if i==j :

            H , edges = np.histogram( x_prop , 100 , range=[x_min,x_max] )

            ax.hist( x_prop , 100 , range=[x_min,x_max] , color='cornflowerblue' )

            ax.plot( [ MAIN_VALUE_median[i] , MAIN_VALUE_median[i] ] , [ 0.0 , 1e10 ] , 'k--' , lw=2 )

            ax.set_ylim( 0 , 1.1 * np.amax(H) )

        else :

            XX_min = x_min - DDX * 0.2
            XX_max = x_max + DDX * 0.2

            YY_min = y_min - DDY * 0.2
            YY_max = y_max + DDY * 0.2

            H , edges_y , edges_x = np.histogram2d( x_prop , y_prop , 100 , range=[[XX_min , XX_max],[YY_min , YY_max]] )

            y_centers = 0.5 * ( edges_y[1:] + edges_y[:-1] ) 
            x_centers = 0.5 * ( edges_x[1:] + edges_x[:-1] ) 

            H_min = np.amin( H )
            H_max = np.amax( H )

            print( 'H_min =' , H_min )
            print( 'H_max =' , H_max )

            N_bins = 10000

            H_Arr = np.linspace( H_min , H_max , N_bins )[::-1]

            fact_up_Arr = np.zeros( N_bins )

            TOTAL_H = np.sum( H )

            for iii in range( 0 , N_bins ):
 
                mask = H > H_Arr[iii]

                fact_up_Arr[iii] = np.sum( H[ mask ] ) / TOTAL_H

            print( fact_up_Arr )

            H_value_68 = np.interp( 0.680 , fact_up_Arr , H_Arr )
            H_value_95 = np.interp( 0.950 , fact_up_Arr , H_Arr )

            print( 'H_value_68 = ' , H_value_68 )
            print( 'H_value_95 = ' , H_value_95 )

            pcolormesh( edges_y , edges_x , H.T , cmap='Blues' )

            ax.contour( y_centers, x_centers , H.T , colors='k' , levels=[ H_value_95 ] ) 
            ax.contour( y_centers, x_centers , H.T , colors='r' , levels=[ H_value_68 ] ) 
            #ax.contour( y_centers, x_centers , H.T , colors='k' , levels=[ H_value_68 ] ) 

            X_VALUE =  MAIN_VALUE_median[j] 
            Y_VALUE =  MAIN_VALUE_median[i] 

            plot( [ X_VALUE , X_VALUE ] , [    -100 ,     100 ] , 'k--' , lw=2 )
            plot( [    -100 ,     100 ] , [ Y_VALUE , Y_VALUE ] , 'k--' , lw=2 )

            x_plot = [ x_min , x_max , x_max , x_min , x_min ]
            y_plot = [ y_min , y_min , y_max , y_max , y_min ]

            ax.plot( x_plot , y_plot , 'w' , lw=2 )

            ax.set_ylim( y_min-0.05*DDY , y_max+0.05*DDY )

        ax.set_xlim( x_min-0.05*DDX , x_max+0.05*DDX )

        if i==2:
            ax.set_xlabel( label_list[j] , size=20 )

        if j==0 and i!=0 :
            ax.set_ylabel( label_list[i] , size=20 )
###################################################################
###################################################################
###################################################################

for i in [ 0 , 1 , 2 ] : 
    plt.setp( ax_list[i].get_xticklabels(), visible=False)

for i in [ 0 , 2 , 4 , 5 ] : 
    plt.setp( ax_list[i].get_yticklabels(), visible=False)

plt.subplots_adjust( left = 0.085 , bottom = 0.13 , right = 0.96 , top = 0.85 , wspace=0.0 , hspace=0.0 )

savefig( 'fig_mcmc_corners_.pdf' )

###################################################################
###################################################################
###################################################################

save_name = 'mcmc_chains_Nw_' + str( N_walkers ) + '_Nd_' + str(N_dim) + '_Ns_' + str( N_steps ) + '_Nb_' + str(N_burn) + '.npy'

np.save( save_name , my_chains_matrix )

print( 'MAIN_VALUE_median = ' , MAIN_VALUE_median )

###################################################################
###################################################################
###################################################################

clf()
figure( figsize=(15,5) )

#w_rest_Arr , MET , AGE , EXT , CTE , stack_OII_Arr , Grid_Dictionary , w_int_min , w_int_max
model_Arr = get_spec( w_rest_Arr , MAIN_VALUE_median[1] , 10**MAIN_VALUE_median[0] , MAIN_VALUE_median[2] , MAIN_VALUE_median[3] , stack_OII_Arr , Grid_Dictionary , w_int_min , w_int_max )

w_min_fit = 1250
w_max_fit = 1550

mask = ( w_rest_Arr > w_min_fit ) * ( w_rest_Arr < w_max_fit )

II_model = np.trapz( model_Arr[ mask ] , w_rest_Arr[mask] )
II_stack = np.trapz( f_stack_Arr[ mask ] , w_rest_Arr[mask] )

my_model_Arr = model_Arr * II_stack / II_model

plot( w_rest_Arr , my_model_Arr , 'r' , lw=2 )

plot( w_rest_Arr , f_stack_Arr , 'k' , lw=2 )

ylim( -1.02 , 4.00 )
xlim( 820, 1900 )

savefig( 'fig_spec_mcmc_best_fit.pdf' )

sys.exit() 
#======================================================#
#======================================================#
#======================================================#

print(  MET , AGE , EXT )

#======================================================#
#======================================================#
#======================================================#
