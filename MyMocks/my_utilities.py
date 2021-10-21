import numpy as np
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import norm
import scipy
import astropy.units as u

#import JBOSS as jp

import time
#==============================================================#
#==============================================================#
#==============================================================#
def Load_Filter( filter_name ):

    '''
        Returns the response of a filter for several wavelengths.

        Input : string : A filter name, e.g. J0378

        Output : 2 arrays:
            first : wavelength (amstrongs)
            second : system respose
    '''

    Trans_dir = './'

    Trans_name = Trans_dir + 'Transmission_Curves_20170316/' + 'JPAS_' + filter_name + '.tab'

    lambda_Amstrongs , Response = np.loadtxt( Trans_name , unpack=True )

    return lambda_Amstrongs , Response
#==============================================================#
#==============================================================#
#==============================================================#
def FWHM_lambda_pivot_filter( filter_name ):

    '''
        Return the lambda pivot and the FWHM of a filter

        Input : string : A filter name, e.g. J0378

        Output : 2 floats:
            first : lambda pivot (amstrongs)
            second : FWHM ( amstrongs )
    '''

    lambda_Arr , Transmission = Load_Filter( filter_name )

    # lambda_pivot

    intergral_up  = np.trapz( Transmission *      lambda_Arr , lambda_Arr )
    intergral_bot = np.trapz( Transmission * 1. / lambda_Arr , lambda_Arr )

    lambda_pivot = np.sqrt( intergral_up * 1. / intergral_bot )

    # FWHM

    mask = Transmission  > np.amax( Transmission ) * 0.5

    FWHM = lambda_Arr[ mask ][ -1 ] - lambda_Arr[ mask ][ 0 ]

    FWHM = np.absolute( FWHM )

    return lambda_pivot , FWHM
#==============================================================#
#==============================================================#
#==============================================================#
def Synthetic_Photometry_measure_flux_simple( lambda_Arr , f_lambda_Arr , filter_name ):

    '''
        Synthetic fotometry for a spectrum. ( Advanced function ( lv2 ) ).
        Use this when running low amount of synthetic photometries.

        Input :
            lambda_Arr   : An array with the wavelenths of the spectrum
            f_lambda_Arr : An array with the fluxes  (per unit of amstrong) of the spectrum

            filter_name  : An string with the filter name. E.g. J0378

        Output :
            f_lambda_mean : A float with the synthetic photometry at the NB
    '''

    lambda_Arr_f , Transmission_Arr_f = Load_Filter( filter_name )

    lambda_pivot , FWHM = FWHM_lambda_pivot_filter( filter_name )

    f_lambda_mean = Synthetic_Photometry_measure_flux( lambda_Arr , f_lambda_Arr , lambda_Arr_f , Transmission_Arr_f , lambda_pivot , FWHM )

    return f_lambda_mean
#==============================================================#
#==============================================================#
#==============================================================#
def Synthetic_Photometry_measure_flux( lambda_Arr , f_lambda_Arr , lambda_Arr_f , Transmission_Arr_f , lambda_pivot , FWHM ):

    '''
        Synthetic fotometry for a spectrum. ( Basic function ( lv1 ) ).
        Use this when running several synthetic photometries to avoid
        loading each time the filters.

        Input :
            lambda_Arr   : An array with the wavelenths of the spectrum
            f_lambda_Arr : An array with the fluxes  (per unit of amstrong) of the spectrum

            lambda_Arr_f       : An array with the wavelenths of the filter
            Transmission_Arr_f : An array with the response   of the filter

            lambda_pivot : A float with the lambda pivot of the filter
            FWHM         : A float with the    FHWM      of the filter

        Output :
            f_lambda_mean : A float with the synthetic photometry at the NB
    '''
    bin_lambda_filter = lambda_Arr_f[1]  - lambda_Arr_f[0]

    bin_lambda_spectrum = lambda_Arr[1] - lambda_Arr[0]

    if bin_lambda_filter > bin_lambda_spectrum :

        mask_integration_spect = ( lambda_Arr > lambda_pivot - FWHM ) * ( lambda_Arr < lambda_pivot + FWHM )

        LAMBDA_to_use = lambda_Arr[ mask_integration_spect ]

        SPECTRUM_to_use = f_lambda_Arr[ mask_integration_spect ]

        TRANMISSION_to_use = np.interp( LAMBDA_to_use , lambda_Arr_f , Transmission_Arr_f , left = 0 , right = 0 )

    else:

        mask_integration_filter = ( lambda_Arr_f > lambda_pivot - FWHM ) * ( lambda_Arr_f < lambda_pivot + FWHM )

        LAMBDA_to_use = lambda_Arr_f[ mask_integration_filter ]

        TRANMISSION_to_use = Transmission_Arr_f[ mask_integration_filter ]

        SPECTRUM_to_use = np.interp( LAMBDA_to_use , lambda_Arr , f_lambda_Arr , left = 0 , right = 0 )

    numerador = np.trapz( LAMBDA_to_use * SPECTRUM_to_use * TRANMISSION_to_use , LAMBDA_to_use )

    denominador = np.trapz( LAMBDA_to_use * TRANMISSION_to_use , LAMBDA_to_use )

    f_lambda_mean = numerador * 1. / denominador

    return f_lambda_mean
#==============================================================#
#==============================================================#
#==============================================================#
def IGM_TRANSMISSION( redshift_Arr , A , B ):

    Transmission_Arr = np.exp( A * ( 1 + redshift_Arr )**B )

    return Transmission_Arr
##==============================================================#
##==============================================================#
##==============================================================#
def Load_BC03_grid_data():

    path = 'BC03_Interpolation/'

    name = 'data_from_BC03.npy'

    file_name = path + '/' + name

    loaded_model = np.load( file_name , allow_pickle=True , encoding='latin1' ).item()

    return loaded_model
#==============================================================#
#==============================================================#
#==============================================================#
def Interpolate_Lines_Arrays_3D_grid_MCMC( Met_value , Age_value , Ext_value , Grid_Dictionary ):

    Grid_Line = Grid_Dictionary['grid']

    met_Arr_Grid = Grid_Dictionary['met_Arr']
    age_Arr_Grid = Grid_Dictionary['age_Arr']
    ext_Arr_Grid = Grid_Dictionary['ext_Arr']

    w_Arr = Grid_Dictionary['w_Arr']

    aux_line = Linear_3D_interpolator( Met_value , Age_value , Ext_value , met_Arr_Grid , age_Arr_Grid , ext_Arr_Grid , Grid_Line )

    return w_Arr , aux_line
#==============================================================#
#==============================================================#
#==============================================================#
def Linear_3D_interpolator( X_prob , Y_prob , Z_prob , X_grid , Y_grid , Z_grid , Field_in_grid ):

    INDEX_X = np.where( ( X_grid < X_prob ) )[0][-1]
    INDEX_Y = np.where( ( Y_grid < Y_prob ) )[0][-1]
    INDEX_Z = np.where( ( Z_grid < Z_prob ) )[0][-1]


    dX_grid = X_grid[ INDEX_X + 1 ] - X_grid[ INDEX_X ]
    dY_grid = Y_grid[ INDEX_Y + 1 ] - Y_grid[ INDEX_Y ]
    dZ_grid = Z_grid[ INDEX_Z + 1 ] - Z_grid[ INDEX_Z ]

    X_min_grid = X_grid[ INDEX_X     ]
    Y_min_grid = Y_grid[ INDEX_Y     ]
    Z_min_grid = Z_grid[ INDEX_Z     ]

    Xprob_X0 = ( X_prob - X_min_grid ) * 1. / dX_grid
    Yprob_Y0 = ( Y_prob - Y_min_grid ) * 1. / dY_grid
    Zprob_Z0 = ( Z_prob - Z_min_grid ) * 1. / dZ_grid

    Vol1 = ( 1. - Xprob_X0 ) * ( 1. - Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol2 = ( 1. - Xprob_X0 ) * (      Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol3 = ( 1. - Xprob_X0 ) * (      Yprob_Y0 ) * (      Zprob_Z0 )
    Vol4 = ( 1. - Xprob_X0 ) * ( 1. - Yprob_Y0 ) * (      Zprob_Z0 )

    Vol5 = (      Xprob_X0 ) * ( 1. - Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol6 = (      Xprob_X0 ) * (      Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol7 = (      Xprob_X0 ) * (      Yprob_Y0 ) * (      Zprob_Z0 )
    Vol8 = (      Xprob_X0 ) * ( 1. - Yprob_Y0 ) * (      Zprob_Z0 )

    Field1 = Field_in_grid[ INDEX_X     , INDEX_Y     , INDEX_Z     ]
    Field2 = Field_in_grid[ INDEX_X     , INDEX_Y + 1 , INDEX_Z     ]
    Field3 = Field_in_grid[ INDEX_X     , INDEX_Y + 1 , INDEX_Z + 1 ]
    Field4 = Field_in_grid[ INDEX_X     , INDEX_Y     , INDEX_Z + 1 ]
    Field5 = Field_in_grid[ INDEX_X + 1 , INDEX_Y     , INDEX_Z     ]
    Field6 = Field_in_grid[ INDEX_X + 1 , INDEX_Y + 1 , INDEX_Z     ]
    Field7 = Field_in_grid[ INDEX_X + 1 , INDEX_Y + 1 , INDEX_Z + 1 ]
    Field8 = Field_in_grid[ INDEX_X + 1 , INDEX_Y     , INDEX_Z + 1 ]

    Field_at_the_prob_point = Vol1 * Field1 + Vol2 * Field2 + Vol3 * Field3 + Vol4 * Field4 + Vol5 * Field5 + Vol6 * Field6 + Vol7 * Field7 + Vol8 * Field8

    return Field_at_the_prob_point
#==============================================================#
#==============================================================#
#==============================================================#
#======================================================#
#======================================================#
#======================================================#
def gaussian_f( x_Arr , mu , sigma , Amp ):

    y_Arr = norm.pdf(x_Arr, mu, sigma) * Amp 

    return y_Arr
#======================================================#
#======================================================#
#======================================================#
def plot_a_rebinned_line( new_wave_Arr , binned_line , Bin ):

    DD = Bin * 1e-10

    XX_Arr = np.zeros( len( new_wave_Arr ) * 2 )
    YY_Arr = np.zeros( len( new_wave_Arr ) * 2 )

    for i in range( 0 , len( new_wave_Arr ) ):

        i_0 = 2 * i
        i_1 = 2 * i + 1

        XX_Arr[ i_0 ] = new_wave_Arr[i] - 0.5 * Bin + DD
        XX_Arr[ i_1 ] = new_wave_Arr[i] + 0.5 * Bin

        YY_Arr[ i_0 ] = binned_line[i]
        YY_Arr[ i_1 ] = binned_line[i]

    return XX_Arr , YY_Arr
#======================================================#
#======================================================#
#======================================================#
def Measure_Lya_flux_props( w_Arr , F_Arr , redshift , i=0 ):

    w_Lya = 1215.67
    
    Delta_w = 200.0 #AA
    
    w_line = ( redshift + 1 ) * w_Lya

    mask_line = ( w_Arr > w_line-0.5*Delta_w ) * ( w_Arr < w_line+0.5*Delta_w )
    
    segment_w_Arr = w_Arr[ mask_line ]
    segment_F_Arr = F_Arr[ mask_line ]
    
    p0 = [ w_line, 5.0 , np.amax(segment_F_Arr) ]
    
    popt , pvoc = scipy.optimize.curve_fit( gaussian_f , segment_w_Arr , segment_F_Arr , p0=p0 )
    
    #print popt
    
    center_line , sigma_line , Amplitude_line = popt[0] , popt[1] , popt[2]
    
    w_gaussian_Arr = np.linspace( np.amin(segment_w_Arr) , np.amax(segment_w_Arr) , 1000 )
    
    gaussian_Arr = gaussian_f( w_gaussian_Arr , center_line , sigma_line , Amplitude_line )
    
    # Flux of Lya
    
    flux_Lya = np.trapz( gaussian_Arr , w_gaussian_Arr )
    
    # Noise_level
    
    mask_lineless = ( segment_w_Arr < center_line-4*sigma_line ) + ( segment_w_Arr > center_line-4*sigma_line )
    
    Noise_level = np.std( segment_F_Arr[mask_lineless] )
  
    mask_not_nans = ~np.isnan( F_Arr )
 
    #g_flux = jp.Synthetic_Photometry_measure_flux_simple( w_Arr[ mask_not_nans ] , F_Arr[ mask_not_nans ] , 'gSDSS' )
    g_flux = Synthetic_Photometry_measure_flux_simple( w_Arr[ mask_not_nans ] , F_Arr[ mask_not_nans ] , 'gSDSS' )

    PLOT_EX=False
    if PLOT_EX: 
        Bin = segment_w_Arr[1]-segment_w_Arr[0]
        
        XX_Arr , YY_Arr = plot_a_rebinned_line( segment_w_Arr , segment_F_Arr , Bin )
        
        plot( XX_Arr , YY_Arr )
        
        plot( w_gaussian_Arr , gaussian_Arr )
        
        Noise_Arr = np.ones( len(segment_w_Arr) ) * Noise_level
        
        fill_between( segment_w_Arr , Noise_Arr , -Noise_Arr , color='k' , alpha=0.2)
        
        savefig( 'fit_example'+str(i)+'.pdf' )


    return center_line , sigma_line , Amplitude_line , Noise_level , g_flux
##======================================================#
##======================================================#
##======================================================#
#def Compute_stack_and_prop( SEDs , w_Arr , redshift_Arr , i=0 , PLOT=False , FIT=True ):
#    w_rest_Arr = np.linspace( 800 , 1600 , 10000 )
#
#    SEDs_rest = Ke.convert_SEDs_to_restframe( w_rest_Arr , SEDs , w_Arr , redshift_Arr )
#
#    stack_rest_Arr = np.nanmean( SEDs_rest , axis=0 )
#
#    if FIT :
#        w_min_fit = 1250
#        w_max_fit = 1550
#
#        mask_fitting = ( w_rest_Arr > w_min_fit )*( w_rest_Arr < w_max_fit )
#
#        my_w_Arr = w_rest_Arr[     mask_fitting ]
#        my_f_Arr = stack_rest_Arr[ mask_fitting ]
#
#        mask_OK = ( ~np.isnan( my_f_Arr ) ) * ( np.isfinite( my_f_Arr ) ) * ( my_f_Arr > 0.0 )
#
#        popt , pvoc = Ke.fit_a_spectrum( my_w_Arr[mask_OK] , my_f_Arr[mask_OK] )
#
#        MET , AGE , EXT , log_Amp = popt[0], popt[1], popt[2], popt[3]
#
#
#    if not FIT:
#        MET , AGE , EXT , log_Amp = 0 , 0 ,0 ,0 
#
#
#    #if PLOT:
#
#    #    import matplotlib
#    #    # see http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
#    #    matplotlib.use('Svg')
#    #    
#    #    from pylab import *
#
#    #    log_fitting_spec = Ke.get_a_fitting_spectrum( w_rest_Arr,  MET , AGE , EXT , log_Amp )
#
#    #    figure( figsize=(7,4) )
#
#    #    plot( w_rest_Arr , stack_rest_Arr , lw=0.1 , label='HETDEX' )
#
#    #    plot( w_rest_Arr , 10**log_fitting_spec , lw=3 , label='Best fit')
#
#    #    xlabel( r'$\rm wavelength \; [\AA ]$' , size=20 )
#    #    ylabel( r'$\rm Flux       \; [a.u.]$' , size=20 )
#
#    #    ylim( -0.1 , 1.5 )
#
#    #    plt.subplots_adjust( left = 0.11 , bottom = 0.17 , right = 0.97 , top = 0.98 , wspace=0., hspace=0.)
#
#    #    savefig( 'iteration_stack_'+str(i)+'.pdf' )
#    #    clf()
#
#    return w_rest_Arr , stack_rest_Arr , MET , AGE , EXT , log_Amp
##======================================================#
##======================================================#
##======================================================#
def compute_cumulative( x_Arr , y_Arr ):

    cum_Arr = np.zeros( len(x_Arr) )

    for i in range( 1 , len(x_Arr) ) :

        cum_Arr[i] = cum_Arr[i-1] + y_Arr[i]

    cum_Arr = cum_Arr * 1. / np.amax( cum_Arr)

    return cum_Arr
#======================================================#
#======================================================#
#======================================================#
def generate_random_number_from_distribution( x_Arr , Dist_Arr , N_random ):

    cum_Arr = compute_cumulative( x_Arr , Dist_Arr )

    random_Arr = np.random.rand( N_random )

    my_random_varible_Arr = np.interp( random_Arr , cum_Arr , x_Arr )

    return my_random_varible_Arr
#======================================================#
#======================================================#
#======================================================#
#def generate_random_property( prop_Arr , N_sources ) :
#
#    Dynamical_range = np.amax( prop_Arr ) - np.amin( prop_Arr )
#
#    min_var = np.amin( prop_Arr ) - 0.1 * Dynamical_range
#    max_var = np.amax( prop_Arr ) + 0.1 * Dynamical_range
#
#    H_z , edges_z = np.histogram( prop_Arr , 30 , range=[min_var,max_var] )
#
#    z_centers = 0.5 * ( edges_z[:-1] + edges_z[1:] )
#
#    Smoothing_scale = 0.05 * Dynamical_range
#
#    smoothed_H_z = Ke.gaussian_filter( z_centers , H_z , Smoothing_scale )
#
#    smoothed_H_z = smoothed_H_z * np.trapz( z_centers , H_z ) * 1. / np.trapz( z_centers , smoothed_H_z )
#
#    N_random=N_sources
#
#    RES = 1000
#    high_res_z_centers = np.linspace( np.amin(z_centers) , np.amax(z_centers) , RES)
#
#    high_res_smoothed_H_z = np.interp( high_res_z_centers , z_centers , smoothed_H_z )
#
#    #random_z_Arr = generate_random_number_from_distribution( z_centers , smoothed_H_z , N_random)
#
#    random_z_Arr = generate_random_number_from_distribution( high_res_z_centers , high_res_smoothed_H_z , N_random)
#
#    return z_centers , H_z , smoothed_H_z , random_z_Arr
#======================================================#
#======================================================#
#======================================================#
def generate_spectrum( LINE , my_z , my_ew , my_flux_g , my_widths , my_noises , my_MET , my_AGE , my_EXT , w_Arr , Grid_Dictionary , Noise_w_Arr , Noise_Arr , T_A , T_B , gSDSS_data ):

    t0 = time.time()

    if LINE == 'Lya':
        w_line = 1215.68

    if LINE == 'OII':
        w_line = 0.5 * (3727.092 + 3729.875)

    cat_w_Arr , cat_rest_spectrum = Interpolate_Lines_Arrays_3D_grid_MCMC(
            my_MET, my_AGE, my_EXT, Grid_Dictionary
            )
    obs_frame_spectrum = np.interp(w_Arr, cat_w_Arr * (1 + my_z), cat_rest_spectrum)
    IGM_obs_continum = np.copy( obs_frame_spectrum )

    if LINE == 'Lya':
        redshift_w_Arr = w_Arr * 1. / w_line - 1.
        IGM_T_w_Arr = IGM_TRANSMISSION(redshift_w_Arr, T_A, T_B)
        mask_IGM = w_Arr < w_line * (1+my_z)
        IGM_obs_continum[mask_IGM] = IGM_obs_continum[mask_IGM] * IGM_T_w_Arr[mask_IGM]

    noisy_spectrum = np.random.normal(0.0, my_noises, len(w_Arr))

    NOISE_w = True
    if NOISE_w :
        Noise_in_my_w_Arr = np.interp( w_Arr , Noise_w_Arr , Noise_Arr )

        Delta_w_noise = 50.0 #A

        w_Lya_observed = ( my_z + 1. ) * w_line

        mask_noise_norms = (w_Arr > w_Lya_observed - 0.5*Delta_w_noise)\
                        * (w_Arr < w_Lya_observed + 0.5*Delta_w_noise)

        I_noise_Arr = np.trapz(Noise_in_my_w_Arr[mask_noise_norms],
                w_Arr[mask_noise_norms]) * 1. / Delta_w_noise

        Noise_in_my_w_Arr = my_noises * Noise_in_my_w_Arr * 1. / I_noise_Arr 

        noisy_spectrum = np.random.normal( 0.0 , Noise_in_my_w_Arr , len(w_Arr) )
    
    g_w_Arr = gSDSS_data['lambda_Arr_f']
    g_T_Arr = gSDSS_data[ 'Transmission_Arr_f' ]
    g_w     = gSDSS_data[ 'lambda_pivot'       ]
    g_FWHM  = gSDSS_data[ 'FWHM'               ]

    Noises_flux_g = Synthetic_Photometry_measure_flux(
        w_Arr, noisy_spectrum, g_w_Arr, g_T_Arr, g_w, g_FWHM
        )
    source_flux_g = Synthetic_Photometry_measure_flux(
        w_Arr, IGM_obs_continum, g_w_Arr, g_T_Arr, g_w, g_FWHM
        )

    '''
    ## Synthetic NB arround emission line##
    snb_w_Arr = g_w_Arr
    snb_T_Arr = np.zeros(snb_w_Arr.shape)
    snb_w     = w_line * (1 + my_z)
    snb_T_Arr[np.where(np.abs(snb_w_Arr - snb_w) < 72.)] = 1.
    snb_FWHM  = 144.
    '''

    Noises_flux_snb = Synthetic_Photometry_measure_flux(
            w_Arr, noisy_spectrum, g_w_Arr, g_T_Arr, g_w, g_FWHM
            )
    source_flux_snb = Synthetic_Photometry_measure_flux(
            w_Arr, IGM_obs_continum, g_w_Arr, g_T_Arr, g_w, g_FWHM
            )
    
    Continum_normalization = (my_flux_g - Noises_flux_snb) * 1. / (source_flux_snb)
    
    cont_around_line = (
            Continum_normalization
            * obs_frame_spectrum[np.where(np.abs(w_Arr-w_line*(1+my_z)) <= 6)]
            )
                       
    obs_lya_line_Arr = norm.pdf(w_Arr, w_line * (1 + my_z) , my_widths)
    my_flux_f = np.mean(cont_around_line) * my_ew * (1 + my_z)
    obs_lya_line_Arr = np.absolute(obs_lya_line_Arr * my_flux_f)

    catalog_obs_spectrum_No_IGM = noisy_spectrum + obs_lya_line_Arr + Continum_normalization * obs_frame_spectrum
    catalog_obs_spectrum = noisy_spectrum + obs_lya_line_Arr + Continum_normalization * IGM_obs_continum
    catalog_obs_spectrum_No_Line= noisy_spectrum + Continum_normalization * IGM_obs_continum

    return catalog_obs_spectrum , catalog_obs_spectrum_No_IGM, catalog_obs_spectrum_No_Line
#======================================================#
#======================================================#
#======================================================#

def generate_catalog_properties( N_sources , redshift_Arr , line_fluxes_Arr , line_flux_g_Arr , line_widths_Arr , Noise_level_Arr , MET , AGE , EXT , MET_OII , AGE_OII , EXT_OII , frac_OII ):

    random_Arr = np.random.rand( N_sources )

    MASK_LAEs = random_Arr > frac_OII

    N_LAEs = sum(  MASK_LAEs )
    N_OIIs = sum( ~MASK_LAEs )


    BC_met_Arr , BC_age_Arr , BC_ext_Arr = Ke.get_BC03_prop_Arr()

    Dynamic_range_MET = np.amax( BC_met_Arr ) - np.amin( BC_met_Arr )
    Dynamic_range_AGE = np.amax( BC_age_Arr ) - np.amin( BC_age_Arr )
    Dynamic_range_EXT = np.amax( BC_ext_Arr ) - np.amin( BC_ext_Arr )

    f_scatter = 0.01

    RANDOM = 'GAUSSIAN'
    if RANDOM == 'GAUSSIAN' :

        catalog_MET_Arr = np.zeros( N_sources )
        catalog_AGE_Arr = np.zeros( N_sources )
        catalog_EXT_Arr = np.zeros( N_sources )

        catalog_MET_Arr[  MASK_LAEs ] = np.random.normal( MET    , f_scatter * Dynamic_range_MET , N_LAEs )
        catalog_AGE_Arr[  MASK_LAEs ] = np.random.normal( AGE    , f_scatter * Dynamic_range_AGE , N_LAEs )
        catalog_EXT_Arr[  MASK_LAEs ] = np.random.normal( EXT    , f_scatter * Dynamic_range_EXT , N_LAEs )

        catalog_MET_Arr[ ~MASK_LAEs ] = np.random.normal( MET_OII , f_scatter * Dynamic_range_MET , N_OIIs )
        catalog_AGE_Arr[ ~MASK_LAEs ] = np.random.normal( AGE_OII , f_scatter * Dynamic_range_AGE , N_OIIs )
        catalog_EXT_Arr[ ~MASK_LAEs ] = np.random.normal( EXT_OII , f_scatter * Dynamic_range_EXT , N_OIIs )

    catalog_EXT_Arr[ catalog_EXT_Arr < 0 ] = 1e-10

    _ , _ , _ , catalog_z_Arr = generate_random_property( redshift_Arr , N_sources )

    w_OII = 0.5 * ( 3727.092 + 3729.875 )
    w_Lya = 1215.67

    catalog_z_True_Arr = np.copy( catalog_z_Arr )
 
    catalog_z_True_Arr[ ~MASK_LAEs ] = w_Lya * ( 1 + catalog_z_True_Arr[ ~MASK_LAEs ] ) / w_OII - 1.

    _ , _ , _ , log_random_flux_g_Arr = generate_random_property( np.log10(line_flux_g_Arr) , N_sources )
    _ , _ , _ , log_random_flux_l_Arr = generate_random_property( np.log10(line_fluxes_Arr) , N_sources )
    _ , _ , _ , log_random_widths_Arr = generate_random_property( np.log10(line_widths_Arr) , N_sources )
    _ , _ , _ , log_random_Noises_Arr = generate_random_property( np.log10(Noise_level_Arr) , N_sources )

    catalog_flux_l_Arr = 10**log_random_flux_l_Arr
    catalog_flux_g_Arr = 10**log_random_flux_g_Arr
    catalog_widths_Arr = 10**log_random_widths_Arr
    catalog_Noises_Arr = 10**log_random_Noises_Arr

    catalog_props_dic={}

    catalog_props_dic['LAE'] = MASK_LAEs

    catalog_props_dic['catalog_MET_Arr'] = catalog_MET_Arr
    catalog_props_dic['catalog_AGE_Arr'] = catalog_AGE_Arr
    catalog_props_dic['catalog_EXT_Arr'] = catalog_EXT_Arr

    catalog_props_dic['catalog_z_Arr'] = catalog_z_Arr

    catalog_props_dic['catalog_z_True_Arr'] = catalog_z_True_Arr

    catalog_props_dic['catalog_flux_l_Arr'] = catalog_flux_l_Arr
    catalog_props_dic['catalog_flux_g_Arr'] = catalog_flux_g_Arr
    catalog_props_dic['catalog_widths_Arr'] = catalog_widths_Arr
    catalog_props_dic['catalog_Noises_Arr'] = catalog_Noises_Arr

    return catalog_props_dic
#==========================================================================================#
#==========================================================================================#
#==========================================================================================#
def generate_mock_catalog( N_sources , redshift_Arr , line_fluxes_Arr , line_flux_g_Arr , line_widths_Arr , Noise_level_Arr , w_Arr , Noise_w_Arr , Noise_Arr , MET_Lya , AGE_Lya , EXT_Lya , MET_OII , AGE_OII , EXT_OII  , frac_OII , T_A , T_B ):

    Grid_Dictionary = Ke.Load_BC03_grid_data()

    cat_props = generate_catalog_properties( N_sources , redshift_Arr , line_fluxes_Arr , line_flux_g_Arr , line_widths_Arr , Noise_level_Arr , MET_Lya , AGE_Lya , EXT_Lya , MET_OII , AGE_OII , EXT_OII , frac_OII )

    MASK_LAEs = cat_props[ 'LAE' ]

    catalog_MET_Arr = cat_props['catalog_MET_Arr']
    catalog_AGE_Arr = cat_props['catalog_AGE_Arr']
    catalog_EXT_Arr = cat_props['catalog_EXT_Arr']

    catalog_z_Arr      = cat_props['catalog_z_Arr'      ]
    catalog_z_True_Arr = cat_props['catalog_z_True_Arr' ]

    catalog_flux_l_Arr = cat_props['catalog_flux_l_Arr']
    catalog_flux_g_Arr = cat_props['catalog_flux_g_Arr']
    catalog_widths_Arr = cat_props['catalog_widths_Arr']
    catalog_Noises_Arr = cat_props['catalog_Noises_Arr']

    catalogs_SEDs    = np.zeros( N_sources * len(w_Arr) ).reshape( N_sources , len(w_Arr) )
    catalogs_SEDs_No = np.zeros( N_sources * len(w_Arr) ).reshape( N_sources , len(w_Arr) )

    gSDSS_lambda_Arr_f , gSDSS_Transmission_Arr_f = Load_Filter( 'gSDSS' )

    gSDSS_lambda_pivot , gSDSS_FWHM = FWHM_lambda_pivot_filter( 'gSDSS' )

    gSDSS_data = {}

    gSDSS_data[ 'lambda_Arr_f'       ] = np.copy( gSDSS_lambda_Arr_f       )
    gSDSS_data[ 'Transmission_Arr_f' ] = np.copy( gSDSS_Transmission_Arr_f )
    gSDSS_data[ 'lambda_pivot'       ] = np.copy( gSDSS_lambda_pivot       )
    gSDSS_data[ 'FWHM'               ] = np.copy( gSDSS_FWHM               )

    for i in range( 0 , N_sources ):

        #print '     ' , i , '/'  , N_sources

        #my_z = catalog_z_Arr[i]
        my_z = catalog_z_True_Arr[i]

        my_flux_f = catalog_flux_l_Arr[i]
        my_flux_g = catalog_flux_g_Arr[i]
        my_widths = catalog_widths_Arr[i]
        my_noises = catalog_Noises_Arr[i]

        my_MET = catalog_MET_Arr[i]
        my_AGE = catalog_AGE_Arr[i]
        my_EXT = catalog_EXT_Arr[i]

        if     MASK_LAEs[i] : LINE = 'Lya'
        if not MASK_LAEs[i] : LINE = 'OII'

        tmp_spec , tmp_spec_no_IGM = generate_spectrum( LINE , my_z , my_flux_f , my_flux_g , my_widths , my_noises , my_MET , my_AGE , my_EXT , w_Arr , Grid_Dictionary , Noise_w_Arr , Noise_Arr , T_A , T_B , gSDSS_data )

        catalogs_SEDs[i] = tmp_spec

        catalogs_SEDs_No[i] = tmp_spec_no_IGM


    my_perfect_new_cat = {}

    my_perfect_new_cat['LAE'] = MASK_LAEs

    my_perfect_new_cat['SEDs'] = catalogs_SEDs
    my_perfect_new_cat['SEDs_No_IGM'] = catalogs_SEDs_No

    my_perfect_new_cat['w_Arr'] = w_Arr

    my_perfect_new_cat['MET_Arr'] = catalog_MET_Arr
    my_perfect_new_cat['AGE_Arr'] = catalog_AGE_Arr
    my_perfect_new_cat['EXT_Arr'] = catalog_EXT_Arr

    my_perfect_new_cat['redshift_Arr'] = z_Arr

    my_perfect_new_cat['flux_l_Arr'] = f_Arr
    my_perfect_new_cat['flux_g_Arr'] = g_Arr
    my_perfect_new_cat['widths_Arr'] = s_Arr
    my_perfect_new_cat['Noises_Arr'] = W_Arr

    return my_perfect_new_cat
#======================================================#


#### Function to compute a volume from z interval

def z_volume(z_min, z_max, area):
    dc_max = cosmo.comoving_distance(z_max).to(u.Mpc).value
    dc_min = cosmo.comoving_distance(z_min).to(u.Mpc).value
    d_side_max = cosmo.kpc_comoving_per_arcmin(z_max).to(u.Mpc/u.deg).value * area**0.5
    d_side_min = cosmo.kpc_comoving_per_arcmin(z_min).to(u.Mpc/u.deg).value * area**0.5
    vol = 1./3. * (d_side_max**2*dc_max - d_side_min**2*dc_min)
    return vol

#### Function to calculate EW from line flux

def L_flux_to_g(L_Arr, rand_z_Arr, rand_EW_Arr):
    dL_Arr = cosmo.luminosity_distance(rand_z_Arr).to(u.cm).value
    return 10**L_Arr / ((1 + rand_z_Arr) * rand_EW_Arr * 4*np.pi * dL_Arr**2) 

### Computes EW array from g and L
def L_g_to_ew(L_Arr, g_Arr, z_Arr):
    dL_Arr = cosmo.luminosity_distance(z_Arr).to(u.cm).value
    return 10**L_Arr / ((1 + z_Arr) * g_Arr * 4*np.pi * dL_Arr**2)

def JPAS_synth_phot(SEDs, w_Arr, tcurves):
    phot_len = len(tcurves['tag'])
    pm = np.zeros(phot_len)

    for fil in range(phot_len):
        w = np.array(tcurves['w'][fil])
        t = np.array(tcurves['t'][fil])

        sed_interp = np.interp(w, w_Arr, SEDs)

        sed_int = np.trapz(w * t * sed_interp, w)
        t_int = np.trapz(w * t, w)
        
        pm[fil] = sed_int / t_int
    return pm
