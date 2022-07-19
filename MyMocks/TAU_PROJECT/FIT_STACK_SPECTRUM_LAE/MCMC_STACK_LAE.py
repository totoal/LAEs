import sys

import numpy as np

import matplotlib
# see http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
# matplotlib.use('Svg')

from pylab import *

#import Keith as Ke

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
# [-1.36344111 41.99792174  0.34791187]
#==============================================================#
#==============================================================#
#==============================================================#


def return_emission_line_catalog():
    Line_list =[[770.409  ,'Ne VIII'    ],[780.324  ,'Ne VIII'    ],[937.814  ,'Ly-epsilon' ],[949.742  ,'Ly-delta'   ],
                [972.02   ,'Ly-gamma'   ],
                [977.030  ,'C III'      ],[989.790  ,'N III'      ],[991.514  ,'N III'      ],[991.579  ,'N III'      ],
                [1025.722 ,'Ly-beta'    ],[1031.912 ,'O VI'       ],[1037.613 ,'O VI'       ],[1066.660 ,'Ar I'       ],
                [1215.670 ,'Ly-alpha'   ],[1238.821 ,'N V'        ],[1242.804 ,'N V'        ],[1260.422 ,'Si II'      ],
                [1264.730 ,'Si II'      ],[1302.168 ,'O I'        ],[1334.532 ,'C II'       ],[1335.708 ,'C II'       ],
                [1393.755 ,'Si IV'      ],[1397.232 ,'O IV]'      ],[1399.780 ,'O IV]'      ],[1402.770 ,'Si IV'      ],
                [1486.496 ,'N IV]'      ],[1548.187 ,'C IV'       ],[1550.772 ,'C IV'       ],[1640.420 ,'He II'      ],
                [1660.809 ,'O III]'     ],[1666.150 ,'O III]'     ],[1746.823 ,'N III]'     ],[1748.656 ,'N III]'     ],
                [1854.716 ,'Al III'     ],[1862.790 ,'Al III'     ],[1892.030 ,'Si III]'    ],[1908.734 ,'C III]'     ],
                [2142.780 ,'N II]'      ],[2320.951 ,'[O III]'    ],[2323.500 ,'C II]'      ],[2324.690 ,'C II]'      ],
                [2648.710 ,'[Fe XI]'    ],[2733.289 ,'He II'      ],[2782.700 ,'[Mg V]'     ],[2795.528 ,'Mg II]'     ],
                [2802.705 ,'Mg II]'     ],[2829.360 ,'[Fe IV]'    ],[2835.740 ,'[Fe IV]'    ],[2853.670 ,'[Ar IV]'    ],
                [2868.210 ,'[Ar IV]'    ],[2928.000 ,'[Mg V]'     ],[2945.106 ,'He I'       ],[3132.794 ,'O III'      ],
                [3187.745 ,'He I'       ],[3203.100 ,'He II'      ],[3312.329 ,'O III'      ],[3345.821 ,'[Ne V]'     ],
                [3425.881 ,'[Ne V]'     ],[3444.052 ,'O III'      ],[3466.497 ,'[N I]'      ],[3466.543 ,'[N I]'      ],
                [3487.727 ,'He I'       ],[3586.320 ,'[Fe VII]'   ],[3662.500 ,'[Fe VI]'    ],[3686.831 ,'H19'        ],
                [3691.551 ,'H18'        ],[3697.157 ,'H17'        ],[3703.859 ,'H16'        ],[3711.977 ,'H15'        ],
                [3721.945 ,'H14'        ],[3726.032 ,'[O II]'     ],[3728.815 ,'[O II]'     ],[3734.369 ,'H13'        ],
                [3750.158 ,'H12'        ],[3758.920 ,'[Fe VII]'   ],[3770.637 ,'H11'        ],[3797.904 ,'H10'        ],
                [3835.391 ,'H9'         ],[3839.270 ,'[Fe V]'     ],[3868.760 ,'[Ne III]'   ],[3888.647 ,'He I'       ],
                [3889.064 ,'H8'         ],[3891.280 ,'[Fe V]'     ],[3911.330 ,'[Fe V]'     ],[3967.470 ,'[Ne III]'   ],
                [3970.079 ,'H-epsilon'  ],[4026.190 ,'He I'       ],[4068.600 ,'[S II]'     ],[4071.240 ,'[Fe V]'     ],
                [4076.349 ,'[S II]'     ],[4101.742 ,'H-delta'    ],[4143.761 ,'He I'       ],[4178.862 ,'Fe II'      ],
                [4180.600 ,'[Fe V]'     ],[4233.172 ,'Fe II'      ],[4227.190 ,'[Fe V]'     ],[4287.394 ,'[Fe II]'    ],
                [4303.176 ,'Fe II'      ],[4317.139 ,'O II'       ],[4340.471 ,'H-gamma'    ],[4363.210 ,'[O III]'    ],
                [4412.300 ,'[Ar XIV]'   ],[4414.899 ,'O II'       ],[4416.830 ,'Fe II'      ],[4452.098 ,'[Fe II]'    ],
                [4471.479 ,'He I'       ],[4489.183 ,'Fe II'      ],[4491.405 ,'Fe II'      ],[4510.910 ,'N III'      ],
                [4522.634 ,'Fe II'      ],[4555.893 ,'Fe II'      ],[4582.835 ,'Fe II'      ],[4583.837 ,'Fe II'      ],
                [4629.339 ,'Fe II'      ],[4634.140 ,'N III'      ],[4640.640 ,'N III'      ],[4641.850 ,'N III'      ],
                [4647.420 ,'C III'      ],[4650.250 ,'C III'      ],[4651.470 ,'C III'      ],[4658.050 ,'[Fe III]'   ],
                [4685.710 ,'He II'      ],[4711.260 ,'[Ar IV]'    ],[4740.120 ,'[Ar IV]'    ],[4861.333 ,'H-beta'     ],
                [4893.370 ,'[Fe VII]'   ],[4903.070 ,'[Fe IV]'    ],[4923.927 ,'Fe II'      ],
                [4958.911 ,'[O III]'    ],[5006.843 ,'[O III]'    ],[5018.440 ,'Fe II'      ],[5084.770 ,'[Fe III]'   ],
                [5145.750 ,'[Fe VI]'    ],[5158.890 ,'[Fe VII]'   ],[5169.033 ,'Fe II'      ],[5176.040 ,'[Fe VI]'    ],
                [5197.577 ,'Fe II'      ],[5200.257 ,'[N I]'      ],[5234.625 ,'Fe II'      ],[5236.060 ,'[Fe IV]'    ],
                [5270.400 ,'[Fe III]'   ],[5276.002 ,'Fe II'      ],[5276.380 ,'[Fe VII]'   ],[5302.860 ,'[Fe XIV]'   ],
                [5309.110 ,'[Ca V]'     ],[5316.615 ,'Fe II'      ],[5316.784 ,'Fe II'      ],[5335.180 ,'[Fe VI]'    ],
                [5424.220 ,'[Fe VI]'    ],[5517.709 ,'[Cl III]'   ],[5537.873 ,'[Cl III]'   ],[5637.600 ,'[Fe VI]'    ],
                [5677.000 ,'[Fe VI]'    ],[5695.920 ,'C III'      ],[5720.700 ,'[Fe VII]'   ],[5754.590 ,'[N II]'     ],
                [5801.330 ,'C IV'       ],[5811.980 ,'C IV'       ],[5875.624 ,'He I'       ],[6046.440 ,'O I'        ],
                [6087.000 ,'[Fe VII]'   ],[6300.304 ,'[O I]'      ],[6312.060 ,'[S III]'    ],[6347.100 ,'Si II'      ],
                [6363.776 ,'[O I]'      ],[6369.462 ,'Fe II'      ],[6374.510 ,'[Fe X]'     ],[6516.081 ,'Fe II'      ],
                [6548.050 ,'[N II]'     ],[6562.819 ,'H-alpha'    ],[6583.460 ,'[N II]'     ],[6716.440 ,'[S II]'     ],
                [6730.810 ,'[S II]'     ],[7002.230 ,'O I'        ],[7005.870 ,'[Ar V]'     ],[7065.196 ,'He I'       ],
                [7135.790 ,'[Ar III]'   ],[7155.157 ,'[Fe II]'    ],[7170.620 ,'[Ar IV]'    ],[7172.000 ,'[Fe II]'    ],
                [7236.420 ,'C II'       ],[7237.260 ,'[Ar IV]'    ],[7254.448 ,'O I'        ],[7262.760 ,'[Ar IV]'    ],
                [7281.349 ,'He I'       ],[7319.990 ,'[O II]'     ],[7330.730 ,'[O II]'     ],[7377.830 ,'[Ni II]'    ],
                [7411.160 ,'[Ni II]'    ],[7452.538 ,'[Fe II]'    ],[7468.310 ,'N I'        ],[7611.000 ,'[S XII]'    ],
                [7751.060 ,'[Ar III]'   ],[7816.136 ,'He I'       ],[7868.194 ,'Ar I'       ],[7889.900 ,'[Ni III]'   ],
                [7891.800 ,'[Fe XI]'    ],[8236.790 ,'He II'      ],[8392.397 ,'Pa20'       ],[8413.318 ,'Pa19'       ],
                [8437.956 ,'Pa18'       ],[8446.359 ,'O I'        ],[8467.254 ,'Pa17'       ],[8498.020 ,'Ca II'      ],
                [8502.483 ,'Pa16'       ],[8542.090 ,'Ca II'      ],[8545.383 ,'Pa15'       ],[8578.700 ,'[Cl II]'    ],
                [8598.392 ,'Pa14'       ],[8616.950 ,'[Fe II]'    ],[8662.140 ,'Ca II'      ],[8665.019 ,'Pa13'       ],
                [8680.282 ,'N I'        ],[8703.247 ,'N I'        ],[8711.703 ,'N I'        ],[8750.472 ,'Pa12'       ],
                [8862.782 ,'Pa11'       ],[8891.910 ,'[Fe II]'    ],[9014.909 ,'Pa10'       ],[9068.600 ,'[S III]'    ],
                [9229.014 ,'Pa9'        ],[9531.100 ,'[S III]'    ],[9545.969 ,'Pa-epsilon' ],[9824.130 ,'[C I]'      ],
                [9850.260 ,'[C I]'      ],[9913.000 ,'[S VIII]'   ],[10027.73 ,'He I'       ],[10031.16 ,'He I'       ],
                [10049.36 ,'Pa-delta],' ],[10286.73 ,'[S II]'     ],[10320.49 ,'[S II]'     ],[10336.41 ,'[S II]'     ],
                [10746.80 ,'[Fe XIII]'  ],[10830.34 ,'He I'       ],[10938.08 ,'Pa-gamma'   ]]

    return Line_list
#==============================================================#
#==============================================================#
#==============================================================#


def mask_spectrum_from_emission_lines(w_Arr, Dw=20, w_min=80., w_max=200000000.):

    Line_list = np.array(return_emission_line_catalog())

    w_line_list = Line_list[:, 0].astype(np.float64)

    mask_lines_to_use = (w_line_list > w_min - Dw) * (w_line_list < w_max + Dw)

    w_line_list = w_line_list[mask_lines_to_use]

    mask_1 = (w_Arr > w_min) * (w_Arr < w_max)

    line_mask = np.ones(len(w_Arr)).astype(bool)

    #print( line_mask )

    for w_line in w_line_list:

        tmp_mask = (w_Arr < w_line - Dw*0.5) + (w_Arr > w_line + Dw*0.5)

        line_mask = line_mask * tmp_mask

    mask_tot = line_mask * mask_1

    return mask_tot
#==============================================================#
#==============================================================#
#==============================================================#


def Linear_3D_interpolator(X_prob, Y_prob, Z_prob, X_grid, Y_grid, Z_grid, Field_in_grid):

    INDEX_X = np.where((X_grid < X_prob))[0][-1]
    INDEX_Y = np.where((Y_grid < Y_prob))[0][-1]
    INDEX_Z = np.where((Z_grid < Z_prob))[0][-1]

    dX_grid = X_grid[INDEX_X + 1] - X_grid[INDEX_X]
    dY_grid = Y_grid[INDEX_Y + 1] - Y_grid[INDEX_Y]
    dZ_grid = Z_grid[INDEX_Z + 1] - Z_grid[INDEX_Z]

    X_min_grid = X_grid[INDEX_X]
    Y_min_grid = Y_grid[INDEX_Y]
    Z_min_grid = Z_grid[INDEX_Z]

    Xprob_X0 = (X_prob - X_min_grid) * 1. / dX_grid
    Yprob_Y0 = (Y_prob - Y_min_grid) * 1. / dY_grid
    Zprob_Z0 = (Z_prob - Z_min_grid) * 1. / dZ_grid

    Vol1 = (1. - Xprob_X0) * (1. - Yprob_Y0) * (1. - Zprob_Z0)
    Vol2 = (1. - Xprob_X0) * (Yprob_Y0) * (1. - Zprob_Z0)
    Vol3 = (1. - Xprob_X0) * (Yprob_Y0) * (Zprob_Z0)
    Vol4 = (1. - Xprob_X0) * (1. - Yprob_Y0) * (Zprob_Z0)

    Vol5 = (Xprob_X0) * (1. - Yprob_Y0) * (1. - Zprob_Z0)
    Vol6 = (Xprob_X0) * (Yprob_Y0) * (1. - Zprob_Z0)
    Vol7 = (Xprob_X0) * (Yprob_Y0) * (Zprob_Z0)
    Vol8 = (Xprob_X0) * (1. - Yprob_Y0) * (Zprob_Z0)

    Field1 = Field_in_grid[INDEX_X, INDEX_Y, INDEX_Z]
    Field2 = Field_in_grid[INDEX_X, INDEX_Y + 1, INDEX_Z]
    Field3 = Field_in_grid[INDEX_X, INDEX_Y + 1, INDEX_Z + 1]
    Field4 = Field_in_grid[INDEX_X, INDEX_Y, INDEX_Z + 1]
    Field5 = Field_in_grid[INDEX_X + 1, INDEX_Y, INDEX_Z]
    Field6 = Field_in_grid[INDEX_X + 1, INDEX_Y + 1, INDEX_Z]
    Field7 = Field_in_grid[INDEX_X + 1, INDEX_Y + 1, INDEX_Z + 1]
    Field8 = Field_in_grid[INDEX_X + 1, INDEX_Y, INDEX_Z + 1]

    Field_at_the_prob_point = Vol1 * Field1 + Vol2 * Field2 + Vol3 * Field3 + \
        Vol4 * Field4 + Vol5 * Field5 + Vol6 * Field6 + Vol7 * Field7 + Vol8 * Field8

    return Field_at_the_prob_point
#==============================================================#
#==============================================================#
#==============================================================#


def Interpolate_Lines_Arrays_3D_grid_MCMC(Met_value, Age_value, Ext_value, Grid_Dictionary):

    Grid_Line = Grid_Dictionary['grid']

    met_Arr_Grid = Grid_Dictionary['met_Arr']
    age_Arr_Grid = Grid_Dictionary['age_Arr']
    ext_Arr_Grid = Grid_Dictionary['ext_Arr']

    w_Arr = Grid_Dictionary['w_Arr']

    aux_line = Linear_3D_interpolator(
        Met_value, Age_value, Ext_value, met_Arr_Grid, age_Arr_Grid, ext_Arr_Grid, Grid_Line)

    return w_Arr, aux_line
#==============================================================#
#==============================================================#
#==============================================================#


def Load_BC03_grid_data():

    path = '/home/alberto/LAEs/MyMocks/TAU_PROJECT/BC03_Interpolation/'

    name = 'data_from_BC03.npy'

    file_name = path + '/' + name

    loaded_model = np.load(file_name, allow_pickle=True,
                           encoding='latin1').item()

    return loaded_model
#==============================================================#
#==============================================================#
#==============================================================#


def Compute_log_like(model_Arr, stack_Arr, stack_err_Arr):

    sigma2 = stack_err_Arr**2

    cc = 1.0

    log_like = -0.5 * np.sum(cc * (model_Arr - stack_Arr)
                             ** 2 / sigma2 + np.log(sigma2))

    return log_like
#======================================================#
#======================================================#
#======================================================#


def prior_f(theta):

    log_AGE, MET, EXT = theta[0], theta[1], theta[2]

    AGE = 10 ** log_AGE

    my_bool = True

    bool_age = (AGE > 10**(-2.55)) * (AGE < 13.00)
    bool_met = (MET > 22.) * (MET < 72)
    bool_ext = (EXT > 0.) * (EXT < 1.0)

    return_bool = my_bool * bool_age * bool_met * bool_ext  # * bool_cte

    return return_bool
#======================================================#
#======================================================#
#======================================================#


def compute_mask_fit(w_stack_Arr):

    w_min_fit = 230
    w_max_fit = 1000700

    mask_line = mask_spectrum_from_emission_lines(
        w_stack_Arr, Dw=20, w_min=w_min_fit, w_max=w_max_fit)

    Dw_OII = 6.0

    w_OII_Arr = [1337.81512605042, 1416.8067226890757,
                 1586.5546218487395, 1526.5]

    for w_line in w_OII_Arr:

        tmp_mask = (w_stack_Arr < w_line - 0.5 * Dw_OII) + \
            (w_stack_Arr > w_line + 0.5 * Dw_OII)

        mask_line = mask_line * tmp_mask

    return mask_line
#======================================================#
#======================================================#
#======================================================#


def normalize_spectrum(w_Arr, flux_Arr, flux_Err=None):

    w_min_red = 1250
    w_max_red = 3350

    mask_red = (w_Arr > w_min_red) * (w_Arr < w_max_red)

    # np.sum( flux_Arr[ mask_red ] )
    Integral = np.percentile(flux_Arr[mask_red], 50)

    new_flux_Arr = flux_Arr * 1. / Integral

    if flux_Err is None:

        return new_flux_Arr

    if not flux_Err is None:

        new_flux_Err = flux_Err * 1. / Integral

        return new_flux_Arr, new_flux_Err
#======================================================#
#======================================================#
#======================================================#


def get_a_fitting_spectrum(w_Arr, Z, AGE, EXT, Grid_Dictionary):

    #Grid_Dictionary = Load_BC03_grid_data()

    w_grid_Arr, flux_Arr = Interpolate_Lines_Arrays_3D_grid_MCMC(
        Z, AGE, EXT, Grid_Dictionary)

    # print('---------------------------------')
    #print( 'w_Arr' , w_Arr )
    #print( 'w_grid_Arr' , w_grid_Arr )
    #print( 'flux_Arr ' , flux_Arr )

    my_flux_Arr = np.interp(w_Arr, w_grid_Arr, flux_Arr)

    #print( 'my_flux_Arr' , my_flux_Arr )
    # print('---------------------------------')

    return my_flux_Arr
#======================================================#
#======================================================#
#======================================================#


def main_f(theta, w_stack_Arr, stack_Arr, stack_err_Arr, Grid_Dictionary, w_int_min, w_int_max):

    if not prior_f(theta):
        return -np.inf

    #print( 'w_stack_Arr' , w_stack_Arr )

    log_AGE, MET, EXT = theta[0], theta[1], theta[2]

    AGE = 10 ** log_AGE

    mask_w = (w_stack_Arr > w_int_min) * (w_stack_Arr < w_int_max)

    mask_line = compute_mask_fit(w_stack_Arr)

    mask_line = mask_line * mask_w

    #print( 'w_stack_Arr[mask_line]' , w_stack_Arr[ mask_line ] )

    f_model_RAW_Arr = get_a_fitting_spectrum(
        w_stack_Arr, MET, AGE, EXT, Grid_Dictionary)

    #print( 'f_model_RAW_Arr' , f_model_RAW_Arr )

    f_obs_Arr, f_err_obs_Arr = normalize_spectrum(
        w_stack_Arr[mask_line], stack_Arr[mask_line], flux_Err=stack_err_Arr[mask_line])
    f_model_Arr = normalize_spectrum(
        w_stack_Arr[mask_line], f_model_RAW_Arr[mask_line])

    # PLOT=True
    # if PLOT :

    #    print( 'AGE , MET , EXT' , AGE , MET , EXT )
    #
    #    figure( figsize=(10,5) )

    #    subplot(121)
    #    plot( w_stack_Arr[mask_line] , f_model_RAW_Arr[mask_line] ,        label='model' )
    #    plot( w_stack_Arr[mask_line] , stack_Arr[mask_line]       , '--' , label='stack' )
    #    legend( loc=0 )

    #    subplot(122)
    #    plot( w_stack_Arr[mask_line] , f_model_Arr , label='model ')
    #    plot( w_stack_Arr[mask_line] , f_obs_Arr , '--' , label='stack' )
    #    legend( loc=0 )

    #    savefig( 'fig_random_' + '.pdf' )
    #    clf()

    ln_like = Compute_log_like(f_model_Arr, f_obs_Arr,  f_err_obs_Arr)

    return ln_like
#======================================================#
#======================================================#
#======================================================#


Grid_Dictionary = Load_BC03_grid_data()

print(Grid_Dictionary.keys())
print(Grid_Dictionary['w_Arr'])

#======================================================#
#======================================================#
#======================================================#
w_Lya = 1215.67
#======================================================#
#======================================================#
#======================================================#
w_int_min = 1350.
w_int_max = 5375.
#======================================================#
#======================================================#
#======================================================#
N_walkers = 100
N_dim = 3
N_steps = 200
N_burn = 200
#======================================================#
#======================================================#
#======================================================#
# Generate a random spectrum to fit!

MET_LAE = 37.
AGE_LAE = 1.2
EXT_LAE = 0.21

print('The target MET is', MET_LAE)
print('The target AGE is', AGE_LAE)
print('The target EXT is', EXT_LAE)

model_w_LAE_Arr, model_f_LAE_Arr = Interpolate_Lines_Arrays_3D_grid_MCMC(
    MET_LAE, AGE_LAE, EXT_LAE, Grid_Dictionary)

print('model_w_LAE_Arr', model_w_LAE_Arr)

print('model_f_LAE_Arr', model_f_LAE_Arr)

model_f_err_LAE_Arr = 0.1 * model_f_LAE_Arr

plot(model_w_LAE_Arr, model_f_LAE_Arr)

fill_between(model_w_LAE_Arr, model_f_LAE_Arr - model_f_err_LAE_Arr,
             model_f_LAE_Arr + model_f_err_LAE_Arr, alpha=0.3)

savefig('fig_test_LAE_spec.pdf')
clf()
#======================================================#
#======================================================#
#======================================================#
# Testing main function

#my_theta = [ np.log10(AGE_LAE) , 37 , 0.21 ]
#
#main_f( my_theta , model_w_LAE_Arr , model_f_LAE_Arr , model_f_err_LAE_Arr , Grid_Dictionary , w_int_min , w_int_max )
#
# sys.exit()
#======================================================#
#======================================================#
#======================================================#
# Generating initial positions for MCMC walkers

theta_0 = np.zeros(N_walkers * N_dim).reshape(N_walkers, N_dim)

log_AGE_min = -2.
log_AGE_max = np.log10(12.99)

MET_min = 22.
MET_max = 72.

EXT_min = 0.0
EXT_max = 1.0

theta_0[:, 0] = np.random.rand(
    N_walkers) * (log_AGE_max - log_AGE_min) + log_AGE_min
theta_0[:, 1] = np.random.rand(N_walkers) * (MET_max - MET_min) + MET_min
theta_0[:, 2] = np.random.rand(N_walkers) * (EXT_max - EXT_min) + EXT_min

#======================================================#
#======================================================#
#======================================================#

args = (model_w_LAE_Arr, model_f_LAE_Arr, model_f_err_LAE_Arr,
        Grid_Dictionary, w_int_min, w_int_max)

sampler = emcee.EnsembleSampler(N_walkers, N_dim, main_f, args=args)

####################################################################
####################################################################
####################################################################
state = sampler.run_mcmc(theta_0, N_burn, progress=True)

sampler.reset()

sampler.run_mcmc(state, N_steps, progress=True)

chains = sampler.get_chain()

flat_samples = np.zeros(N_walkers * N_steps *
                        N_dim).reshape(N_walkers * N_steps, N_dim)

for i in range(0, N_dim):
    flat_samples[:, i] = chains[:, :, i].ravel()

matrix_sol = np.zeros(N_dim)

for i in range(0, N_dim):
    matrix_sol[i] = np.mean(flat_samples[:, i])

my_chains_matrix = np.copy(flat_samples)
####################################################################
####################################################################
####################################################################
ax_list = []

label_list = [r'$\log \; Age \; [Gyr]$',
              r'$\rm Metallicity$', r'$\rm Extintion$']

MAIN_VALUE_mean = np.zeros(N_dim)
MAIN_VALUE_median = np.zeros(N_dim)
MAIN_VALUE_MAX = np.zeros(N_dim)

for i in range(0, N_dim):

    x_prop = my_chains_matrix[:, i]

    x_prop_min = np.percentile(x_prop, 5)
    x_prop_50 = np.percentile(x_prop, 50)
    x_prop_max = np.percentile(x_prop, 95)

    x_min = x_prop_50 - (x_prop_max - x_prop_min) * 1.00
    x_max = x_prop_50 + (x_prop_max - x_prop_min) * 1.00

    mamamask = (x_prop > x_min) * (x_prop < x_max)

    MAIN_VALUE_mean[i] = np.mean(x_prop[mamamask])
    MAIN_VALUE_median[i] = np.percentile(x_prop[mamamask], 50)

    HH, edges_HH = np.histogram(x_prop[mamamask], 100, range=[
                                x_prop_min, x_prop_max])

    #MAIN_VALUE_MAX[ i ] = edges_HH[ np.where((HH==np.amax(HH)))]
####################################################################
####################################################################
####################################################################
dic_dic = {}

dic_dic['chains'] = my_chains_matrix
dic_dic['mean'] = MAIN_VALUE_mean
dic_dic['median'] = MAIN_VALUE_median

save_name = 'mcmc_chains_LAE_Nw_' + str(N_walkers) + '_Nd_' + str(
    N_dim) + '_Ns_' + str(N_steps) + '_Nb_' + str(N_burn) + '.npy'

np.save(save_name, dic_dic)
####################################################################
####################################################################
####################################################################

AGE_LAE_ans = 10**MAIN_VALUE_mean[0]
MET_LAE_ans = MAIN_VALUE_mean[1]
EXT_LAE_ans = MAIN_VALUE_mean[2]

answer_w_LAE_Arr, answer_f_LAE_Arr = Interpolate_Lines_Arrays_3D_grid_MCMC(
    MET_LAE_ans, AGE_LAE_ans, EXT_LAE_ans, Grid_Dictionary)

answe_f_LAE_normed_Arr = normalize_spectrum(answer_w_LAE_Arr, answer_f_LAE_Arr)
model_f_LAE_normed_Arr, model_f_err_normed_Arr = normalize_spectrum(
    model_w_LAE_Arr,  model_f_LAE_Arr, flux_Err=model_f_err_LAE_Arr)

semilogy(answer_w_LAE_Arr, answe_f_LAE_normed_Arr, label='answer')

semilogy(model_w_LAE_Arr, model_f_LAE_normed_Arr, label='model')

fill_between(model_w_LAE_Arr, model_f_LAE_normed_Arr - model_f_err_normed_Arr,
             model_f_LAE_normed_Arr + model_f_err_normed_Arr, alpha=0.3, label='unvertainty')

ylim(1e-6, 1e5)
xlim(w_int_min, w_int_max)

savefig('fig_answer.pdf')
####################################################################
####################################################################
####################################################################
figure(figsize=(15, 15))

for i in range(0, N_dim):

    y_prop = my_chains_matrix[:, i]

    y_prop_min = np.percentile(y_prop, 5)
    y_prop_50 = np.percentile(y_prop, 50)
    y_prop_max = np.percentile(y_prop, 95)

    y_min = y_prop_50 - (y_prop_max - y_prop_min) * 1.00
    y_max = y_prop_50 + (y_prop_max - y_prop_min) * 1.00

    for j in range(0, N_dim):

        if i < j:
            continue

        x_prop = my_chains_matrix[:, j]

        x_prop_min = np.percentile(x_prop, 5)
        x_prop_50 = np.percentile(x_prop, 50)
        x_prop_max = np.percentile(x_prop, 95)

        x_min = x_prop_50 - (x_prop_max - x_prop_min) * 1.00
        x_max = x_prop_50 + (x_prop_max - x_prop_min) * 1.00

        ax = plt.subplot2grid((N_dim, N_dim), (i, j))

        ax_list += [ax]

        DDX = x_max - x_min
        DDY = y_max - y_min

        if i == j:

            H, edges = np.histogram(x_prop, 100, range=[x_min, x_max])

            ax.hist(x_prop, 100, range=[x_min, x_max], color='cornflowerblue')

            ax.plot([MAIN_VALUE_median[i], MAIN_VALUE_median[i]],
                    [0.0, 1e10], 'k--', lw=2)

            ax.set_ylim(0, 1.1 * np.amax(H))

        else:

            XX_min = x_min - DDX * 0.2
            XX_max = x_max + DDX * 0.2

            YY_min = y_min - DDY * 0.2
            YY_max = y_max + DDY * 0.2

            H, edges_y, edges_x = np.histogram2d(x_prop, y_prop, 100, range=[
                                                 [XX_min, XX_max], [YY_min, YY_max]])

            y_centers = 0.5 * (edges_y[1:] + edges_y[:-1])
            x_centers = 0.5 * (edges_x[1:] + edges_x[:-1])

            H_min = np.amin(H)
            H_max = np.amax(H)

            N_bins = 10000

            H_Arr = np.linspace(H_min, H_max, N_bins)[::-1]

            fact_up_Arr = np.zeros(N_bins)

            TOTAL_H = np.sum(H)

            for iii in range(0, N_bins):

                mask = H > H_Arr[iii]

                fact_up_Arr[iii] = np.sum(H[mask]) / TOTAL_H

            H_value_68 = np.interp(0.680, fact_up_Arr, H_Arr)
            H_value_95 = np.interp(0.950, fact_up_Arr, H_Arr)

            pcolormesh(edges_y, edges_x, H.T, cmap='Blues')

            ax.contour(y_centers, x_centers, H.T,
                       colors='k', levels=[H_value_95])
            ax.contour(y_centers, x_centers, H.T,
                       colors='r', levels=[H_value_68])

            X_VALUE = MAIN_VALUE_median[j]
            Y_VALUE = MAIN_VALUE_median[i]

            plot([X_VALUE, X_VALUE], [-100,     100], 'k--', lw=2)
            plot([-100,     100], [Y_VALUE, Y_VALUE], 'k--', lw=2)

            x_plot = [x_min, x_max, x_max, x_min, x_min]
            y_plot = [y_min, y_min, y_max, y_max, y_min]

            ax.plot(x_plot, y_plot, 'w', lw=2)

            ax.set_ylim(y_min-0.05*DDY, y_max+0.05*DDY)

        ax.set_xlim(x_min-0.05*DDX, x_max+0.05*DDX)

        if i == N_dim-1:
            ax.set_xlabel(label_list[j], size=20)

        if j == 0 and i != 0:
            ax.set_ylabel(label_list[i], size=20)
###################################################################
###################################################################
###################################################################
for i in [0, 1, 2]:
    plt.setp(ax_list[i].get_xticklabels(), visible=False)

for i in [0, 2, 4, 5]:
    plt.setp(ax_list[i].get_yticklabels(), visible=False)

plt.subplots_adjust(left=0.085, bottom=0.13, right=0.96,
                    top=0.85, wspace=0.0, hspace=0.0)

savefig('fig_mcmc_corners_CTE_.pdf')
clf()
####################################################################
####################################################################
####################################################################
