import numpy as np

from pylab import *

f_name = 'Source_cat_100.npy'

mock = np.load( f_name , allow_pickle=True ).item()

print( mock.keys() )

#dict_keys(['LAE', 'SEDs', 'SEDs_No_IGM', 'w_Arr', 'MET_Arr', 'AGE_Arr', 'EXT_Arr', 'redshift_Lya_Arr', 'redshift_Arr', 'flux_l_Arr', 'flux_g_Arr', 'widths_Arr', 'Noises_Arr'])

# LAE              , if 1 then LAE, else OII                                                [ dimensionless ]
# SEDs             , numpy array with all the SEDs, mock['SEDs'][i] is the SED of galaxy i  [ erg/s/A       ]
# SEDs_No_IGM      , numpy array with all the SEDs but without IGM absorption               [ erg/s/A       ]
# w_Arr            , wavelength array where the SEDs is evaluated                           [ Angstroms     ]
# MET_Arr          , metallicty of the stelar population used to generate the SED           [ ????          ]
# AGE_Arr          , age        of the stelar population used to generate the SED           [ Gyr           ]
# EXT_Arr          , extintion of the stelar population used to generate the SED            [ dimensionless ]
# redshift_Lya_Arr , redshift if the line is Lyman-alpha                                    [ dimensionless ]
# redshift_Arr     , true redshift of the source                                            [ dimensionless ]
# flux_l_Arr       , Lyman-alpha line flux used to generate the SED                         [ erg/s ??      ]
# flux_g_Arr       , flux density in the g band                                             [ erg/s/A       ]
# widths_Arr       , width of the Lyman-alpha line                                          [ A             ]
# Noises_Arr       , noise flux density in the SED                                          [ erg/s/A       ]

N_plot = 5

cm = get_cmap( 'rainbow' )

figure( figsize=(20,5) )

for i in range( 0 , N_plot ):

    cte = i * 1. / ( N_plot - 1 )

    semilogy( mock['w_Arr'] , mock['SEDs'][i] , color=cm(cte) , lw=2 )

xlabel( r'$\rm wavelength [\AA]$' , size=20 )
ylabel( r'$\rm flux density [erg/s/\AA]$' , size=20 )

show()
