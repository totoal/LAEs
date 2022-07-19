import os 

import numpy as np

from pylab import *

#import Keith as Ke

#==============================================================#
#==============================================================#
#==============================================================#
def get_BC03_prop_Arr() :

    path = '/global/users/sidgurung/CHARLOTE_BRUZALE/BC03_sid/'

    bc_list = os.listdir( path )

    metal_Arr = []

    age_Arr = []

    ext_Arr = []

    for file_name in bc_list :

        if file_name.split('_')[0] == 'bc2003' :


            metal = file_name.split('_')[2].strip('m')

            age = file_name.split('_')[5].strip('age').strip('.outnew')

            ext = file_name.split('_')[6].strip('E')


            metal_Arr += [ metal ]

            age_Arr += [ age ]

            ext_Arr += [ ext ]


    metal_Arr = np.array( metal_Arr ).astype( np.float64 )
    age_Arr   = np.array( age_Arr ).astype( np.float64 )
    ext_Arr   = np.array( ext_Arr ).astype( np.float64 )

    metal_Arr = np.unique( metal_Arr )
    age_Arr   = np.unique( age_Arr )
    ext_Arr   = np.unique( ext_Arr )

    return metal_Arr , age_Arr , ext_Arr
#==============================================================#
#==============================================================#
#==============================================================#
def Load_BC03_grid_data():

    path = '/global/users/sidgurung/KEITH/BC03_Interpolation/'

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

met_Arr , age_Arr , ext_Arr = get_BC03_prop_Arr()

print( 'Z  ' , np.amin( met_Arr ) , np.amax( met_Arr ) )
print( 'AGE' , np.amin( age_Arr ) , np.amax( age_Arr ) )
print( 'EXT' , np.amin( ext_Arr ) , np.amax( ext_Arr ) )
#==============================================================#
#==============================================================#
#==============================================================#

Grid_Dictionary = Load_BC03_grid_data()

Met_value = 47
Ext_value = 0.46

Age_value_Arr = 10**np.linspace( np.log10(0.01) , np.log10(13.0) , 20 )

cm = get_cmap( 'rainbow' )

for i, Age_value in enumerate( Age_value_Arr ):

    cte = i * 1. / ( len(Age_value_Arr) - 1. )

    print( Met_value , Age_value , Ext_value )

    w_Arr , flux_Arr = Interpolate_Lines_Arrays_3D_grid_MCMC( Met_value , Age_value , Ext_value , Grid_Dictionary )
    
    plot( w_Arr , flux_Arr , lw=3 , color=cm(cte) )

show() 







