import os 

import numpy as np

from pylab import *

import Keith as Ke

#===========================================================================#
#===========================================================================#
#===========================================================================#
met_Arr , age_Arr , ext_Arr = Ke.get_BC03_prop_Arr()

print 'Z  ' , np.amin( met_Arr ) , np.amax( met_Arr )
print 'AGE' , np.amin( age_Arr ) , np.amax( age_Arr )
print 'EXT' , np.amin( ext_Arr ) , np.amax( ext_Arr )

#===========================================================================#
#===========================================================================#
#===========================================================================#

Grid_Dictionary = Ke.Load_BC03_grid_data()

Met_value = 47
Ext_value = 0.46

Age_value_Arr = 10**np.linspace( np.log10(0.01) , np.log10(13.0) , 20 )

cm = get_cmap( 'rainbow' )

for i, Age_value in enumerate( Age_value_Arr ):

    cte = i * 1. / ( len(Age_value_Arr) - 1. )

    print Met_value , Age_value , Ext_value

    w_Arr , flux_Arr = Ke.Interpolate_Lines_Arrays_3D_grid_MCMC( Met_value , Age_value , Ext_value , Grid_Dictionary )
    
    plot( w_Arr , flux_Arr , lw=3 , color=cm(cte) )

show() 
