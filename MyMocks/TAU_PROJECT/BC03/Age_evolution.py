import numpy as np

from pylab import *

dir_with_files = 'BC03_sid/'

List_of_files_name = 'BC03_sid/List_of_files'

prop1_Arr = [ 22. , 32. , 42. , 52. , 62. , 72. ]

age_Arr = [ 1.000e-03 , 1.995e-03 , 3.020e-03 , 3.981e-03 , 5.012e-03 , 6.026e-03 , 7.943e-03 ,
            1.000e-02 , 1.318e-02 , 1.585e-02 , 1.995e-02 , 2.512e-02 , 3.200e-02 , 4.000e-02 ,
            5.000e-02 , 6.405e-02 , 8.064e-02 , 1.015e-01 , 1.278e-01 , 1.609e-01 , 2.026e-01 ,
            2.550e-01 , 3.210e-01 , 4.042e-01 , 5.088e-01 , 6.405e-01 , 8.064e-01 , 1.015e+00 ,
            1.278e+00 , 1.609e+00 , 2.000e+00 , 2.500e+00 , 3.250e+00 , 4.000e+00 , 5.000e+00 ,
            6.250e+00 , 8.000e+00 , 1.000e+01 , 1.250e+01 , 1.375e+01 ]

E_Arr = [   0.   , 0.05 , 0.1  , 0.15 , 0.2  ,
            0.25 , 0.3  , 0.35 , 0.4  , 0.45 , 
            0.5  , 0.55 , 0.6  , 0.65 , 0.7  , 
            0.75 , 0.8  , 0.85 , 0.9  , 0.95 , 1.  ]

z_Arr = [ 0. ]

def generate_a_file_name( prop1 , age , E , z ):
    return 'bc2003_hr_m' + str( int(prop1) ) + '_salp_ssp_age' + str(age) + '.outnew_E' + str( E ) + '_z' + str(z ) + '_orig_madau.spec'

cm = get_cmap('rainbow')

for i , age in enumerate( age_Arr ):

    print age

    data = np.loadtxt( dir_with_files + generate_a_file_name( prop1_Arr[0] , age , E_Arr[0] , z_Arr[0]  ) )

    cte = i * 1. / ( len( age_Arr ) - 1 )

    plot( data[:,0] , data[:,1] , color=cm(cte) )


show()
