import numpy as np
from pylab import *


wave_arr = np.linspace( 3000 , 11000 , 1000 )

DATA = np.zeros( len(wave_arr) * 6 ).reshape( len(wave_arr) , 6 )

for i in range(1,6) :

    table = np.loadtxt( 'JPAS_iSDSS_'+str(i)+'.tab' )

    print( len(table) )

    plot(table[:,0] , table[:,1])

    DATA[:,i-1] = np.interp( wave_arr , table[:,0] , table[:,1] , right=0 , left=0 )


tranmission = np.median( DATA , axis=1 )

matrix = np.zeros( len(wave_arr) * 2 ).reshape( len(wave_arr) , 2 )

matrix[:,0] = wave_arr
matrix[:,1] = tranmission

np.savetxt( 'JPAS_iSDSS.tab' , matrix )

plot( wave_arr , tranmission, 'o' )

show()
