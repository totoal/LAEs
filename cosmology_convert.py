import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.cosmology import Planck18 as my_cosmo

def convert_cosmology_luminosity(log_L_Arr, redshift, this_H0, this_Om0, this_Ode0):
    this_cosmo = LambdaCDM(H0=this_H0, Om0=this_Om0, Ode0=this_Ode0)
    
    my_dL_Arr = my_cosmo.luminosity_distance(redshift).value
    this_dL_Arr = this_cosmo.luminosity_distance(redshift).value


    return log_L_Arr - np.log10((my_dL_Arr / this_dL_Arr) ** 2)

def convert_cosmology_Phi(Phi_Arr, redshift, new_H0, new_Om0, new_Ode0):
    this_cosmo = LambdaCDM(H0=new_H0, Om0=new_Om0, Ode0=new_Ode0)

    my_dV = my_cosmo.differential_comoving_volume(redshift).value
    this_dV = this_cosmo.differential_comoving_volume(redshift).value

    return Phi_Arr / my_dV * this_dV