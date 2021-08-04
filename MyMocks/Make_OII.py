import numpy as np
from astropy.cosmology import Planck18 as cosmo
from my_utilities import *
import csv

####    Line wavelengths
w_lya = 1215.67
w_oii = 0.5 * ( 3727.092 + 3729.875 )

####    Mock parameters. MUST BE THE SAME AS IN 'Make_LAE.py'   ####
z_lya = [2, 3.5] # LAE z interval
z_oii = [ w_lya*(z + 1)/w_oii - 1 for z in z_lya] # OII z interval computed from z_lya
obs_area = 1 # deg**2

####################################################################

#####   Load OII LF (Gilbank 2010)

filepath = '../csv/Gilbank2010-OII_LF.csv'
OII_LF = []
with open(filepath, mode='r') as csvfile:
    rdlns = csv.reader(csvfile, delimiter=',')
    
    for line in rdlns:
        OII_LF.append(line)
OII_LF = np.array(OII_LF).astype(float)

####    Compute the number of sources and L_line distribution 

bin_width = OII_LF[1,0] - OII_LF[0,0]
Volume_OII = z_volume(z_oii[0], z_oii[1], obs_area)
N_sources_OII = int(np.sum(OII_LF[:,1]) * bin_width * Volume_OII)
LF_p_cum = np.cumsum(OII_LF[:,1])
LF_p_cum /= np.max(LF_p_cum)
L_Arr = np.interp(
        np.random.rand(100000),
        LF_p_cum, OII_LF[:,0]
        )
print(L_Arr)
