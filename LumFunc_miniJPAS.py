import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

def LumFunc(f_lambda, w_pivot, w_fwhm, n_bins):
    w_lya = 1215.7 # A
    z = w_pivot/w_lya - 1
    L_line = w_fwhm * f_lambda * 4*np.pi \
            * (cosmo.luminosity_distance(z).to(u.cm)**2).value
    L_line = np.log10(L_line)
    L_line = L_line[np.invert(np.isnan(L_line))]

    dc_max = cosmo.comoving_distance((w_pivot + 0.5*w_fwhm)/w_lya - 1).value
    dc_min = cosmo.comoving_distance((w_pivot - 0.5*w_fwhm)/w_lya - 1).value
    side_d = 60*1000*cosmo.kpc_comoving_per_arcmin(z).value

    volume = (dc_max - dc_min) * side_d**2

    L_max = np.amax(L_line)
    L_min = np.amin(L_line)

    binning = np.linspace(L_min, L_max, n_bins + 1)
    bin_width = (L_max - L_min)*1./n_bins

    hist = []
    for i in len(binning)-1:
        hist.append(len(np.where((L_line >= binning[i]) & (L_line < (binning[i] + bin_width)))[0]))

    print(hist)
    Phi = np.array(hist)/volume/bin_width
    return binning[:-1]+0.5*bin_width, Phi
