import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

def LumFunc(f_lambda, w_pivot, w_fwhm, n_bins, L_min = 0, L_max = 0):
    w_lya = 1215.7 # A
    z = w_pivot/w_lya - 1
    L_line = w_fwhm * f_lambda * 4*np.pi \
            * (cosmo.luminosity_distance(z).to(u.cm)**2).value
    L_line = np.log10(L_line)
    L_line = L_line[np.invert(np.isnan(L_line))]

    dc_max = cosmo.comoving_distance((w_pivot + 0.5*w_fwhm)/w_lya - 1).value
    dc_min = cosmo.comoving_distance((w_pivot - 0.5*w_fwhm)/w_lya - 1).value
    
    eff_side_deg = np.sqrt(0.895) # deg
    side_d = eff_side_deg*cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc/u.deg).value

    volume = (dc_max - dc_min) * side_d**2

    if L_min == 0 and L_max == 0:
        L_max = np.amax(L_line)
        L_min = np.amin(L_line)

    binning = np.linspace(L_min, L_max, n_bins + 1)
    bin_width = (L_max - L_min)*1./n_bins

    hist = []
    for i in range(len(binning)-1):
        hist.append(len(np.where((L_line >= binning[i])
            & (L_line < (binning[i] + bin_width)))[0]))

    print(hist)
    Phi = np.array(hist)/volume/bin_width
    error = np.sqrt(hist)/volume/bin_width

    return binning[:-1]+0.5*bin_width, Phi, error

def LumFunc_hist(f_lambda, w_pivot, w_fwhm, n_bins, L_min = 0, L_max = 0):
    w_lya = 1215.7 # A
    z = w_pivot/w_lya - 1
    L_line = w_fwhm * f_lambda * 4*np.pi \
            * (cosmo.luminosity_distance(z).to(u.cm)**2).value
    L_line = np.log10(L_line)
    L_line = L_line[np.invert(np.isnan(L_line))]

    dc_max = cosmo.comoving_distance((w_pivot + 0.5*w_fwhm)/w_lya - 1).value
    dc_min = cosmo.comoving_distance((w_pivot - 0.5*w_fwhm)/w_lya - 1).value
    
    eff_side_deg = np.sqrt(0.895) # deg
    side_d = eff_side_deg*cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc/u.deg).value

    volume = (dc_max - dc_min) * side_d**2

    if L_min == 0 and L_max == 0:
        L_max = np.amax(L_line)
        L_min = np.amin(L_line)

    binning = np.linspace(L_min, L_max, n_bins + 1)
    bin_width = (L_max - L_min)*1./n_bins

    hist = []
    for i in range(len(binning)-1):
        hist.append(len(np.where((L_line >= binning[i])
            & (L_line < (binning[i] + bin_width)))[0]))

    print(hist)
    # Phi = np.array(hist)/volume/bin_width
    error = np.sqrt(hist)/volume/bin_width
    x = binning[:-1]+0.5*bin_width
    return x, hist, volume, bin_width
