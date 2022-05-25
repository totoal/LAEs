import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from my_functions import z_volume

def LumFunc_hist(f_lambda, w_pivot, w_fwhm, n_bins=15, L_min=0, L_max=0,
                 obs_area=0.895):
    w_lya = 1215.67 # A
    z = w_pivot / w_lya - 1
    L_line = w_fwhm * f_lambda * 4*np.pi \
            * (cosmo.luminosity_distance(z).to(u.cm)**2).value
    L_line = np.log10(L_line)
    L_line = L_line[np.invert(np.isnan(L_line))]

    z_max = (w_pivot + 0.5 * w_fwhm) / w_lya - 1
    z_min = (w_pivot - 0.5 * w_fwhm) / w_lya - 1

    volume = z_volume(z_min, z_max, obs_area)

    if L_min == 0 and L_max == 0:
        L_max = np.amax(L_line)
        L_min = np.amin(L_line)

    hist, bins = np.histogram(L_line, bins=np.linspace(L_min, L_max, n_bins + 1))
    bin_width = bins[1] - bins[0]
    bin_centers = np.array([(bins[i + 1] + bins[i]) / 2 for i in range(n_bins)])

    return bin_centers, hist, volume, bin_width

def weights_LF(L_Arr, r_Arr, puri2d, comp2d, L_bins, r_bins):
    w_mat = puri2d / comp2d
    w_mat[np.isnan(w_mat) | np.isinf(w_mat)] = 0.