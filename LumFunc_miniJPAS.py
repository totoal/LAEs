import numpy as np

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

from LAEs.my_functions import z_volume, central_wavelength, nb_fwhm
from LAEs.LF_puricomp_corrections import weights_LF

w_lya = 1215.67
w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))

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

def LF_perturb_err(corr_L, L_Arr, L_e_Arr, nice_lya, mag, z_Arr, starprob,
                   bins, survey_name, tile_id, which_w=[0, 2],
                   return_puri=False, dirname='', return_hist_i_mat=False,
                   N_iter=500, save_hist_i_mat=True):
    N_bins = len(bins) - 1

    hist_i_mat = np.zeros((N_iter, N_bins))

    puri_list = []

    for k in range(N_iter):
        randN = np.random.randn(len(L_Arr))
        L_perturbed = np.empty_like(L_Arr)
        L_perturbed[randN <= 0] = (L_Arr + L_e_Arr[0] * randN)[randN <= 0]
        L_perturbed[randN > 0] = (L_Arr + L_e_Arr[1] * randN)[randN > 0]
        L_perturbed[np.isnan(L_perturbed)] = 0.
        
        L_perturbed_corr = L_perturbed - corr_L

        puri, comp = weights_LF(
            L_perturbed[nice_lya], mag[nice_lya],
            z_Arr[nice_lya], starprob[nice_lya], tile_id[nice_lya],
            survey_name, dirname, which_w, True
        )

        w = np.random.rand(len(puri))
        include_mask = (w < puri)
        w[:] = 1.
        w[~include_mask] = 0.
        w[include_mask] = 1. / comp[include_mask]
        w[comp < 0.2] = 0. # Mask very low completeness
        w[np.isnan(w) | np.isinf(w)] = 0.

        hist_i_mat[k], _ = np.histogram(L_perturbed_corr[nice_lya], bins=bins, weights=w)
        puri_list.append(puri)
    
    puri = np.mean(puri_list, axis=0)
    
    # Save hist_i_mat
    if save_hist_i_mat:
        np.save(f'{dirname}/hist_i_mat_{survey_name}.npy', hist_i_mat)

    L_LF_err_percentiles = np.percentile(hist_i_mat, [16, 50, 84], axis=0)
    if return_hist_i_mat:
        return hist_i_mat
    if not return_puri:
        return L_LF_err_percentiles
    else:
        return L_LF_err_percentiles, puri