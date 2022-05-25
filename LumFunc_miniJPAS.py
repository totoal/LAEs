import numpy as np

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

from my_functions import z_volume, central_wavelength, nb_fwhm
from LF_puricomp_corrections import weights_LF

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

def effective_volume(nb_min, nb_max, survey_name):
    '''
    Due to NB overlap, specially when considering single filters, the volume probed by one
    NB has to be corrected because some sources could be detected in that NB or in either
    of the adjacent ones.
    
    ## Tile_IDs ##
    AEGIS001: 2241
    AEGIS002: 2243
    AEGIS003: 2406
    AEGIS004: 2470
    '''
    
    if survey_name == 'jnep':
        area = 0.24
    elif survey_name == 'minijpas':
        area = 0.895
    elif survey_name == 'both':
        area = 0.24 + 0.895
    else:
        raise ValueError('Survey name not known')

    z_min_overlap = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5) / w_lya - 1
    z_max_overlap = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5) / w_lya - 1

    z_min_abs = (w_central[nb_min - 1] + nb_fwhm_Arr[nb_min - 1] * 0.5) / w_lya - 1
    z_max_abs = (w_central[nb_max + 1] - nb_fwhm_Arr[nb_min + 1] * 0.5) / w_lya - 1

    # volume_abs is a single scalar value in case of 'jnep' and an array of
    # 4 values for each pointing in case of 'minijpas
    volume_abs = z_volume(z_min_abs, z_max_abs, area)
    volume_overlap = (
        z_volume(z_min_overlap, z_min_abs, area)
        + z_volume(z_max_abs, z_max_overlap, area)
    )

    return volume_abs + volume_overlap * 0.5


def LF_perturb_err(L_Arr, L_e_Arr, nice_lya, mag, z_Arr, starprob,
                   bins, puri2d, comp2d, puri2d_err, comp2d_err, L_bins,
                   r_bins, survey_name, tile_id, nb_min, nb_max):
    which_w = [0, 2]
    N_bins = len(bins) - 1

    N_iter = 200
    hist_i_mat = np.zeros((N_iter, N_bins))

    for k in range(N_iter):
        L_perturbed = np.log10(
            10 ** L_Arr + L_e_Arr * np.random.randn(len(L_e_Arr))
        )
        L_perturbed[np.isnan(L_perturbed)] = 0.

        volume = effective_volume(nb_min, nb_max, survey_name)

        # if survey_name == 'jnep':
            # volume_jn = effective_volume(nb_min, nb_max, survey_name)
        # elif survey_name == 'minijpas':
            # volume_mj_1 = effective_volume(nb_min, nb_max, survey_name, 2241)
            # volume_mj_2 = effective_volume(nb_min, nb_max, survey_name, 2243)
            # volume_mj_3 = effective_volume(nb_min, nb_max, survey_name, 2406)
            # volume_mj_4 = effective_volume(nb_min, nb_max, survey_name, 2470)

            # volume_Arr[tile_id == 2241] = volume_mj_1
            # volume_Arr[tile_id == 2243] = volume_mj_2
            # volume_Arr[tile_id == 2406] = volume_mj_3
            # volume_Arr[tile_id == 2470] = volume_mj_4
        # else:
        #     raise ValueError('Survey name not known')

        puri, comp = weights_LF(
            L_perturbed[nice_lya], mag[nice_lya], puri2d, comp2d, puri2d_err, comp2d_err,
            L_bins, r_bins, z_Arr[nice_lya], starprob[nice_lya], tile_id, survey_name,
            which_w, True
        )

        w = np.random.rand(len(puri))
        include_mask = (w < puri)
        w[:] = 1.
        w[~include_mask] = 0.
        w[include_mask] = 1. / comp[include_mask] / volume
        w[np.isnan(w) | np.isinf(w)] = 0.

        hist_i_mat[k], _ = np.histogram(L_perturbed[nice_lya], bins=bins, weights=w)

    L_LF_err_percentiles = np.percentile(hist_i_mat, [16, 50, 84], axis=0)
    return L_LF_err_percentiles