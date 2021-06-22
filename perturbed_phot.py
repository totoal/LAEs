import numpy as np
import matplotlib.pyplot as plt
from my_functions import *
from colorplot import *

def perturb_phot(mag, err, error_curve, x_m):
    new_mag = err * np.random.randn(err.shape) + mag
    new_err = np.interp(new_mag, x_m, error_curve)
    return new_mag, new_err

def selection(bb, nb, bbcut, nbcut, ewmin, nb_ind):
    bbnb = bb - nb
    m_bias = np.nanmedian(bbnb)
    colorcut = color_cut(ewmin, nb_ind) + m_bias

    sel, = np.where((bbnb > colorcut) & (bb < bbcut)\
         & (nb < nbcut))
    return sel

if __name__ == '__main__':
    cat = load_noflag_cat('pkl/catalogDual_pz.pkl')

    nb_ind = 11 # J0480
    bb_ind = -3 # g
    mask_fzero = (cat['MAG'][:, nb_ind] < 90) & (cat['MAG'][:, bb_ind] < 90)

    nb_m = cat['MAG'][mask_fzero, nb_ind]
    bb_m = cat['MAG'][mask_fzero, bb_ind]
    nb_e = cat['ERR'][mask_fzero, nb_ind]
    bb_e = cat['ERR'][mask_fzero, bb_ind]

    #Define binning
    m_min = 14
    m_max = 26
    m_bin_n = 75
    x_e = np.linspace(m_min, m_max, m_bin_n)

    error_curve_nb = m_err_bin(nb_m, nb_e, x_e, nb_m)
    error_curve_bb = m_err_bin(bb_m, bb_e, x_e, nb_m)

    bbcut = x_e[np.nanargmin(np.abs(m_err_bin(bb_m, bb_e, x_e, bb_m) - 0.24))]
    nbcut = x_e[np.nanargmin(np.abs(m_err_bin(nb_m, bb_e, x_e, nb_m) - 0.24))]

    ewmin = 30 # Angstrom
    
    candidates_hist = np.zeros(nb_m.shape)

    for i in range(100):
        new_nb_m, new_nb_e = perturb_phot(nb_m, nb_m, error_curve_nb, x_e)
        new_bb_m, new_bb_e = perturb_phot(bb_m, bb_m, error_curve_bb, x_e)
        
        candidates = selection(new_bb_m, new_nb_m, bbcut, nbcut, ewmin, nb_ind)
        candidates_hist[candidates] += 1

    fig, ax = plt.subplots()
    ax.hist(candidates_hist)
    plt.show()
