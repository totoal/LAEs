import numpy as np
import matplotlib.pyplot as plt
from my_functions import *
from colorplot import *

def perturb_phot(mag_nb, err_nb, mag_bb, err_bb,
                 ewmin, nb_ind, n_iter,
                 use_curve = False):
    sel_hist = np.zeros(len(mag_nb))
    m_bias = np.nanmedian(mag_bb - mag_nb)
    colorcut = color_cut(ewmin, nb_ind) + m_bias
    if use_curve:
        x_e = np.linspace(14,26,75)
        err_curve_bb = m_err_bin(mag_bb, err_bb, x_e, mag_nb)
        err_curve_nb = m_err_bin(mag_nb, err_nb, x_e, mag_nb)
        err_curve_bbnb = np.sqrt(err_curve_bb**2 + err_curve_nb**2)
        err_curve_bbnb = np.interp(mag_nb, x_e, err_curve_bbnb)
    for i in range(n_iter):
        print('\b' + str(i) + '/' + str(n_iter))
        new_mag_nb = err_nb * np.random.randn(len(err_nb)) + mag_nb
        new_mag_bb = err_bb * np.random.randn(len(err_bb)) + mag_bb
        bbnb = new_mag_bb - new_mag_nb
        if use_curve:
            sel, = np.where((bbnb > colorcut) & (new_mag_bb < bbcut)\
                 & (new_mag_nb < nbcut) & (bbnb > err_curve_bbnb))
        if not use_curve:
            sel, = np.where((bbnb > colorcut) & (new_mag_bb < bbcut)\
                 & (new_mag_nb < nbcut))
        sel_hist[sel] += 1
    return sel_hist

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

    bbcut = x_e[np.nanargmin(np.abs(m_err_bin(bb_m, bb_e, x_e, bb_m) - 0.24))]
    nbcut = x_e[np.nanargmin(np.abs(m_err_bin(nb_m, nb_e, x_e, nb_m) - 0.24))]

    ewmin = 30 # Angstrom
    
    n_iter = 1000
    cand_wec = perturb_phot(nb_m, nb_e, bb_m, bb_e, ewmin, nb_ind, n_iter, True)
    cand_woec = perturb_phot(nb_m, nb_e, bb_m, bb_e, ewmin, nb_ind, n_iter)

    fig, ax = plt.subplots()
    ax.plot(cand_wec, '.', label = 'Using the error curve')
    ax.plot(cand_woec, '.', label = 'Not using the error curve')
    plt.show(block = False)
    
    detec_wec = []
    detec_woec = []

    for i in np.linspace(0,100,11):
        pd_wec = cand_wec*1./n_iter * 100
        pd_woec = cand_woec*1./n_iter * 100
        detec_wec.append(len(np.where(pd_wec > i)[0]))
        detec_woec.append(len(np.where(pd_woec > i)[0]))

    detec_wec = np.array(detec_wec)
    detec_woec = np.array(detec_woec)

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,100,11), detec_wec*1./len(nb_m)*100,
            '.', label = 'Using the error curve')
    ax.plot(np.linspace(0,100,11), detec_woec*1./len(nb_m)*100,
            '.', label = 'Not using the error curve')
    ax.set_ylabel('N')
    ax.set_xlabel('% detections')
    ax.legend()
    plt.show()
