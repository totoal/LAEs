import numpy as np
import matplotlib.pyplot as plt
import colorplot
from my_functions import load_noflag_cat

def perturb_phot(mag_nb, err_nb, mag_bb, err_bb,
                 ewmin, nb_ind, n_iter,
                 bbcut, nbcut,
                 use_curve = False,
                 use_pbp = False,
                 use_noe = False):

    if use_curve and use_pbp and use_noe:
        raise ValueError('Incompatible options')

    sel_hist = np.zeros(len(mag_nb))
    m_bias = np.nanmedian(mag_bb - mag_nb)
    colorcut = colorplot.color_cut(ewmin, nb_ind) + m_bias
    if use_curve:
        x_e = np.linspace(14,26,75)
        err_curve_bb = colorplot.m_err_bin(mag_bb, err_bb, x_e, mag_nb)
        err_curve_nb = colorplot.m_err_bin(mag_nb, err_nb, x_e, mag_nb)
        err_curve_bbnb = np.sqrt(err_curve_bb**2 + err_curve_nb**2)
        err_curve_bbnb = np.interp(mag_nb, x_e, err_curve_bbnb)
    if use_pbp:
        Sigma = 3
        err_arr = Sigma * np.sqrt(err_bb**2 + err_nb**2) + m_bias
    for i in range(n_iter):
        print(str(i + 1) + '/' + str(n_iter), end = '\r')
        new_mag_nb = err_nb * np.random.randn(len(err_nb)) + mag_nb
        new_mag_bb = err_bb * np.random.randn(len(err_bb)) + mag_bb
        new_bbnb = new_mag_bb - new_mag_nb
        if use_curve:
            sel, = np.where((new_bbnb > colorcut) & (new_mag_bb < bbcut)\
                 & (new_mag_nb < nbcut) & (new_bbnb > err_curve_bbnb))
        elif use_pbp:
            sel, = np.where((new_bbnb > colorcut) & (new_mag_bb < bbcut)\
                 & (new_mag_nb < nbcut) & (new_bbnb > err_arr))
        elif use_noe:
            sel, = np.where(new_bbnb > colorcut)
        else:
            sel, = np.where((new_bbnb > colorcut) & (new_mag_bb < bbcut)\
                 & (new_mag_nb < nbcut))
        sel_hist[sel] += 1
    print('')
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

    print('N sources: ' + str(len(nb_m)))

    #Define binning
    m_min = 14
    m_max = 26
    m_bin_n = 75
    x_e = np.linspace(m_min, m_max, m_bin_n)

    bbcut = x_e[np.nanargmin(np.abs(colorplot.m_err_bin(bb_m, bb_e, x_e, bb_m) - 0.24))]
    nbcut = x_e[np.nanargmin(np.abs(colorplot.m_err_bin(nb_m, nb_e, x_e, nb_m) - 0.24))]

    ewmin = 30 # Angstrom

    cand_woec = []
    
    n_iter = 20
    n_bins = 500
    for i in range(n_bins):
        cand_woec.append(perturb_phot(nb_m, nb_e, bb_m, bb_e, ewmin,
                                 nb_ind, n_iter, bbcut, nbcut))

    cand_woec = np.cumsum(cand_woec, axis=0)

    fig, ax = plt.subplots()
    ax.plot(cand_woec[-1], '.', label = 'Not using the error curve')
    ax.legend()
    ax.set_xlabel('source #')
    ax.set_ylabel('N of detections')
    fig.savefig('/home/alberto/cosmos/presentacion29junio/detec2',
            bbox_inches = 'tight', pad_inches = 0, transparent=True)
    plt.show(block = False)
    
    detec_woec = []

    x_bins = np.linspace(0,100,21)
    j = 0
    for row in cand_woec:
        j += 1
        aux = []
        for x in x_bins:
            aux.append(len(row[np.where(row*1./(n_iter*j) *100 >= x)]))
        detec_woec.append(aux)

    detec_woec = np.array(detec_woec)

    fig, ax = plt.subplots()
    for i in range(len(detec_woec)):
        ax.plot(x_bins, detec_woec[i]*1./len(nb_m)*100,
                '.', markersize = 10, label = 'N_iter = %s' %str(i*100))
    ax.set_ylabel('% N')
    ax.set_xlabel('% detections')
    ax.legend()
    fig.savefig('/home/alberto/cosmos/presentacion29junio/detec1',
            bbox_inches = 'tight', pad_inches = 0, transparent=True)
    plt.show(block = False)

    fig, ax = plt.subplots()
    ax.plot((np.arange(n_bins) + 1)*n_iter, detec_woec[:,12]*1./len(nb_m)*100,
            '.', markersize=10)
    ax.set_ylabel('% of detections')
    ax.set_xlabel('N of iterations')
    ax.set_title('Evolution of detections over iterations for 60% of tolerance')
    plt.show()
