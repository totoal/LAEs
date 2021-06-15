import numpy as np
import matplotlib.pyplot as plt
from my_functions import *
import os
from astropy.stats import median_absolute_deviation as absdev

def m_err_bin(mag, err, x_e, ref_m):

    err_arr = []

    m_step = m_step = (x_e[-1] - x_e[0]) / (len(x_e) - 1)

    for m_i in x_e:
        arr_ind = (np.abs(ref_m - m_i) <= m_step/2)
        e_arr_aux = err[arr_ind]

        bin_err = np.nanmedian(e_arr_aux)

        err_arr.append(bin_err)
    
    return np.array(err_arr)

def m_dev_bin(mag, x_e, ref_m):

    err_arr = []

    m_step = m_step = (x_e[-1] - x_e[0]) / (len(x_e) - 1)

    for m_i in x_e:
        arr_ind = (np.abs(ref_m - m_i) <= m_step/2)
        e_arr_aux = mag[arr_ind]

        bin_err = absdev(e_arr_aux, ignore_nan=True)

        err_arr.append(bin_err)
    
    return np.array(err_arr)

def make_colorplot(nf_cat, bb_ind, nb_ind, x_axis = 'NB', plot = True):
    
    # nb_ind = 12  # i = 12 for J0490
    # bb_ind = -3  # i = -3 for gSDSS

    nb_m = nf_cat['MAG'][:, nb_ind]
    bb_m = nf_cat['MAG'][:, bb_ind]
    nb_e = nf_cat['ERR'][:, nb_ind]
    bb_e = nf_cat['ERR'][:, bb_ind]

    bbnb = bb_m - nb_m

    m_min = 14  # Define binning
    m_max = 26
    m_bin_n = 75
    x_e = np.linspace(m_min, m_max, m_bin_n)

    if x_axis == 'NB':
        ref_m = nb_m
    elif x_axis == 'BB':
        ref_m = bb_m
    else:
        print('Valid x_axis values are \'NB\' or \'BB\'.')
        return

    nb_bin_e = m_err_bin(nb_m, nb_e, x_e, ref_m)
    bb_bin_e = m_err_bin(bb_m, bb_e, x_e, ref_m)
    bbnb_bin_e = np.sqrt(nb_bin_e**2 + bb_bin_e**2)

    m_bias = np.nanmedian(bbnb)

    ## Error Curve

    Sigma = 3
    err_curve = Sigma*np.interp(np.linspace(14,26,100), x_e, bbnb_bin_e) + m_bias
    # Curve computed for every bb_m point
    #err_curve_bbnb = Sigma*np.interp(ref_m, x_e, bbnb_bin_e) + m_bias

    true_err_Arr = np.sqrt(nb_e**2 + bb_e**2)
    true_err_curve = Sigma*true_err_Arr + m_bias
    ##

    ## Color cut
    ew0 = 30
    colorcut = color_cut(ew0, nb_ind) + m_bias
    x_colorcut = np.linspace(14, 30, 100)
    y_colorcut = np.ones(100) * colorcut
    ##

    ## Magnitude cut
    bbcut = x_e[np.nanargmin(np.abs(m_err_bin(bb_m, bb_e, x_e, bb_m) - 0.24))]
    nbcut = x_e[np.nanargmin(np.abs(m_err_bin(nb_m, bb_e, x_e, nb_m) - 0.24))]
    ##

    selection, = np.where((bbnb > colorcut) & (bb_m < bbcut)\
         & (bbnb > true_err_curve) & (nb_m < nbcut))

    if plot == False: return selection

    ## PLOT ##
    plt.figure(figsize=(10,3))

    plt.plot(np.linspace(14,26,100), err_curve, color='orange', label='Eror curve')
    plt.plot(x_colorcut, y_colorcut, color = 'red', label='EW$_0$ = '+str(ew0)+' $\AA$')
    
    if x_axis == 'NB':
        plt.plot(nb_m, bbcut - nb_m, c='black', label='BB cut')
        plt.scatter(nb_m, bb_m-nb_m, marker='.')
        plt.axvline(x = nbcut, color = 'green', label='NB cut')
    if x_axis == 'BB':
        plt.plot(bb_m, bb_m - nbcut, c='green', label='NB cut')
        plt.scatter(bb_m, bb_m-nb_m, marker='.')
        plt.axvline(x = bbcut, color = 'black', label='BB cut')

    plt.scatter(ref_m[selection], bbnb[selection], marker='.', c='saddlebrown')

    plt.ylim((-1, 3))
    plt.xlim((14,24))
    
    plt.ylabel('BB - NB', size=10)

    if x_axis == 'NB':
        plt.xlabel('NB', size=10)
    if x_axis == 'BB':
        plt.xlabel('BB', size=10)

    filters_tags = load_filter_tags()

    w_central = central_wavelength(load_tcurves(filters_tags))

    if x_axis == 'NB':
        filter_name = 'NB' + str(filters_tags[nb_ind]) + 'S' + str(Sigma)
    if x_axis == 'BB':
        filter_name = 'BB' + str(filters_tags[nb_ind]) + 'S' + str(Sigma)

    plt.legend()

    try:
        os.mkdir('./miniJPAS_photometry/' + filter_name)
    except:
        print('Directory ' + filter_name + ' already exists.')

    plt.savefig('./miniJPAS_photometry/'+ filter_name +'/color_diagram.png',
                bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    return selection

def plot_selection(selection, nb_ind, x_axis = 'NB'):

    filters_tags = load_filter_tags()
    tcurves = load_tcurves(filters_tags)
    w_central = central_wavelength(load_tcurves(filters_tags))

    bb_fwhm = [
        nb_fwhm(tcurves, -4, True),
        nb_fwhm(tcurves, -3, True),
        nb_fwhm(tcurves, -2, True),
        nb_fwhm(tcurves, -1, True)
    ]

    print('N sources: ' + str(len(selection)))

    Sigma = 3

    j = 0

    for i in selection:
        if j == 50: break

        j += 1

        pm = nf_cat['MAG'][i, :]
        pm_e = nf_cat['ERR'][i, :]

        plt.figure(figsize=(5,5))

        plt.errorbar(
            w_central[:-3], pm[:-3], yerr = pm_e[:-3],
            fmt='none', zorder=-1
        )
        plt.errorbar(
            w_central[-4], pm[-4], yerr = pm_e[-4], c='purple', 
            fmt='none', zorder=-1
        )
        plt.errorbar(
            w_central[-3], pm[-3], yerr = pm_e[-3], c='green', 
            fmt='none', zorder=-1
        )
        plt.errorbar(
            w_central[-2], pm[-2], yerr = pm_e[-2], c='red', 
            fmt='none', zorder=-1
        )
        plt.errorbar(
            w_central[-1], pm[-1], yerr = pm_e[-1], c='dimgray', 
            fmt='none', zorder=-1
        )

        plt.errorbar(
            w_central[-4], pm[-4], xerr=bb_fwhm[-4]/2, fmt='none', 
            color='purple' , elinewidth=4
        )
        plt.errorbar(
            w_central[-3], pm[-3], xerr=bb_fwhm[-3]/2, fmt='none', 
            color='green'  , elinewidth=4
        )
        plt.errorbar(
            w_central[-2], pm[-2], xerr=bb_fwhm[-2]/2, fmt='none', 
            color='red'    , elinewidth=4
        )
        plt.errorbar(
            w_central[-1], pm[-1], xerr=bb_fwhm[-1]/2, fmt='none', 
            color='dimgray', elinewidth=4
        )

        plt.scatter(w_central[:-3], pm[:-3])
        plt.scatter(w_central[-4], pm[-4], c='purple' , marker='s')
        plt.scatter(w_central[-3], pm[-3], c='green'  , marker='s')
        plt.scatter(w_central[-2], pm[-2], c='red'    , marker='s')
        plt.scatter(w_central[-1], pm[-1], c='dimgray', marker='s')

        plt.scatter(w_central[nb_ind], pm[nb_ind], c='black')

        photoz_txt = (
            'PHOTOZ = ' + str(nf_cat['PHOTOZ'][i])
            + '\nODDS = ' + str(nf_cat['PZODDS'][i])
            )
        plt.text(3500, 19, photoz_txt)

        plt.ylim((17,27))

        plt.gca().invert_yaxis()

        plt.ylabel('m$_{AB}$', size = 15)
        plt.xlabel('$\lambda$ ($\AA$)', size = 15)

        if x_axis == 'NB':
            filter_name = 'NB' + str(filters_tags[nb_ind]) + 'S' + str(Sigma)
        if x_axis == 'BB':
            filter_name = 'BB' + str(filters_tags[nb_ind]) + 'S' + str(Sigma)

        plt.savefig('./miniJPAS_photometry/' + filter_name + '/pm' + str(i),
                    bbox_inches = 'tight', pad_inches = 0)
        plt.close()    

#Color cut
def color_cut(ew0, nb_ind):
    tcurves = load_tcurves(load_filter_tags())
    w_central = central_wavelength(tcurves)
    w = w_central[nb_ind]
    Lya_w = 1215.67
    z = w/Lya_w - 1
    EW = ew0 * (1+z)  # A
    t    = np.array(tcurves['t'][nb_ind])
    w_nb = np.array(tcurves['w'][nb_ind])
    T_lambda = t[np.argmax(t)]
    w_t_max = w_nb[np.argmax(t)]
    beta = (T_lambda * w_t_max) / simps(t*w_nb, w_nb)

    color_cut = 2.5*np.log10(EW*beta + 1)

    return color_cut


if __name__ == '__main__':
    nf_cat = load_noflag_cat('catalogDual_pz.pkl')

    for nb_ind in [9, 10, 11, 12, 13]:
        selection = make_colorplot(nf_cat, -3, nb_ind, 'BB', True)
        plot_selection(selection, nb_ind, 'BB')
        selection = make_colorplot(nf_cat, -3, nb_ind, 'NB', True)
        plot_selection(selection, nb_ind, 'NB')
        print('-')
