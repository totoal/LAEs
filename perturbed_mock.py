import numpy as np
import matplotlib.pyplot as plt
from my_functions import load_tcurves, load_filter_tags, load_noflag_cat, mag_to_flux
from scipy.interpolate import interp1d
import my_functions
from scipy.integrate import simps
from perturbed_phot import perturb_phot
import colorplot

def photo(tcurves, mock_SEDs, w_Arr, errors):
    
    photo_len = len(tcurves['tag'])
    photo = np.zeros(photo_len)      # Initialize pm vector
    
    for fil in range(0,photo_len):   # For each filter
        
        w = np.array(tcurves['w'][fil])
        t = np.array(tcurves['t'][fil])
        
        f = interp1d(w, t, fill_value='extrapolate')
        
        sed = f(w_Arr)*mock_SEDs
                
        sed_int = simps(w_Arr*sed,w_Arr)
        t_int   = simps(w*t,w)
        
        err = np.random.normal()*errors[fil,1]
        
        photo[fil] = sed_int/t_int + err
        
    return np.array(photo)

global w_central
w_central = my_functions.central_wavelength(load_tcurves(load_filter_tags()))

def mag_error(m,fil,pm):
    w = np.array(w_central)
    ww = w[fil]
    f = mag_to_flux(m, ww)
    if pm == '+':
        return 2.5 * np.log10( (f+3*errors[fil,1]) / f )
    if pm == '-':
        return 2.5 * np.log10( (f-3*errors[fil,1]) / f )
    else:
        print('ERROR: No + or - specified.')


if __name__ == '__main__':
    tcurves = load_tcurves(load_filter_tags())
    '''mock = np.load('Source_cats/Source_cat_2000b.npy', allow_pickle = True).item()
    '''
    errors = np.load('npy/errors5Sigma.npy')
    '''
    LAE_SEDs = mock['SEDs'][mock['LAE']]
    pm = np.zeros((len(LAE_SEDs), len(load_filter_tags())))
    
    for i in range(len(LAE_SEDs)):
        pm[i, :] = photo(tcurves, LAE_SEDs[i], mock['w_Arr'], errors)
        print(str(i), end='\r')
    np.save('npy/pm_mock_LAEs.npy', pm)'''
    pm = np.load('npy/pm_mock_LAEs.npy')
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
    bbcut = x_e[np.nanargmin(np.abs(colorplot.m_err_bin(bb_m, bb_e, x_e, bb_m) - 0.24))]
    nbcut = x_e[np.nanargmin(np.abs(colorplot.m_err_bin(nb_m, nb_e, x_e, nb_m) - 0.24))]
    
    nb_ind = 11
    bb_ind = -3

    mag_nb = pm[:, 11]
    mag_bb = pm[:, -3]

    w_bb = np.array(w_central[bb_ind])
    w_nb = np.array(w_central[nb_ind])
    c = 29979245800

    mag_nb = -2.5*np.log10(mag_nb * w_nb**2/c *1e-8) - 48.60
    mag_bb = -2.5*np.log10(mag_bb * w_bb**2/c *1e-8) - 48.60

    mask_nb = np.invert(np.isnan(mag_nb))
    mag_nb = mag_nb[mask_nb]
    mag_bb = mag_bb[mask_nb]
    mask_bb = np.invert(np.isnan(mag_bb))
    mag_nb = mag_nb[mask_bb]
    mag_bb = mag_bb[mask_bb]
    
    err_nb = mag_error(mag_nb, nb_ind, '+')
    err_bb = mag_error(mag_bb, bb_ind, '+')

    n_iter = 1000
    sel_hist = perturb_phot(mag_nb, err_nb, mag_bb, err_bb,
                            30, nb_ind, n_iter, bbcut, nbcut)

    detec = []

    x_bins = np.linspace(0,100,21)
    for i in x_bins:
        pd = sel_hist*1./n_iter * 100
        detec.append(len(np.where(pd >= i)[0]))

    detec = np.array(detec)*1./len(sel_hist) * 100

    fig, ax = plt.subplots()
    ax.plot(x_bins, detec, '.')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(mag_nb, mag_bb - mag_nb, '.')
    plt.show()
