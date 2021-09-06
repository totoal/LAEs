import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import erf

def mag_to_flux(m, w):
    c = 29979245800
    return 10**((m + 48.60) / (-2.5)) * c/w**2 * 1e8

def flux_to_mag(f, w):
    c = 29979245800
    return -2.5 * np.log10(f * w**2/c * 1e-8) - 48.60

def load_filter_tags():
    filepath = './JPAS_Transmission_Curves_20170316/minijpas.Filter.csv'
    filters_tags = []

    with open(filepath, mode='r') as csvfile:
        rdlns = csv.reader(csvfile, delimiter=',')

        next(rdlns, None)
        next(rdlns, None)

        for line in rdlns:
            filters_tags.append(line[1])

    filters_tags[0] = 'J0348'

    return filters_tags

def load_tcurves(filters_tags):
    filters_w = []
    filters_trans = []

    for tag in filters_tags:

        filename = './JPAS_Transmission_Curves_20170316/JPAS_'+tag+'.tab'
        f = open(filename, mode='r')
        lines = f.readlines()[12:]
        w = []
        trans = []

        for l in range(len(lines)):
            w.append(float(lines[l].split()[0]))
            trans.append(float(lines[l].split()[1]))

        filters_w.append(w)
        filters_trans.append(trans)


    tcurves = {
        "tag"  :  filters_tags,
        "w"    :  filters_w ,
        "t"    :  filters_trans
    }
    return tcurves

def central_wavelength(tcurves):
    w_central = []

    for fil in range(0,len(tcurves['tag'])):
        w_c = sum(np.array(tcurves['w'][fil])*np.array(tcurves['t'][fil]))     \
               / sum(tcurves['t'][fil])
        w_central.append(w_c)

    return w_central

### FWHM of a curve

def nb_fwhm(tcurves, nb_ind, give_fwhm = False):
    
    t = tcurves['t'][nb_ind]
    w = tcurves['w'][nb_ind]
    
    tmax = np.amax(t)
    
    for i in range(len(w)):
        if t[i] < tmax/2:
            pass
        else:
            w_min = w[i]
            break
            
    for i in range(len(w)):
        if t[-i] < tmax/2:
            pass
        else:
            w_max = w[-i]
            break
            
    if give_fwhm == False:
        return w_max, w_min
    if give_fwhm == True:
        return w_max-w_min

### Load no flag catalog

def load_noflag_cat(filename):
    with open(filename, mode='rb') as file:
        catalog = pickle.load(file)
    
    noflag_cat = {}

    pz        = []
    odds      = []
    mag       = []
    mag_err   = []


    for i in range(catalog['MAG'].shape[0]):
        fsum = sum(catalog['FLAGS'][i] + catalog['MFLAGS'][i])
            
        if fsum == 0:
            mag.append(catalog['MAG'][i])
            mag_err.append(catalog['ERR'][i])
            pz.append(catalog['PHOTOZ'][i])
            odds.append(catalog['PZODDS'][i])
            
    noflag_cat['MAG'] = np.array(mag)
    noflag_cat['ERR'] = np.array(mag_err)
    noflag_cat['W'] = catalog['W_CENTRAL']
    noflag_cat['FILTER'] = catalog['FILTER']
    noflag_cat['PHOTOZ'] = np.array(pz)
    noflag_cat['PZODDS'] = np.array(odds)

    return noflag_cat

def load_flambda_cat(filename):

    cat = {
            'FLAMBDA': np.array([]),
            'RELERR': np.array([]),
    }

    with open(filename, mode='r') as csvfile:
        rdlns = csv.reader(csvfile, delimiter=',')
        next(rdlns, None)
        next(rdlns, None)

        flx = []
        err = []
        flg = []
        mfl = []

        for line in rdlns:
            flx.append(line[0].split())
            err.append(line[1].split())
            flg.append(line[2].split())
            mfl.append(line[3].split())

        flg = np.array(flg).astype(float)
        mfl = np.array(mfl).astype(float)
        flx = np.array(flx).astype(float)
        err = np.array(err).astype(float)
        # Mask sources with flags and negative flux values
        mask_flagged = (np.sum(flg, axis=1) + np.sum(mfl, axis=1)) == 0

        cat['FLAMBDA'] = flx[mask_flagged]
        cat['RELERR'] = err[mask_flagged]

        return cat


## Color plot BB-NB
def plot_bbnb(mock, pm, bb_ind, nb_ind, ew0, plot_error = False):

    islae = mock['LAE']
    EW_Arr = mock['EW_Arr']

    filters_tags = load_filter_tags()
    tcurves = load_tcurves(filters_tags)
    w_central = central_wavelength(tcurves)

    w = np.array(w_central)

    c = 299792458 *100 # c in cgs
    mag = np.zeros(pm.shape)

    for i in range(pm.shape[1]):
        mag[:,i] = -2.5*np.log10(pm[:,i] * w**2/c *1e-8) - 48.60

    print(tcurves['tag'][nb_ind])
    print(tcurves['tag'][bb_ind])

    bb = mag[bb_ind,:] 
    nb = mag[nb_ind,:]
    
    bbnb = bb - nb    

    ## Color cut
    Lya_w = 1215.67
    z = w_central[nb_ind]/Lya_w - 1
    EW     = ew0 * (1+z) # A
    w_max_nb, w_min_nb = nb_fwhm(tcurves, nb_ind)
    fwhm = w_max_nb - w_min_nb
    
    print('z = ' + str(z))
    
    color_cut = 2.5*np.log10(EW/fwhm + 1)

    x_color_cut = np.linspace(15,31,100)
    y_color_cut = np.ones(100) * color_cut
    ##
    
    z_max = w_max_nb/Lya_w - 1
    z_min = w_min_nb/Lya_w - 1
    
    z_Arr = mock['redshift_Lya_Arr']
    
    isz = []
    for z in z_Arr:
        if z < z_max and z > z_min:
            isz.append(True)
        else:
            isz.append(False)
            
    bb_rightz    =    bb[np.where(isz)]
    bbnb_rightz  =  bbnb[np.where(isz)]
    islae_rightz = islae[np.where(isz)]
    
    ##### PLOT ####
    
    plt.figure(figsize=(7,5))

    plt.scatter(bb[np.where(islae == False)],  bbnb[np.where(islae == False)],
                edgecolor = 'purple', facecolor='None')
    plt.scatter(bb[np.where(islae == True )],  bbnb[np.where(islae == True )],
                marker='s', edgecolor='green', facecolor='None')
    
    plt.scatter(bb_rightz[np.where(islae_rightz == False)],
                bbnb_rightz[np.where(islae_rightz == False)],
                edgecolor = 'purple', facecolor='purple')
    plt.scatter(bb_rightz[np.where(islae_rightz == True )],
                bbnb_rightz[np.where(islae_rightz == True )],
                marker='s', edgecolor='green', facecolor='green')
    
    plt.plot(x_color_cut, y_color_cut, 'orange')

    plt.ylabel('bb-nb', size='15')
    plt.xlabel('bb'   , size='15')

    plt.ylim( (-3  ,  3 ) )
    plt.xlim( ( 19 ,  27) )

    plt.show()

def load_mags(nb_ind, bb_ind):
#     nb_ind = 11 # J0480
    bb_ind = -3 # g
    cat = load_noflag_cat('pkl/catalogDual_pz.pkl')

    mask_fzero = (cat['MAG'][:, nb_ind] < 90) & (cat['MAG'][:, bb_ind] < 90)

    nb_m = cat['MAG'][mask_fzero, nb_ind]
    bb_m = cat['MAG'][mask_fzero, bb_ind]
    nb_e = cat['ERR'][mask_fzero, nb_ind]
    bb_e = cat['ERR'][mask_fzero, bb_ind]

    # Define binning
    m_min = 14
    m_max = 26
    m_bin_n = 75
    x_e = np.linspace(m_min, m_max, m_bin_n)

    # SNR=5 cut
    w_central = central_wavelength(load_tcurves(load_filter_tags()))
    errors = np.load('npy/errors5Sigma.npy')
    bbcut = flux_to_mag(errors[bb_ind,1]*5,w_central[bb_ind]) 
    nbcut = flux_to_mag(errors[nb_ind,1]*5,w_central[nb_ind]) 
    # bbcut = x_e[np.nanargmin(np.abs(m_err_bin(bb_m, bb_e, x_e, bb_m) - 0.24))]
    # nbcut = x_e[np.nanargmin(np.abs(m_err_bin(nb_m, nb_e, x_e, nb_m) - 0.24))]
    
    return nb_m, bb_m, nb_e, bb_e, bbcut, nbcut

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
    
def select_sources(nb_ind, bb_ind, min_score, mode = 1):
    bb_ind = -3
    nb_m, bb_m, nb_e, bb_e, bbcut, nbcut = load_mags(nb_ind, bb_ind)

    mu = bb_m - nb_m
    sigma = np.sqrt(bb_e**2 + nb_e**2)
    m_ew = color_cut(30, nb_ind) + np.nanmedian(mu)
    p_bbnb = 0.5 - 0.5*erf((m_ew - mu) / (np.sqrt(2)*sigma)) 
    p_bb = 0.5*erf((bbcut - bb_m) / (np.sqrt(2)*bb_e))\
            - 0.5*erf((0 - bb_m) / (np.sqrt(2)*bb_e))
    p_nb = 0.5*erf((nbcut - nb_m) / (np.sqrt(2)*nb_e))\
            - 0.5*erf((0 - nb_m) / (np.sqrt(2)*nb_e))
        
    p_line = p_bbnb * p_bb * p_nb
    
    if mode == 1:
        selection, = np.where(
              (p_bbnb > erf(min_score/np.sqrt(2)))
            & (nb_m/nb_e > 5)
            & (bb_m/bb_e > 5)
        )
    if mode == 2:
        selection, = np.where(p_line > erf(min_score/np.sqrt(2))**3)
        
    return selection

# Function to compute the NB excess with a linear cont estimate
def nbex_cont_estimate(pm, err, nb_ind, w_central, N_nb, ew0, nb_fwhm):
    if N_nb > nb_ind: raise ValueError('N_nb cannot be larger than nb_ind')
    
    z = 1215.67/w_central[nb_ind] - 1
    ew = ew0*(1 + z)

    filter_ind_Arr = [*range(nb_ind-N_nb,nb_ind), *range(nb_ind+1, nb_ind+N_nb-1)]
    if nb_ind < 12: filter_ind_Arr += [-4]
    if nb_ind < 26: filter_ind_Arr += [-3] # Add the BBs
    if nb_ind > 12: filter_ind_Arr += [-2]
    if nb_ind > 26: filter_ind_Arr += [-1]
    filter_ind_Arr = np.array(filter_ind_Arr)

    # Fitting
    N_sources = len(pm)
    nbex = np.zeros(N_sources)
    f_cont = np.zeros(N_sources)
    cf = np.zeros((N_sources,2))
    cont_err = np.zeros(N_sources)
    for i in range(N_sources):
        print('{}/{}'.format(i+1, N_sources), end='\r')
        pm_mag = pm[i]
        pm_err = err[i]
    
        x = w_central[filter_ind_Arr]
        y = pm_mag[filter_ind_Arr]
        weights = np.zeros(len(filter_ind_Arr))
        errors = np.copy(pm_err)
        if nb_ind < 20:               ref_bb = -3
        if nb_ind > 20 & nb_ind < 35: ref_bb = -2
        if nb_ind > 35:               ref_bb = -1
        for idx in filter_ind_Arr:
            bbnb = pm_mag[idx] - pm_mag[ref_bb] # Excess NB-gSDSS
            if bbnb > 3*pm_err[idx] + ew*pm_mag[-3]/nb_fwhm:
                errors[idx] = 999.
        weights = errors[filter_ind_Arr]
        cf[i,:], cov = np.polyfit(x, y, 1, w = 1./weights, cov = True)
        cont_fit = cf[i,:]
        f_cont[i] = cont_fit[1] + cont_fit[0]*w_central[nb_ind]
        nbex[i] = pm_mag[nb_ind] - f_cont[i]
        cont_err[i] = (cov[1,1] + cov[0,0]*w_central[nb_ind]**2)**0.5
        print(cov)
    
    line = nbex - ew*f_cont/nb_fwhm > 1*(err[:,nb_ind]**2 + cont_err**2)**0.5
    return line, cf, cont_err


## Function that loads from a csv file the DualABMag minijpas catalog with associated pz,
## odds, and GAIA apparent move data.
def load_cat_photoz_gaia(filename):
    with open(filename, mode='r') as csvfile:
        rdlns = csv.reader(csvfile, delimiter=',')
        next(rdlns, None)
        next(rdlns, None)
        
        number = []
        flx = []
        flx_err = []
        flags = []
        mflags = []
        photoz = []
        odds = []
        parallax = []
        parallax_err = []
        pmra = []
        pmra_err = []
        pmdec = []
        pmdec_err = []
        
        for line in rdlns:
            number.append(line[0])
            flx.append(line[1].split())
            flx_err.append(line[2].split())
            flags.append(line[3].split())
            mflags.append(line[4].split())
            photoz.append(line[5])
            odds.append(line[6])
            parallax.append(line[7])
            parallax_err.append(line[8])
            pmra.append(line[9])
            pmra_err.append(line[10])
            pmdec.append(line[11])
            pmdec_err.append(line[12])
            
    columns = [
        number, flx, flx_err,
        flags, mflags, photoz,
        odds, parallax, parallax_err,
        pmra, pmra_err, pmdec,
        pmdec_err
    ]
    cat_keys = [
        'number', 'flx', 'flx_err', 'flags',
        'mflags', 'photoz', 'odds', 'parallax',
        'parallax_err', 'pmra', 'pmra_err',
        'pmdec', 'pmdec_err'
    ]
    cat_types = [
        int, float, float, int,
        int, float, float, float,
        float, float, float, float,
        float
    ]
    cat = {}
    
    for col,key in zip(columns, cat_keys):
        cat[key] = np.array(col)
    # Substitute empty values by NumPy NaN
    for k,t in zip(cat.keys(), cat_types):
        cat[k][np.where(cat[k] == '')] = np.nan
        cat[k] = cat[k].astype(t)
        
    # Remove flagged sources
    flags_arr = np.sum(cat['flags'], axis = 1) + np.sum(cat['mflags'], axis = 1)
    mask_flags = flags_arr == 0
    for k in cat.keys():
        cat[k] = cat[k][mask_flags]
    
    return cat

if __name__ == '__main__':
    cat = load_noflag_cat('catalogDual.pkl')
    print(cat.keys())
