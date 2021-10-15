import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import simpson 
from scipy.special import erf
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

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

def nb_fwhm(tcurves, nb_ind, give_fwhm = True):
    
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

# Stack estimation
def stack_estimation(pm_flx, pm_err, nb_c, N_nb, w_central):
    '''
    Returns the weighted average and error of N_nb Narrow Bands
    arround the central one.
    '''
    nb_idx_Arr = np.array([*range(nb_c-N_nb, nb_c+N_nb+1)])
    
    IGM_T = IGM_TRANSMISSION(np.array(w_central[nb_c-N_nb:nb_c])).reshape(-1, 1)

    flx = pm_flx[nb_idx_Arr]
    flx[:N_nb] /= IGM_T
    err_i = pm_err[nb_idx_Arr]
    err_i[:N_nb] /= IGM_T
    err_i[N_nb] = 999.
    
    ## First compute the continuum to find outliers to this first estimate
    avg = np.average(flx, axis=0, weights=err_i**-2)
    sigma =  ((len(nb_idx_Arr) - 1) / np.sum(err_i**-2, axis=0))**0.5

    ew0min = 0
    fwhm_nb = nb_fwhm(load_tcurves(load_filter_tags()), nb_c, True)

    # Sigma clipping
    for _ in range(5):
        err = err_i
        bbnb = flx - avg
        bbnb_err = (err**2 + sigma**2)**0.5
        z = (np.array(w_central)[nb_idx_Arr] / 1215.67 + 1).reshape(-1, 1)\
                * np.ones(bbnb.shape)
        outliers = (
                (np.abs(bbnb) > 3*bbnb_err)
                & (np.abs(bbnb) > ew0min * (1 + z) * avg / fwhm_nb)
        )
        out = np.where(outliers)
        out_symmetric = (N_nb - (out[0] - N_nb), out[1])
        err[out] = 999.
        err[out_symmetric] = 999.
        avg = np.average(flx, axis=0, weights=err**-2)
        sigma = ((len(nb_idx_Arr) - 1) / np.sum(err**-2, axis=0))**0.5

    mask = err == 999.
    flx_ma = np.ma.array(flx, mask=mask)
    err_ma = np.ma.array(err**-2, mask=mask)

    ## Now recompute this but with no outliers
    avg = np.array(np.ma.average(flx_ma, weights=err**-2, axis=0))
    sigma =  np.array((1. / err_ma.sum(axis=0))**0.5)
    return avg, sigma

def NB_synthetic_photometry(f, w_Arr, w_c, fwhm):
    '''
    Returns the synthetic photometry of a set f of (N_sources x binning) in a
    central wavelength w_c with a fwhm.
    '''
    synth_tcurve = np.zeros(w_Arr.shape)
    synth_tcurve[np.where(np.abs(w_Arr - w_c) < fwhm*0.5)] += 1.
    T_integrated = simpson(synth_tcurve * w_Arr, w_Arr)

    if len(f.shape) == 1:
        return simpson(synth_tcurve * f * w_Arr, w_Arr) / T_integrated 
    else:
        return simpson(synth_tcurve * f * w_Arr, w_Arr, axis=1) / T_integrated 

def z_volume(z_min, z_max, area):
    '''Computes the comoving volume in an observed area in a range of redshifts'''
    dc_max = cosmo.comoving_distance(z_max).to(u.Mpc).value
    dc_min = cosmo.comoving_distance(z_min).to(u.Mpc).value
    d_side_max = cosmo.kpc_comoving_per_arcmin(z_max).to(u.Mpc/u.deg).value * area**0.5
    d_side_min = cosmo.kpc_comoving_per_arcmin(z_min).to(u.Mpc/u.deg).value * area**0.5
    vol = 1./3. * (d_side_max**2*dc_max - d_side_min**2*dc_min)
    return vol

def IGM_TRANSMISSION(w_Arr, A=-0.001845, B=3.924):
    '''
    Returns the IGM transmission associated with the Lya Break.
    '''
    Transmission_Arr = np.exp(A * (w_Arr / 1215.67)**B)
    return Transmission_Arr

def conf_matrix(line_Arr, z_Arr, nb_c):
    '''
    Confusion matrix of selection.
    Inputs: Bool array of selection (line_Arr), Array with the real redshifts of all
    the lines, nb_c.
    '''
    tcurves = load_tcurves(load_filter_tags())
    w_in = list(nb_fwhm(tcurves, nb_c, False))
    w_in.sort()
    w_in += np.array([-10, 10])
    z_in = np.array([w / 1215.67 - 1 for w in w_in])
    
    TP = len(np.where( line_Arr &  ((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])
    FP = len(np.where( line_Arr & ~((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])
    TN = len(np.where(~line_Arr & ~((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])
    FN = len(np.where(~line_Arr &  ((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])

    return np.array([[TP, FP], [FN, TN]]) 
