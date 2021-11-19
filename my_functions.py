import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.integrate import simpson 
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.table import Table
import time

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

def central_wavelength():
    data_tab = Table.read('fits/FILTERs_table.fits', format='fits')
    w_central = data_tab['wavelength']

    return np.array(w_central)

### FWHM of a curve

def nb_fwhm(nb_ind, give_fwhm = True):
    '''
    Returns the FWHM of a filter in tcurves if give_fwhm is True. If it is False, the
    function returns a tuple with (w_central - fwhm/2, w_central + fwhm/2)
    '''
    data_tab = Table.read('fits/FILTERs_table.fits', format='fits')
    w_central = data_tab['wavelength'][nb_ind]
    fwhm = data_tab['width'][nb_ind]
            
    if give_fwhm == False:
        return w_central + fwhm / 2, w_central - fwhm / 2
    if give_fwhm == True:
        return fwhm

# Stack estimation
def stack_estimation(pm_flx, pm_err, nb_c, N_nb, IGM_T_correct=True):
    '''
    Returns the weighted average and error of N_nb Narrow Bands
    arround the central one.
    '''
    w_central = central_wavelength()
    nb_idx_Arr = np.array([*range(nb_c-N_nb, nb_c+N_nb+1)])
    
    if IGM_T_correct:
        IGM_T = IGM_TRANSMISSION(np.array(w_central[nb_c-N_nb:nb_c])).reshape(-1, 1)
    else:
        IGM_T = 1.

    flx = pm_flx[nb_idx_Arr]
    flx[:N_nb] /= IGM_T
    err_i = pm_err[nb_idx_Arr]
    err_i[:N_nb] /= IGM_T

    err_i[N_nb - 1 : N_nb + 2] = 999.

    ## Let's discard NB too faint. flux because in the mocks they have small errs and
    ## that's wrong
    zero_mask = np.where(flx < 1e-20)
    zero_mask_symmetric = (N_nb - (zero_mask[0] - N_nb), zero_mask[1])
    err_i[zero_mask] = 999.
    err_i[zero_mask_symmetric] = 999.

    err = err_i
    
    ## First compute the continuum to find outliers to this first estimate
    avg = np.average(flx, axis=0, weights=err_i**-2)
    sigma =  ((len(nb_idx_Arr) - 1) / np.sum(err_i**-2, axis=0))**0.5

    mask = err == 999.
    flx_ma = np.ma.array(flx, mask=mask)
    err_ma = np.ma.array(err**-2, mask=mask)

    ## Now recompute this but with no outliers
    avg = np.array(np.ma.average(flx_ma, weights=err**-2, axis=0))
    sigma =  np.array((1. / err_ma.sum(axis=0))**0.5)
    return avg, sigma

def estimate_continuum(NB_flx, NB_err, N_nb=6, IGM_T_correct=True):
    '''
    Returns a matrix with the continuum estimate at any NB in all sources.
    '''
    NB_flx = NB_flx[:56]
    NB_err = NB_err[:56]

    cont_est = np.ones(NB_flx.shape) * 99.
    cont_err = np.ones(NB_flx.shape) * 99.
    w_central = central_wavelength()


    for nb_idx in range(1, NB_flx.shape[0]):
        if nb_idx < N_nb:
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                        np.array(w_central[: nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            NBs_to_avg = np.vstack((
                NB_flx[: nb_idx - 1] / IGM_T,
                NB_flx[nb_idx + 2 : nb_idx + N_nb + 1]
            ))
            NBs_errs = np.vstack((
                NB_err[: nb_idx - 1] / IGM_T,
                NB_err[nb_idx + 2 : nb_idx + N_nb + 1]
            ))

        if N_nb <= nb_idx < (NB_flx.shape[0] - 6):
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                        np.array(w_central[nb_idx - N_nb : nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            NBs_to_avg = np.vstack((
                NB_flx[nb_idx - N_nb : nb_idx - 1] / IGM_T,
                NB_flx[nb_idx + 2 : nb_idx + N_nb + 1]
            ))
            NBs_errs = np.vstack((
                NB_err[nb_idx - N_nb : nb_idx - 1] / IGM_T,
                NB_err[nb_idx + 2 : nb_idx + N_nb + 1]
            ))

        if nb_idx >= (NB_flx.shape[0] - 6):
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                        np.array(w_central[nb_idx - N_nb : nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            NBs_to_avg = np.vstack((
                NB_flx[nb_idx - N_nb : nb_idx - 1] / IGM_T,
                NB_flx[nb_idx + 2 :]
            ))
            NBs_errs = np.vstack((
                NB_err[nb_idx - N_nb : nb_idx - 1] / IGM_T,
                NB_err[nb_idx + 2 :]
            ))

        cont_est[nb_idx] = np.average(NBs_to_avg, weights=NBs_errs ** -2, axis=0)
        cont_err[nb_idx] = np.sum(NBs_errs ** -2, axis=0) ** -0.5
    return cont_est, cont_err


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
    vol = 1./3. * (d_side_max**2 * dc_max - d_side_min**2 * dc_min)
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
    w_in = list(nb_fwhm(nb_c, give_fwhm=False, tcurves=tcurves))
    w_in.sort()
    w_in += np.array([-10, 10])
    z_in = np.array([w / 1215.67 - 1 for w in w_in])
    
    TP = len(np.where( line_Arr &  ((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])
    FP = len(np.where( line_Arr & ~((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])
    TN = len(np.where(~line_Arr & ~((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])
    FN = len(np.where(~line_Arr &  ((z_in[0] < z_Arr) & (z_Arr < z_in[1])))[0])

    return np.array([[TP, FP], [FN, TN]]) 

def plot_JPAS_source(flx, err, set_ylim=True):
    '''
    Generates a plot with the JPAS data.
    '''

    data_tab = Table.read('fits/FILTERs_table.fits', format='fits')
    cmap = data_tab['color_representation'][:-4]
    w_central = data_tab['wavelength']
    fwhm_Arr = data_tab['width']

    data_max = np.max(flx)
    data_min = np.min(flx)
    y_max = (data_max - data_min) * 2/3 + data_max
    y_min = data_min - (data_max - data_min) * 2/3

    ax = plt.gca()
    for i, w in enumerate(w_central[:-4]):
        ax.errorbar(w, flx[i], yerr=err[i],
            marker='o', markeredgecolor='dimgray', markerfacecolor=cmap[i],
            markersize=8, ecolor='dimgray', capsize=4, capthick=1, linestyle='',
            zorder=-99)
    ax.errorbar(w_central[-4], flx[-4], yerr=err[-4],
        xerr=fwhm_Arr[-4] / 2,
        fmt='none', color='purple', elinewidth=5)
    ax.errorbar(w_central[-3], flx[-3], yerr=err[-3],
        xerr=fwhm_Arr[-3] / 2,
        fmt='none', color='green', elinewidth=5)
    ax.errorbar(w_central[-2], flx[-2], yerr=err[-2],
        xerr=fwhm_Arr[-2] / 2,
        fmt='none', color='red', elinewidth=5)
    ax.errorbar(w_central[-1], flx[-1], yerr=err[-1],
        xerr=fwhm_Arr[-1] / 2,
        fmt='none', color='saddlebrown', elinewidth=5)

    if set_ylim: ax.set_ylim((y_min, y_max))

    ax.set_xlabel('$\lambda\ (\AA)$', size=15)
    ax.set_ylabel('$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', size=15)

    return ax

def identify_lines(line_Arr, qso_flx, qso_err, nb_min=0, first=False):
    '''
    Returns a list of N lists with the index positions of the lines.

    Input: 
    line_Arr: Bool array of 3sigma detections in sources. Dim N_filters x N_sources
    qso_flx:  Flambda data
    nb_min
    '''
    N_fil, N_src = line_Arr.shape
    line_list = []

    for src in range(N_src):
        fil = 0
        this_src_lines = []
        while fil < N_fil:
            this_line = []
            while ~line_Arr[fil, src]:
                fil += 1
                if fil == N_fil - 1: break
            if fil == N_fil - 1: break
            while line_Arr[fil, src]:
                this_line.append(fil)
                fil += 1
                if fil == N_fil - 1: break
            if fil == N_fil - 1: break
            
            if first:
                this_src_lines.append(
                    np.average(
                        np.array(this_line) - len(this_line) + nb_min + fil,
                        weights=qso_err[np.array(this_line), src]**-2
                    )
                    
                )
            if not first:
                this_src_lines.append(
                    np.argmax(qso_flx[np.array(this_line) + nb_min, src])
                    - len(this_line) + nb_min + fil
                )
        
        if first:
            try:
                line_list.append(this_src_lines[0])
            except:
                line_list.append(-1)
        if not first:
            line_list.append(this_src_lines)
    return line_list

def mask_proper_motion(cat):
    '''
    Masks sources with significant proper motion measurement in Gaia
    '''
    parallax_sn = np.abs(cat['parallax'] / cat['parallax_error'])
    pmra_sn = np.abs(cat['pmra'] / cat['pmra_error'])
    pmdec_sn = np.abs(cat['pmdec'] / cat['pmdec_error'])
    mask = (
        (np.sqrt(parallax_sn ** 2 + pmra_sn ** 2 + pmdec_sn**2) < 27 ** 0.5)
        | (np.isnan(parallax_sn) | np.isnan(pmra_sn) | np.isnan(pmdec_sn))
    )
    return mask.to_numpy()

def is_there_line(pm_flx, pm_err, cont_est, cont_err, ew0min, mask=True):
    w_central = central_wavelength()[:-4]
    fwhm_Arr = nb_fwhm(range(56)).reshape(-1, 1)
    z_nb_Arr = (w_central / 1215.67 - 1).reshape(-1, 1)
    line = (
        # 3-sigma flux excess
        (
            pm_flx[:-4] - cont_est > 3 * (pm_err[:-4]**2 + cont_err**2) ** 0.5
        )
        # EW0 min threshold
        & (
            pm_flx[:-4] - cont_est > ew0min * (1 + z_nb_Arr) * cont_est / fwhm_Arr
        )
        # S/N > 5 on the selected band
        & (
            pm_flx[:-4] / pm_err[:-4] > 5
        )
        # Masks
        & (
            mask
        )
    )
    return line

def QSO_find_lines(qso_flx, qso_err, nb_c_min=6, nb_c_max=50,
    ew0min_lya=30, ew0min_other=15, N_nb=6):
    '''
    Input:
    qso_flx - Matrix of flambda (N_filters x N_sources)
    qso_err - Matrix of flambda errors
    nb_c_min - First filter to look at
    nb_c_max - Last filter to look at
    ew0min_lya - Rest-frame EW minimum for the Lya line
    ew0min_other - Rest-frame EW minimum for all other lines
    N_nb - Number of NB to use on each side to estimate the continuum

    Output:
    nice_lya_list - List of sources compatible with LAE QSOs
    '''
    t0 = time.time()
    data_tab = Table.read('fits/FILTERs_table.fits', format='fits')
    w_central = data_tab['wavelength']
    fwhm_Arr = data_tab['width']

    N_sources = qso_flx.shape[1]

    # Line rest-frame wavelengths (Angstroms)
    w_lya = 1215.67
    w_SiIV = 1397.61
    w_CIV = 1549.48
    w_CIII = 1908.73
    w_MgII = 2799.12
    
    line_qso_lya = np.zeros((nb_c_max - nb_c_min, N_sources)).astype(bool)
    line_qso_other = np.zeros((nb_c_max - nb_c_min, N_sources)).astype(bool)

    # With this we obtain the first line with the ew0min_lya given, for each source
    cont_est_Arr = []
    cont_err_Arr = []
    i = 0
    for nb_c in range(nb_c_min, nb_c_max):
        z_nb = w_central[nb_c] / w_lya - 1
        fwhm = fwhm_Arr[nb_c]
        cont_est_qso, cont_err_qso = stack_estimation(
            qso_flx, qso_err, nb_c, N_nb, False
        )

        line_qso_lya[i] = (
            (qso_flx[nb_c] - cont_est_qso > 3 * (cont_err_qso**2 + qso_err[nb_c]**2)\
                ** 0.5 )
            & (qso_flx[nb_c] - cont_est_qso > ew0min_lya*(1 + z_nb) * cont_est_qso/fwhm)
        )
        cont_est_Arr.append(cont_est_qso)
        cont_err_Arr.append(cont_err_qso)
        i += 1
    line_list_lya = identify_lines(
        line_qso_lya, qso_flx, nb_c_min, first=True
    )
    print('Lya list done. ({0:0.1f} s)'.format(time.time() - t0))
    t0 = time.time()

    # Now we compute the redshift array assuming the first line is Lya
    z_nb_Arr = np.ones(N_sources) * 999 # 999 means no line here, so no z
    for src in range(N_sources):
        l_lya = line_list_lya[src]
        z_nb_Arr[src] = w_central[l_lya] / w_lya - 1

    # Get the line positions array with ew0min_other
    i = 0
    for nb_c in range(nb_c_min, nb_c_max):
        fwhm = fwhm_Arr[nb_c]
        cont_est_qso = cont_est_Arr[i]
        cont_err_qso = cont_err_Arr[i]

        line_qso_other[i] = (
            (qso_flx[nb_c] - cont_est_qso > 3 * (cont_err_qso**2 + qso_err[nb_c]**2)\
                ** 0.5 )
            & (qso_flx[nb_c] - cont_est_qso > ew0min_other*(1 + z_nb_Arr)\
                * cont_est_qso / fwhm)
        )
        i += 1
    line_list_other = identify_lines(line_qso_other, qso_flx, nb_c_min)
    print('Other lines list done. ({0:0.1f} s)'.format(time.time() - t0))
    t0 = time.time()

    # Time to check if the lines are compatible with QSOs
    nice_lya_list = []
    nice_lya_list_single = []
    for src in range(N_sources):
        l_lya = line_list_lya[src]
        if l_lya == -1: continue
        z_src = z_nb_Arr[src]

        w_obs_lya = (1 + z_src) * w_lya
        w_obs_SiIV = (1 + z_src) * w_SiIV
        w_obs_CIV = (1 + z_src) * w_CIV
        w_obs_CIII = (1 + z_src) * w_CIII
        w_obs_MgII = (1 + z_src) * w_MgII

        lya_flx = qso_flx[l_lya, src]

        nice_lya = True

        for l in line_list_other[src]:
            w_obs_l = w_central[l]
            if ~(   
                # Lines are in expected possitions for QSOs
                (
                (np.abs(w_obs_l - w_obs_lya) < fwhm / 2)
                | (np.abs(w_obs_l - w_obs_SiIV) < fwhm / 2)
                | (np.abs(w_obs_l - w_obs_CIV) < fwhm / 2)
                | (np.abs(w_obs_l - w_obs_CIII) < fwhm / 2)
                | (np.abs(w_obs_l - w_obs_MgII) < fwhm / 2)
                | (w_obs_l > w_obs_MgII + fwhm / 2)
                )
                # The Lya line flux is the highest
                & (qso_flx[l, src] - cont_est_Arr[l - nb_c_min][src]
                    <= lya_flx - cont_est_Arr[l_lya - nb_c_min][src])
                # g > r
                # & (qso_flx[-3, src] - qso_flx[-2, src]\
                    # > (qso_err[-3, src]**2 + qso_err[-2, src]**2) ** 0.5)
                # Max z for LAE set to 4.3
                & (line_list_lya[src] < 28)
                # Cannot be other lines bluer than Lya
                & (l >= l_lya)
            ):
                nice_lya = False
        if (len(line_list_other[src]) > 1) & nice_lya:
            nice_lya_list.append(src)
        if (line_list_lya[src] != -1):
            nice_lya_list_single.append(src)
    print('Nice Lya list done. ({0:0.1f} )'.format(time.time() - t0))
    return nice_lya_list, nice_lya_list_single, line_list_lya, line_list_other

def nice_lya_select(lya_lines, other_lines, pm_flx, cont_est):
    N_sources = len(lya_lines)
    w_central = central_wavelength()
    fwhm_Arr = nb_fwhm(range(56))
    nice_lya = np.zeros(N_sources).astype(bool)

    # Line rest-frame wavelengths (Angstroms)
    w_lya = 1215.67
    w_SiIV = 1397.61
    w_CIV = 1549.48
    w_CIII = 1908.73
    w_MgII = 2799.12

    for src in np.where(np.array(lya_lines) != -1)[0]:
        l_lya = lya_lines[src]
        z_src = w_central[l_lya] / w_lya - 1
    
        w_obs_lya = (1 + z_src) * w_lya
        w_obs_SiIV = (1 + z_src) * w_SiIV
        w_obs_CIV = (1 + z_src) * w_CIV
        w_obs_CIII = (1 + z_src) * w_CIII
        w_obs_MgII = (1 + z_src) * w_MgII

        this_nice = True
        for l in other_lines[src]:
            w_obs_l = w_central[l]
            fwhm = fwhm_Arr[l]
            if ~(   
                # Lines are in expected possitions for QSOs
                (
                    (np.abs(w_obs_l - w_obs_lya) < fwhm / 2)
                    | (np.abs(w_obs_l - w_obs_SiIV) < fwhm / 2)
                    | (np.abs(w_obs_l - w_obs_CIV) < fwhm / 2)
                    | (np.abs(w_obs_l - w_obs_CIII) < fwhm / 2)
                    | (np.abs(w_obs_l - w_obs_MgII) < fwhm / 2)
                    | (w_obs_l > w_obs_MgII + fwhm / 2)
                )
                # The Lya line flux is the highest
                & (
                    (pm_flx[l_lya, src] - cont_est[l_lya, src])
                    - (pm_flx[l, src] - cont_est[l, src])
                    >= 0
                )
                # Max z for LAE set to 4.3
                & (l_lya < 28)
                # Cannot be other lines bluer than Lya
                & (l >= l_lya)
            ):
                this_nice = False
        if this_nice:
            nice_lya[src] = True
    return nice_lya
