import numpy as np

import pickle

import pandas as pd

import csv

import matplotlib.pyplot as plt

from scipy.integrate import simpson

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.table import Table

c = 29979245800  # cm / s


def mag_to_flux(m, w):
    return 10**((m + 48.60) / (-2.5)) * c/w**2 * 1e8


def flux_to_mag(f, w):
    log_arg = np.atleast_1d(f * w**2/c * 1e-8).astype(np.float)
    return -2.5 * np.log10(log_arg) - 48.60


def ang_area(DEC0, delta_DEC, delta_RA):
    '''
    Input:
    DEC0: central DEC coordinate
    delta_DEC, delta_RA: apertures of the box in DEC and RA
    Returns:
    Angular area in deg2
    '''
    # First convert deg to rad and to spherical coordinates (DEC->azimuth)
    DEC0 = np.pi * 0.5 - np.deg2rad(DEC0)
    delta_DEC = np.deg2rad(delta_DEC) / 2
    delta_RA = np.deg2rad(delta_RA) / 2
    # define the result of the spherical integral in two parts
    a = np.cos(DEC0 - delta_DEC) - np.cos(DEC0 + delta_DEC)
    b = 2 * delta_RA
    ang_area = a * b
    # Give the result back in deg
    ang_area = ang_area * 180**2 / np.pi**2
    return ang_area


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

        filename = './JPAS_Transmission_Curves_20170316/JPAS_' + tag + '.tab'
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
        "tag":  filters_tags,
        "w":  filters_w,
        "t":  filters_trans
    }
    return tcurves


def central_wavelength():
    data_tab = Table.read('fits/FILTERs_table.fits', format='fits')
    w_central = data_tab['wavelength']

    return np.array(w_central)

# FWHM of a curve


def nb_fwhm(nb_ind, give_fwhm=True):
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


def estimate_continuum(NB_flx, NB_err, N_nb=7, IGM_T_correct=True,
                       only_right=False, N_nb_min=0, N_nb_max=47):
    '''
    Returns a matrix with the continuum estimate at any NB in all sources.
    '''
    NB_flx = NB_flx[:56]
    NB_err = NB_err[:56]

    cont_est = np.ones(NB_flx.shape) * 99.
    cont_err = np.ones(NB_flx.shape) * 99.
    w_central = central_wavelength()

    for nb_idx in range(1, NB_flx.shape[0]):
        # Limits on where to make the estimation
        if nb_idx < N_nb_min:
            continue
        if nb_idx > N_nb_max:
            break

        if (nb_idx < N_nb) or only_right:
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                    np.array(w_central[: nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.

            # Stack filters at both sides or only at the right of the central one
            if not only_right:
                NBs_to_avg = np.vstack((
                    NB_flx[: nb_idx - 1] / IGM_T,
                    NB_flx[nb_idx + 2: nb_idx + N_nb + 1]
                ))
                NBs_errs = np.vstack((
                    NB_err[: nb_idx - 1] / IGM_T,
                    NB_err[nb_idx + 2: nb_idx + N_nb + 1]
                ))
            if only_right:
                NBs_to_avg = NB_flx[nb_idx + 2: nb_idx + N_nb + 1]
                NBs_errs = NB_err[nb_idx + 2: nb_idx + N_nb + 1]

        if (N_nb <= nb_idx < (NB_flx.shape[0] - 6)) and not only_right:
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                    np.array(w_central[nb_idx - N_nb: nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            NBs_to_avg = np.vstack((
                NB_flx[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_flx[nb_idx + 2: nb_idx + N_nb + 1]
            ))
            NBs_errs = np.vstack((
                NB_err[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_err[nb_idx + 2: nb_idx + N_nb + 1]
            ))

        if nb_idx >= (NB_flx.shape[0] - 6):
            if IGM_T_correct:
                IGM_T = IGM_TRANSMISSION(
                    np.array(w_central[nb_idx - N_nb: nb_idx - 1])
                ).reshape(-1, 1)
            else:
                IGM_T = 1.
            NBs_to_avg = np.vstack((
                NB_flx[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_flx[nb_idx + 2:]
            ))
            NBs_errs = np.vstack((
                NB_err[nb_idx - N_nb: nb_idx - 1] / IGM_T,
                NB_err[nb_idx + 2:]
            ))

        # Conver errors to relative errors for the weights
        # NBs_rel_errs = NBs_errs / NBs_to_avg
        # NBs_rel_errs[NBs_rel_errs < 0] = 99.
        w = NBs_errs ** -2

        cont_est[nb_idx] = np.average(NBs_to_avg, weights=w, axis=0)
        cont_err[nb_idx] = np.sum(NBs_errs ** -2, axis=0) ** -0.5

    return cont_est, cont_err


def NB_synthetic_photometry(f, w_Arr, w_c, fwhm):
    '''
    Returns the synthetic photometry of a set f of (N_sources x binning) in a
    central wavelength w_c with a fwhm.
    '''
    synth_tcurve = np.zeros(w_Arr.shape)
    synth_tcurve[np.where(np.abs(w_Arr - w_c) <= fwhm*0.5)] += 1.
    T_integrated = simpson(synth_tcurve * w_Arr, w_Arr)

    if len(f.shape) == 1:
        return simpson(synth_tcurve * f * w_Arr, w_Arr) / T_integrated
    else:
        return simpson(synth_tcurve * f * w_Arr, w_Arr, axis=1) / T_integrated


def z_volume(z_min, z_max, area):
    '''
    Returns the comoving volume for an observation area between a range of redshifts
    '''
    z_x = np.linspace(z_min, z_max, 1000)
    dV = cosmo.differential_comoving_volume(z_x).to(u.Mpc**3 / u.sr).value
    area *= (2 * np.pi / 360) ** 2
    theta = np.arccos(1 - area / (2 * np.pi))
    Omega = 2 * np.pi * (1 - np.cos(theta))
    vol = simpson(dV, z_x) * Omega
    return vol


def IGM_TRANSMISSION(w_Arr, A=-0.001845, B=3.924):
    '''
    Returns the IGM transmission associated to the Lya Break.
    '''
    return np.exp(A * (w_Arr / 1215.67)**B)


def plot_JPAS_source(flx, err, set_ylim=True, e17scale=False):
    '''
    Generates a plot with the JPAS data.
    '''

    if e17scale:
        flx = flx * 1e17
        err = err * 1e17

    data_tab = Table.read('fits/FILTERs_table.fits', format='fits')
    cmap = data_tab['color_representation']
    w_central = data_tab['wavelength']
    # fwhm_Arr = data_tab['width']

    data_max = np.max(flx)
    data_min = np.min(flx)
    y_max = (data_max - data_min) * 2/3 + data_max
    y_min = data_min - (data_max - data_min) * 0.5

    ax = plt.gca()
    for i, w in enumerate(w_central[:-4]):
        ax.errorbar(w, flx[i], yerr=err[i],
                    marker='o', markeredgecolor='dimgray', markerfacecolor=cmap[i],
                    markersize=8, ecolor='dimgray', capsize=4, capthick=1, linestyle='',
                    zorder=-99)
    ax.errorbar(w_central[-4], flx[-4], yerr=err[-4], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-4], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)
    ax.errorbar(w_central[-3], flx[-3], yerr=err[-3], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-3], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)
    ax.errorbar(w_central[-2], flx[-2], yerr=err[-2], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-2], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)
    ax.errorbar(w_central[-1], flx[-1], yerr=err[-1], markeredgecolor='dimgray',
                fmt='s', markerfacecolor=cmap[-1], markersize=10,
                ecolor='dimgray', capsize=4, capthick=1)

    try:
        if set_ylim:
            ax.set_ylim((y_min, y_max))
    except:
        pass

    ax.set_xlabel('$\lambda\ (\AA)$', size=15)
    if e17scale:
        ax.set_ylabel(
            r'$f_\lambda\cdot10^{17}$ (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', size=15)
    else:
        ax.set_ylabel(
            '$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', size=15)

    return ax


def identify_lines(line_Arr, qso_flx, cont_flx, nb_min=0, first=False,
                   return_line_width=False):
    '''
    Returns a list of N lists with the index positions of the lines.

    Input: 
    line_Arr: Bool array of 3sigma detections in sources. Dim N_filters x N_sources
    qso_flx:  Flambda data
    nb_min
    '''
    N_fil, N_src = line_Arr.shape
    line_list = []
    line_len_list = []
    line_cont_list = []

    for src in range(N_src):
        fil = 0
        this_src_lines = []  # The list of lines
        this_cont_lines = []  # The list of continuum indices of lines
        this_src_line_flx = [] # The list of lengths of this src lines

        while fil < N_fil:
            this_line = []  # The list of contiguous indices of this line
            while ~line_Arr[fil, src]:
                fil += 1
                if fil == N_fil - 1:
                    break
            if fil == N_fil - 1:
                break
            while line_Arr[fil, src]:
                this_line.append(fil)
                fil += 1
                if fil == N_fil - 1:
                    break
            if fil == N_fil - 1:
                break

            aux = -len(this_line) + nb_min + fil

            if first:  # If first=True, append continuum index to list
                this_cont_lines.append(
                    np.average(
                        np.array(this_line),
                        weights=qso_flx[np.array(this_line), src] ** 2
                    )
                )
            # Append index of the max flux of this line to the list
            this_src_lines.append(
                np.argmax(qso_flx[np.array(this_line) + nb_min, src]) + aux
            )
            this_src_line_flx.append(qso_flx[np.array(this_line) + nb_min, src].sum())

        if first:  # If first=True,
            try:
                # idx = np.argmax(
                #     qso_flx[np.array(this_src_lines), src]
                #     - cont_flx[np.array(this_src_lines), src]
                # )
                idx = np.argmax(
                    np.array(this_src_line_flx)
                    - cont_flx[np.array(this_src_lines), src]
                )

                line_list.append(this_src_lines[idx])
                line_len_list.append(this_src_lines)
                line_cont_list.append(this_cont_lines[idx])
            except:
                line_list.append(-1)
                line_len_list.append([-1])
                line_cont_list.append(-1)

        if not first:
            line_list.append(this_src_lines)

    if first:
        if return_line_width:
            return line_list, line_cont_list, line_len_list
        else:
            return line_list, line_cont_list
    return line_list


def z_NB(cont_line_pos):
    '''
    Computes the Lya z for a continuum NB index.
    '''
    w_central = central_wavelength()

    # Convert to numpy arr
    cont_line_pos = np.atleast_1d(cont_line_pos)

    w1 = w_central[cont_line_pos.astype(int)]
    w2 = w_central[cont_line_pos.astype(int) + 1]

    w = (w2 - w1) * cont_line_pos % 1 + w1

    return w / 1215.67 - 1


def NB_z(z):
    '''
    Takes a redshift as an argument and returns the corresponding NB to that redshift.
    Returns -1 if the Lya redshift is out of boundaries.
    '''
    z = np.atleast_1d(z)
    w_central_NB = central_wavelength()[:56]
    w_lya_obs = (z + 1) * 1215.67

    n_NB = np.zeros(len(z)).astype(int)
    for i, w in enumerate(w_lya_obs):
        n_NB[i] = int(np.argmin(np.abs(w_central_NB - w)))

    # 0 means the medium band, so thats a bad value -> assign it -1
    # 55 It's too much, so let's assign also -1
    n_NB[(n_NB < 1) | (n_NB > 54)] = -1

    # If only one value passed, return as a number instead of numpy array
    if len(n_NB) == 1:
        n_NB = n_NB[0]

    return n_NB


def mask_proper_motion(parallax_sn, pmra_sn, pmdec_sn):
    '''
    Masks sources with significant proper motion measurement in Gaia
    '''
    mask = (
        (np.sqrt(parallax_sn ** 2 + pmra_sn ** 2 + pmdec_sn**2) < 27 ** 0.5)
        | (np.isnan(parallax_sn) | np.isnan(pmra_sn) | np.isnan(pmdec_sn))
    )
    return mask


def is_there_line(pm_flx, pm_err, cont_est, cont_err, ew0min,
                  mask=True, obs=False, sigma=3):
    w_central = central_wavelength()[:-4]
    fwhm_Arr = nb_fwhm(range(56)).reshape(-1, 1)

    if not obs:
        z_nb_Arr = (w_central / 1215.67 - 1).reshape(-1, 1)
        ew_Arr = ew0min * (1 + z_nb_Arr)
    if obs:
        ew_Arr = ew0min

    line = (
        # 3-sigma flux excess
        (
            pm_flx[:-4] - cont_est > sigma * (pm_err[:-4]**2 + cont_err**2) ** 0.5
        )
        # EW0 min threshold
        & (
            pm_flx[:-4] / cont_est > 1 + ew_Arr / fwhm_Arr
        )
        & (
            pm_flx[:-4] > cont_est
        )
        # Masks
        & (
            mask
        )
        # Check that cont_est is ok
        & (
            np.isfinite(cont_est)
        )
    )
    return line


def nice_lya_select(lya_lines, other_lines, pm_flx, pm_err, cont_est, z_Arr, mask=None):
    N_sources = len(lya_lines)
    w_central = central_wavelength()
    fwhm_Arr = nb_fwhm(range(56))
    nice_lya = np.zeros(N_sources).astype(bool)

    # Line rest-frame wavelengths (Angstroms)
    w_lyb = 1025.7220
    w_lya = 1215.67
    w_SiIV = 1397.61
    w_CIV = 1549.48
    w_CIII = 1908.73
    w_MgII = 2799.12

    i = flux_to_mag(pm_flx[-1], w_central[-1])
    r = flux_to_mag(pm_flx[-2], w_central[-2])
    g = flux_to_mag(pm_flx[-3], w_central[-3])
    gr = g - r
    ri = r - i
    # For z > 3
    # color_aux1 = (-1.5 * ri + 1.7 > gr)
    color_aux1 = (ri < 0.6) & (gr < 1.5)
    # For z < 3
    # color_aux2 = (-1.5 * ri + 2.5 > gr) & (ri < 1.)
    color_aux2 = (ri < 0.6) & (gr < 0.7)

    # color_aux1 = np.ones(g.shape).astype(bool)
    # color_aux2 = np.ones(g.shape).astype(bool)

    for src in np.where(np.array(lya_lines) != -1)[0]:
        # l_lya = lya_lines[src]
        z_src = z_Arr[src]

        w_obs_lya = (1 + z_src) * w_lya
        w_obs_lyb = (1 + z_src) * w_lyb
        w_obs_SiIV = (1 + z_src) * w_SiIV
        w_obs_CIV = (1 + z_src) * w_CIV
        w_obs_CIII = (1 + z_src) * w_CIII
        w_obs_MgII = (1 + z_src) * w_MgII

        this_nice = True

        # Check the Lyman limit
        w_central_0 = w_central / (1 + z_src)
        w_central_0[0] = 99999
        w_central_0[-4] = 99999
        Lybreak_flx_Arr = pm_flx[w_central_0 < 912, src]
        Lybreak_err_Arr = pm_err[w_central_0 < 912, src]
        if len(Lybreak_flx_Arr) != 0:
            Lybreak_flx = np.average(Lybreak_flx_Arr, weights=Lybreak_err_Arr ** -2)
            Lybreak_err = np.sum(Lybreak_err_Arr ** -2) ** -0.5

            if Lybreak_flx - pm_flx[-3, src] > 3 * Lybreak_err:
                this_nice = False

        for l in other_lines[src]:
            # Ignore very red and very blue NBs
            if (l > 50) | (l < 1):
                continue

            w_obs_l = w_central[l]
            fwhm = fwhm_Arr[l]

            good_l = (
                (np.abs(w_obs_l - w_obs_lya) < fwhm * 1.5)
                | (np.abs(w_obs_l - w_obs_lyb) < fwhm * 1.5)
                | (np.abs(w_obs_l - w_obs_SiIV) < fwhm * 1.5)
                | (np.abs(w_obs_l - w_obs_CIV) < fwhm * 1.5)
                | (np.abs(w_obs_l - w_obs_CIII) < fwhm * 1.5)
                | (np.abs(w_obs_l - w_obs_MgII) < fwhm * 1.5)
                | (w_obs_l > w_obs_MgII + fwhm)
            )

            if ~good_l:
                this_nice = False
                break

        if not this_nice:
            continue
        elif len(other_lines[src]) > 1:
            pass
        else:
            if z_src > 3.:
                good_colors = color_aux2[src]
            else:
                good_colors = color_aux1[src]
            if ~good_colors:
                this_nice = False

        if this_nice:
            nice_lya[src] = True

    if mask is None:
        return nice_lya
    else:
        return nice_lya & mask


def count_true(arr):
    '''
    Counts how many True values in bool array
    '''
    return sum(arr)


def schechter(L, phistar, Lstar, alpha):
    '''
    Just the regular Schechter function
    '''
    return (phistar / Lstar) * (L / Lstar)**alpha * np.exp(-L / Lstar)


def double_schechter(L, phistar1, Lstar1, alpha1, phistar2, Lstar2, alpha2,
                     scale1=1., scale2=1.):
    '''
    A double schechter.
    '''
    Phi1 = schechter(L, phistar2, Lstar2, alpha2) * scale1
    Phi2 = schechter(L, phistar1, Lstar1, alpha1) * scale2

    return Phi1 + Phi2


def EW_L_NB(pm_flx, pm_err, cont_flx, cont_err, z_Arr, lya_lines, F_bias=None,
            nice_lya=None, N_nb=0):
    '''
    Returns the EW0 and the luminosity from a NB selection given by lya_lines
    '''

    w_central = central_wavelength()

    N_sources = pm_flx.shape[1]
    nb_fwhm_Arr = np.array(nb_fwhm(range(56)))

    if nice_lya is None:
        nice_lya = np.ones(N_sources).astype(bool)

    EW_nb_Arr = np.zeros(N_sources)
    EW_nb_e = np.zeros(N_sources)
    L_Arr = np.zeros(N_sources)
    L_e_Arr = np.zeros(N_sources)
    cont = np.zeros(N_sources)
    cont_e = np.zeros(N_sources)
    flambda = np.zeros(N_sources)
    flambda_e = np.zeros(N_sources)

    for src in np.where(nice_lya)[0]:
        l = lya_lines[src]
        if l == -1:
            continue

        cont[src] = cont_flx[l, src]
        cont_e[src] = cont_err[l, src]

        # Let's integrate the NB flux over the transmission curves to obtain Flambda
        l_start = np.max([0, l - N_nb])

        lw = np.arange(l_start, l + N_nb + 1)

        IGM_T_Arr = np.ones(len(lw))
        IGM_T_Arr[: l -
                  l_start] = IGM_TRANSMISSION(w_central[lw[: l - l_start]])
        IGM_T_Arr[l -
                  l_start] = (IGM_TRANSMISSION(w_central[lw[l - l_start]]) + 1) * 0.5

        pm_flx[l_start: l + N_nb + 1, src] /= IGM_T_Arr
        pm_flx[l_start: l + N_nb + 1, src][pm_flx[l_start: l +
                                                  N_nb + 1, src] < cont[src]] = cont[src]

        intersec = 0.
        for i in range(lw[0], lw[-1]):
            intersec_dlambda = (
                (nb_fwhm_Arr[i] + nb_fwhm_Arr[i + 1]) * 0.5
                - (w_central[i + 1] - w_central[i])
            )
            intersec += np.min(
                [(pm_flx[i, src]) * intersec_dlambda,
                 (pm_flx[i + 1, src]) * intersec_dlambda]
            )

        flambda_cont = cont[src] * (
            w_central[l + N_nb] + nb_fwhm_Arr[l + N_nb] * 0.5
            - (w_central[l_start] - nb_fwhm_Arr[l_start] * 0.5)
        )

        flambda[src] = np.sum(
            (pm_flx[lw[0]: lw[-1] + 1, src]) * nb_fwhm_Arr[lw[0]: lw[-1] + 1]
        ) - intersec - flambda_cont
        flambda_e[src] = (
            np.sum(
                (pm_err[lw[0]: lw[-1] + 1, src] *
                 nb_fwhm_Arr[lw[0]: lw[-1] + 1]) ** 2
            )
            + (flambda_cont / cont[src] * cont_e[src]) ** 2
        ) ** 0.5

    if F_bias is not None:
        flambda /= F_bias[np.array(lya_lines)]

    EW_nb_Arr = flambda / cont / (1 + z_Arr)
    EW_nb_e = flambda_e / cont / (1 + z_Arr)

    def LumDist(z): return cosmo.luminosity_distance(z).to(u.cm).value
    def Redshift(w): return w / 1215.67 - 1
    dL = LumDist(z_Arr)
    dL_e = (
        LumDist(
            Redshift(
                w_central[lya_lines] + 0.5 * nb_fwhm_Arr[lya_lines]
            )
        )
        - LumDist(
            Redshift(
                w_central[lya_lines]
            )
        )
    )

    L_Arr = np.log10(flambda * 4*np.pi * dL ** 2)
    L_e_Arr = (
        (dL ** 2 * flambda_e) ** 2
        + (2*dL * dL_e * flambda) ** 2
    ) ** 0.5 * 4*np.pi

    return EW_nb_Arr, EW_nb_e, L_Arr, L_e_Arr, flambda, flambda_e


def Zero_point_error(tile_id_Arr, catname):
    # Load Zero Point magnitudes
    w_central = central_wavelength()
    zpt_cat = pd.read_csv(
        f'csv/{catname}.CalibTileImage.csv', sep=',', header=1)

    zpt_mag = zpt_cat['ZPT'].to_numpy()
    zpt_err = zpt_cat['ERRZPT'].to_numpy()

    ones = np.ones((len(w_central), len(zpt_mag)))

    zpt_err = (
        mag_to_flux(ones * zpt_mag, w_central.reshape(-1, 1))
        - mag_to_flux(ones * (zpt_mag + zpt_err), w_central.reshape(-1, 1))
    )

    # Duplicate rows to match the tile_ID of each source
    idx = np.empty(tile_id_Arr.shape).astype(int)

    zpt_id = zpt_cat['TILE_ID'].to_numpy()
    for src in range(len(tile_id_Arr)):
        idx[src] = np.where(
            (zpt_id == tile_id_Arr[src]) & (
                zpt_cat['IS_REFERENCE_METHOD'] == 1)
        )[0][0]

    zpt_err = zpt_err[:, idx]

    return zpt_err


def ML_predict_L(pm_flx, pm_err, z_Arr, L_Arr, regname, L_lya=None):
    '''
    Predicts the L given the photometry fluxes and the peviously estimated z and L.
    '''
    w_central = central_wavelength()
    NNdata = np.hstack(
        (
            pm_flx[2:55].T,
            pm_flx[-3:].T,
            pm_err[2:55].T / pm_flx[2:55].T,
            pm_err[-3:].T / pm_flx[-3:].T,
            L_Arr.reshape(-1, 1),
            z_Arr.reshape(-1, 1)
        )
    )

    ####### Pre-processing #######

    # Take the relative fluxes to the selected one
    NB_lya_position = NB_z(NNdata[:, -1].reshape(-1,))
    for i, nb in enumerate(NB_lya_position - 2):
        NNdata[i, :53] = (
            flux_to_mag(NNdata[i, :53], w_central[:53])
            - flux_to_mag(NNdata[i, :53][nb], w_central[nb + 2])
        )
        NNdata[i, 53: 53 +
               3] = flux_to_mag(NNdata[i, 53: 53 + 3], w_central[-3:])

    NNdata[~np.isfinite(NNdata)] = 99.

    # MinMaxScaler
    with open(f'MLmodels/{regname}_QSO-SF_scaler.sav', 'rb') as file:
        mms = pickle.load(file)
    NNdata = mms.transform(NNdata)

    # PCA
    with open(f'MLmodels/{regname}_QSO-SF_pca.sav', 'rb') as file:
        pca = pickle.load(file)
    NNdata = pca.transform(NNdata)

    ##############################

    # Import the regressor
    with open(f'MLmodels/{regname}_QSO-SF_regressor.sav', 'rb') as file:
        reg = pickle.load(file)

    reg.set_params(n_jobs=-1, verbose=0)

    L_Arr_pred = reg.predict(NNdata)
    if L_lya is not None:
        print(f'Score: {reg.score(NNdata, L_lya)}')

    return L_Arr_pred
