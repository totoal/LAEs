import numpy as np

from scipy.stats import binned_statistic_2d
from scipy.integrate import simpson

from my_functions import mag_to_flux

import pandas as pd

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

def completeness_curve(m50, k, mag):
    return 1. - 1. / (np.exp(-k * (mag - m50)) + 1)

def r_intrinsic_completeness(star_prob, r_Arr, tile_id, survey_name):
    '''
    Computes the completeness for each source based on its r-band flux according to the
    completeness curves of minijpas. Bonoli et al. 2021
    '''
    if survey_name == 'minijpas':
        TileImage = pd.read_csv('csv/minijpas.TileImage.csv', header=1)
    elif survey_name == 'jnep':
        TileImage = pd.read_csv('csv/jnep.TileImage.csv', header=1)
    elif survey_name == 'both':
        TileImage = pd.concat(
            [
                pd.read_csv('csv/minijpas.TileImage.csv', header=1),
                pd.read_csv('csv/jnep.TileImage.csv', header=1)
            ]
        ).reset_index()
    else:
        raise ValueError('Survey name not known')

    where = np.zeros(r_Arr.shape).astype(int)

    for src in range(len(r_Arr)):
        where[src] = np.where(
            (TileImage['TILE_ID'] == tile_id[src])
            & (TileImage['FILTER_ID'] == 59)
        )[0]

    m50s = TileImage['M50S'][where]
    ks = TileImage['KS'][where]
    m50g = TileImage['M50G'][where]
    kg = TileImage['KG'][where]

    isstar = (star_prob >= 0.5)

    intcomp = np.empty(r_Arr.shape)
    intcomp[isstar] = completeness_curve(m50s[isstar], ks[isstar], r_Arr[isstar])
    intcomp[~isstar] = completeness_curve(m50g[~isstar], kg[~isstar], r_Arr[~isstar])

    return intcomp

def puricomp2d_weights(L_Arr, r_Arr, survey_name, dirname, give_puri_comp=False):
    '''
    Computes the weight (purity/completeness) of each source based on the selection in 
    mocks.
    '''

    L_bins = np.load(f'{dirname}/puricomp2d_L_bins.npy')
    r_bins = np.load(f'{dirname}/puricomp2d_r_bins.npy')
    puri2d = np.load(f'{dirname}/puri2d_{survey_name}.npy')
    comp2d = np.load(f'{dirname}/comp2d_{survey_name}.npy')

    w_mat = puri2d / comp2d
    w_mat[np.isnan(w_mat) | np.isinf(w_mat)] = 0.

    # Add a zeros row & column to w_mat for perturbed luminosities exceeding the binning
    w_mat = np.vstack([w_mat, np.zeros(w_mat.shape[1])])
    w_mat = np.hstack([w_mat, np.zeros(w_mat.shape[0]).reshape(-1, 1)])

    # If L_Arr is empty, return empty weights lists
    if len(L_Arr) == 0:
        if not give_puri_comp:
            return np.array([])
        else:
            return np.array([]), np.array([]), np.array([])

    bs = binned_statistic_2d(
        L_Arr, r_Arr, None, 'count', bins=[L_bins, r_bins], expand_binnumbers=True
    )

    if give_puri_comp:
        puri_mat = puri2d
        comp_mat = comp2d
        
        puri_mat[np.isnan(puri_mat) | np.isinf(puri_mat)] = 0.
        comp_mat[np.isnan(comp_mat) | np.isinf(comp_mat)] = 0.

        puri_mat = np.vstack([puri_mat, np.zeros(puri_mat.shape[1])])
        puri_mat = np.hstack([puri_mat, np.zeros(puri_mat.shape[0]).reshape(-1, 1)])
        comp_mat = np.vstack([comp_mat, np.zeros(comp_mat.shape[1])])
        comp_mat = np.hstack([comp_mat, np.zeros(comp_mat.shape[0]).reshape(-1, 1)])

    xx, yy = bs.binnumber

    if not give_puri_comp:
        return w_mat[xx - 1, yy - 1]
    else:
        return w_mat[xx - 1, yy - 1], puri_mat[xx - 1, yy - 1], comp_mat[xx - 1, yy - 1]

def Lya_intrisic_completeness(L, z, starprob=None):
    '''
    Computes the completeness for each source based on its L and its star/galaxy
    classification given in the minijpas.TileImage table only using morphology.
    '''
    if starprob is None:
        starprob = np.ones(L.shape)
    isstar = (starprob >= 0.5)

    ## MiniJPAS limiting r magnitudes
    mag = np.ones(L.shape) * 23.6
    mag[~isstar] = 22.7

    Fline = 10 ** L / (cosmo.luminosity_distance(z).to(u.cm).value ** 2 * 4*np.pi)
    fcont = mag_to_flux(mag, 6750)

    EW_max = Fline / fcont / (1 + z)

    ew_x = np.linspace(20, 1000, 10000)
    w_0 = 75
    ew_dist = lambda ew_xx: np.exp(-ew_xx / w_0)

    total_ew = simpson(ew_dist(ew_x), ew_x)

    completeness = np.empty(L.shape)

    for src in range(len(L)):
        src_ew_x = np.linspace(20, EW_max[src], 1000)
        completeness[src] = simpson(ew_dist(src_ew_x), src_ew_x) / total_ew

    return completeness

def weights_LF(L_Arr, mag, z_Arr, starprob, tile_id,
               survey_name, dirname, which_w=[0, 2], give_puri_comp=False):
    '''
    Combines the contribution of the 3 above functions.
    '''
    args1 = (L_Arr, mag, survey_name, dirname, give_puri_comp)
    args2 = (L_Arr, z_Arr, starprob)
    args3 = (starprob, mag, tile_id, survey_name[:8])

    w1 = 1.
    w2 = 1.
    w3 = 1.

    for i in which_w:
        if i == 0:
            if give_puri_comp:
                w1, puri, comp = puricomp2d_weights(*args1)
            else:
                w1 = puricomp2d_weights(*args1)
        if i == 1:
            w2 = Lya_intrisic_completeness(*args2) ** -1
        if i == 2:
            w3 = r_intrinsic_completeness(*args3) ** -1

    w_Arr = [w1, w2, w3]

    wt = np.ones(L_Arr.shape)
    for i in which_w:
        wt *= w_Arr[i]

    if not give_puri_comp:
        return wt
    else:
        return puri, comp * w3