import numpy as np
import pandas as pd
from LAEs.my_functions import Zero_point_error

def load_minijpas_jnep(cat_list=['minijpas', 'jnep'], selection=False,
                       flags_mask=True):
    # If selection, return the valuable items for visual_inspection.py only
    pm_flx = np.array([]).reshape(60, 0)
    pm_err = np.array([]).reshape(60, 0)
    tile_id = np.array([])
    parallax_sn = np.array([])
    pmra_sn = np.array([])
    pmdec_sn = np.array([])
    starprob = np.array([])
    starlhood = np.array([])
    spCl = np.array([])
    zsp = np.array([])
    photoz = np.array([])
    photoz_odds = np.array([])
    photoz_chi_best = np.array([])
    x_im = np.array([])
    y_im = np.array([])
    RA = np.array([])
    DEC = np.array([])
    number = np.array([])

    N_minijpas = 0
    split_converter = lambda s: np.array(s.split()).astype(float)
    sum_flags = lambda s: np.sum(np.array(s.split()).astype(float))

    for name in cat_list:
        cat = pd.read_csv(f'csv/{name}.Flambda_aper3_photoz_gaia_3.csv', sep=',', header=1,
            converters={0: int, 1: int, 2: split_converter, 3: split_converter, 4: sum_flags,
            5: sum_flags})

        cat = cat[np.array([len(x) for x in cat['FLUX_APER_3_0']]) != 0] # Drop bad rows due to bad query

        if flags_mask:
            cat = cat[(cat.FLAGS == 0) & (cat.MASK_FLAGS == 0)] # Drop flagged
        cat = cat.reset_index()

        tile_id_i = cat['TILE_ID'].to_numpy()

        parallax_i = cat['parallax'].to_numpy() / cat['parallax_error'].to_numpy()
        pmra_i = cat['pmra'].to_numpy() / cat['pmra_error'].to_numpy()
        pmdec_i = cat['pmdec'].to_numpy() / cat['pmdec_error'].to_numpy()

        pm_flx_i = np.stack(cat['FLUX_APER_3_0'].to_numpy()).T * 1e-19
        pm_err_i = np.stack(cat['FLUX_RELERR_APER_3_0'].to_numpy()).T * pm_flx_i

        if name == 'minijpas':
            N_minijpas = pm_flx_i.shape[1]
        
        starprob_i = cat['morph_prob_star']
        starlhood_i = cat['morph_lhood_star']

        RA_i = cat['ALPHA_J2000']
        DEC_i = cat['DELTA_J2000']

        pm_err_i = (pm_err_i ** 2 + Zero_point_error(cat['TILE_ID'], name) ** 2) ** 0.5

        spCl_i = cat['spCl']
        zsp_i = cat['zsp']

        photoz_i = cat['PHOTOZ']
        photoz_odds_i = cat['ODDS']
        photoz_chi_best_i = cat['CHI_BEST']

        x_im_i = cat['X_IMAGE']
        y_im_i = cat['Y_IMAGE']

        number_i = cat['NUMBER']

        pm_flx = np.hstack((pm_flx, pm_flx_i))
        pm_err = np.hstack((pm_err, pm_err_i))
        tile_id = np.concatenate((tile_id, tile_id_i))
        pmra_sn = np.concatenate((pmra_sn, pmra_i))
        pmdec_sn = np.concatenate((pmdec_sn, pmdec_i))
        parallax_sn = np.concatenate((parallax_sn, parallax_i))
        starprob = np.concatenate((starprob, starprob_i))
        starlhood = np.concatenate((starlhood, starlhood_i))
        spCl = np.concatenate((spCl, spCl_i))
        zsp = np.concatenate((zsp, zsp_i))
        photoz = np.concatenate((photoz, photoz_i))
        photoz_odds = np.concatenate((photoz_odds, photoz_odds_i))
        photoz_chi_best = np.concatenate((photoz_chi_best, photoz_chi_best_i))
        x_im = np.concatenate((x_im, x_im_i))
        y_im = np.concatenate((y_im, y_im_i))
        RA = np.concatenate((RA, RA_i))
        DEC = np.concatenate((DEC, DEC_i))
        number = np.concatenate((number, number_i))

    if selection:
        return pm_flx, pm_err, x_im, y_im, tile_id, number, starprob, spCl,\
            photoz, photoz_chi_best, photoz_odds, RA, DEC
    else:
        return pm_flx, pm_err, tile_id, pmra_sn, pmdec_sn, parallax_sn, starprob, starlhood,\
            spCl, zsp, photoz, photoz_chi_best, photoz_odds, N_minijpas, x_im, y_im, RA, DEC

def load_sdss_xmatch():
    filename = 'csv/xmatch_sdss_dr12.csv'
    cat = pd.read_csv(filename, sep=',', header=1)

    return cat['NUMBER'], cat['TILE_ID'], cat['SpObjID'], cat['f_zsp'], cat['zsp']