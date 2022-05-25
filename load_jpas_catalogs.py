import numpy as np
import pandas as pd
from my_functions import Zero_point_error

def load_minijpas_jnep(name_list):
    pm_flx = np.array([]).reshape(60, 0)
    pm_err = np.array([]).reshape(60, 0)
    ra = np.array([])
    dec = np.array([])
    tile_id = np.array([])
    parallax_sn = np.array([])
    pmra_sn = np.array([])
    pmdec_sn = np.array([])
    starprob = np.array([])
    spCl = np.array([])
    zsp = np.array([])

    N_minijpas = 0
    split_converter = lambda s: np.array(s.split()).astype(float)
    sum_flags = lambda s: np.sum(np.array(s.split()).astype(float))

    for name in name_list:
        cat = pd.read_csv(f'csv/{name}.Flambda_aper3_photoz_gaia_3.csv', sep=',', header=1,
            converters={0: int, 1: int, 2: split_converter, 3: split_converter, 4: sum_flags,
            5: sum_flags})

        cat = cat[np.array([len(x) for x in cat['FLUX_APER_3_0']]) != 0] # Drop bad rows due to bad query
        cat = cat[(cat.FLAGS == 0) & (cat.MASK_FLAGS == 0)] # Drop flagged
        cat = cat.reset_index()

        ra_i = cat['ALPHA_J2000']
        dec_i = cat['DELTA_J2000']

        tile_id_i = cat['TILE_ID'].to_numpy()

        parallax_i = cat['parallax'].to_numpy() / cat['parallax_error'].to_numpy()
        pmra_i = cat['pmra'].to_numpy() / cat['pmra_error'].to_numpy()
        pmdec_i = cat['pmdec'].to_numpy() / cat['pmdec_error'].to_numpy()

        pm_flx_i = np.stack(cat['FLUX_APER_3_0'].to_numpy()).T * 1e-19
        pm_err_i = np.stack(cat['FLUX_RELERR_APER_3_0'].to_numpy()).T * pm_flx_i

        if name == 'minijpas':
            N_minijpas = pm_flx_i.shape[1]

        starprob_i = cat['morph_prob_star']

        pm_err_i = (pm_err_i ** 2 + Zero_point_error(cat['TILE_ID'], name) ** 2) ** 0.5

        spCl_i = cat['spCl']
        zsp_i = cat['zsp']

        pm_flx = np.hstack((pm_flx, pm_flx_i))
        pm_err = np.hstack((pm_err, pm_err_i))
        ra = np.concatenate((ra, ra_i))
        dec = np.concatenate((dec, dec_i))
        tile_id = np.concatenate((tile_id, tile_id_i))
        pmra_sn = np.concatenate((pmra_sn, pmra_i))
        pmdec_sn = np.concatenate((pmdec_sn, pmdec_i))
        parallax_sn = np.concatenate((parallax_sn, parallax_i))
        starprob = np.concatenate((starprob, starprob_i))
        spCl = np.concatenate((spCl, spCl_i))
        zsp = np.concatenate((zsp, zsp_i))

    return pm_flx, pm_err, ra, dec, tile_id, pmra_sn, pmdec_sn, parallax_sn, starprob,\
        spCl, zsp, N_minijpas