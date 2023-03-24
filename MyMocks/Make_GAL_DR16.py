import numpy as np
import pandas as pd
import pickle
from my_utilities import *
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

w_central = central_wavelength()

def bin_centers(bins):
    return np.array([bins[i : i + 2].sum() * 0.5 for i in range(len(bins) - 1)])

def M_to_m(M, redshift, x, y, z):
    '''
    Computes the apparent magnitude from the absolute magnitude
    Inputs:
    M: Absolute magnitude
    redshift: Redshift of the source
    x, y, z: Coordinates in the Lightcone (pc)
    '''
    # Luminosity distance:
    dL = cosmo.luminosity_distance(redshift).to(u.pc).value.reshape(-1, 1)

    return M + 5 * np.log10(dL) - 5

def load_r_dist():
    lc = np.load('/home/alberto/almacen/David_lightcone/LightCone_512_15sep2022.npy')

    xx, yy, zz = lc['pos'].T
    z = lc['redshift']
    lc_rmag = M_to_m(lc['ObsMagDustLine'], z, xx, yy, zz)[:, -2]
    
    return lc_rmag




if __name__ == '__main__':
    # Define N_sources
    N_sources = 2_000_000

    # First load the prior r distribution
    lc_rmag = load_r_dist()

    rbins = np.linspace(16, 24.5, 100)
    rbins_c = bin_centers(rbins)
    h_rmag = np.histogram(lc_rmag, rbins)[0]

    # Generate array of r
    r_min, r_max = 17, 24.5
    r_cum_x = np.linspace(r_min, r_max, 1000)
    r_counts_cum = np.cumsum(np.interp(r_cum_x, rbins_c, h_rmag))
    r_counts_cum /= r_counts_cum.max()

    my_r_Arr = np.interp(np.random.rand(N_sources),
                         r_counts_cum, r_cum_x)

    # Load the DR16 GAL photometry
    filename_pm_DR16 = ('../csv/J-SPECTRA_GAL_Superset_DR16_v2.csv')
    pm_SEDs_DR16 = pd.read_csv(
        filename_pm_DR16, usecols=np.arange(1, 64)
    ).to_numpy()[:, 0:60].T

    # Generate random array of src to use from the GAL_DR16
    src_Arr = np.random.choice(np.arange(pm_SEDs_DR16.shape[1]), N_sources)


    # Initialize matrix of pm_flx
    out_pm_flx = np.empty((60, N_sources))

    for i in range(N_sources):
        this_src = src_Arr[i]
        this_r_flx = mag_to_flux(my_r_Arr[i], w_central[-2])

        flx_factor = this_r_flx / pm_SEDs_DR16[-2, this_src]

        out_pm_flx[:, i] = pm_SEDs_DR16[:, this_src] * flx_factor


    # Save the cat
    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()
    gal_fts = pd.read_csv('../csv/Gal_fts_DR16_v2.csv')
    z = gal_fts['zspec'].to_numpy().flatten()

    data = {
        'pm_flx': out_pm_flx,
        'z': z[src_Arr]
    }
    dirname = '/home/alberto/almacen/Source_cats'
    filename = f'{dirname}/GAL_DR16.npy'

    with open(filename, 'wb') as f:
        pickle.dump(data, f)