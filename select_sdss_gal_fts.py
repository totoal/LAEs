from astropy.io import fits
import pandas as pd


filename = '/home/alberto/almacen/SDSS_spectra_fits/DR16/DR16Q_Superset_v3.fits'
with fits.open(filename) as fread:
    good_gal = (
                (fread[1].data['ZWARNING'] == 0)
                & (fread[1].data['SN_MEDIAN_ALL'] > 0)
                & (fread[1].data['CLASS_PERSON'] == 4)
                & (fread[1].data['Z_CONF'] >= 2)
            )

    plate = fread[1].data['PLATE'][good_gal]
    mjd = fread[1].data['MJD'][good_gal]
    fiber = fread[1].data['FIBERID'][good_gal]

    # The array of redshifts is taken from this file
    z_spec = fread[1].data['Z'][good_gal]

data = {
    'zspec': z_spec,
    'mjd': mjd,
    'plate': plate,
    'fiber': fiber
}

pd.DataFrame(data).to_csv('csv/Gal_fts_DR16_v2.csv')