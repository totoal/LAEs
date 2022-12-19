from astropy.io import fits
import pandas as pd
import numpy as np

filename = '/home/alberto/almacen/SDSS_spectra_fits/DR16/DR16Q_Superset_v3.fits'
with fits.open(filename) as fread:
    good_qso = (
        (fread[1].data['ZWARNING'] == 0)
        & (fread[1].data['SN_MEDIAN_ALL'] > 0)
        & (fread[1].data['IS_QSO_FINAL'] > 0)
    )

    plate = fread[1].data['PLATE'][good_qso]
    mjd = fread[1].data['MJD'][good_qso]
    fiber = fread[1].data['FIBERID'][good_qso]

    # The array of redshifts is taken from this file
    z_qso = fread[1].data['Z'][good_qso]


N_sources = len(fiber)

lya_cont = np.zeros(N_sources)
lya_cont_err = np.zeros(N_sources)
lya_F = np.zeros(N_sources)
NV_F = np.zeros(N_sources)
NV_F_err = np.zeros(N_sources)
lya_F_err = np.zeros(N_sources)
lya_z = np.zeros(N_sources)
lya_EW = np.zeros(N_sources)
lya_EW_err = np.zeros(N_sources)
NV_EW = np.zeros(N_sources)
NV_EW_err = np.zeros(N_sources)

fread = fits.open('/home/alberto/almacen/prolly_useful_files/spAllLine-v5_13_0.fits')
lineinfo = fread[1].data[np.where(fread[1].data['LINEWAVE'] == 1215.67)]
lineinfo_NV = fread[1].data[np.where(fread[1].data['LINEWAVE'] == 1240.81)]

for src in range(N_sources):
    if src % 100 == 0:
        print(f'{src} / {N_sources}', end='\r')

    where_mjd = np.where(lineinfo['MJD'] == mjd[src])
    where_mjd_pl = np.where(lineinfo['PLATE'][where_mjd] == plate[src])
    where_mjd_pl_fi = np.where(lineinfo['FIBERID'][where_mjd[0][where_mjd_pl]] == fiber[src])

    where = where_mjd[0][where_mjd_pl[0][where_mjd_pl_fi]]

    lya_F[src] = lineinfo['LINEAREA'][where]
    NV_F[src] = lineinfo_NV['LINEAREA'][where]
    lya_F_err[src] = lineinfo['LINEAREA_ERR'][where]
    NV_F_err[src] = lineinfo_NV['LINEAREA_err'][where]
    lya_cont[src] = lineinfo['LINECONTLEVEL'][where]
    lya_cont_err[src] = lineinfo['LINECONTLEVEL_ERR'][where]
    lya_z[src] = lineinfo['LINEZ'][where]
    lya_EW[src] = lineinfo['LINEEW'][where]
    NV_EW[src] = lineinfo_NV['LINEEW'][where]
    lya_EW_err[src] = lineinfo['LINEEW_ERR'][where]
    NV_EW_err[src] = lineinfo_NV['LINEEW_ERR'][where]

# neg_cont = np.where(lya_cont < 0)
# lya_cont_corrected = np.copy(lya_cont)
# lya_cont_corrected[neg_cont] = np.abs(lya_cont_err[neg_cont])

data = {
    'LyaEW': lya_EW,
    'LyaEW_err': lya_EW_err,
    'LyaF': np.abs(lya_F),
    'LyaF_err': lya_F_err,
    'NVF': NV_F,
    'NVF_err': NV_F_err,
    'NVEW': NV_EW,
    'NVFEW_err': NV_EW_err,
    'LyaCont': lya_cont,
    'LyaCont_err': lya_cont_err,
    'Lya_z': z_qso,
    'mjd': mjd,
    'plate': plate,
    'fiberid': fiber
}

pd.DataFrame(data).to_csv('csv/Lya_fts_DR16_v2.csv')