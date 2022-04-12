from astropy.io import fits
import pandas as pd
import numpy as np

filename = ('/home/alberto/cosmos/JPAS_mocks_sep2021/'
    'JPAS_mocks_classification_19nov_model11/Fluxes_model_11/Qso_jpas_mock_flam_test.cat')

plate = pd.read_csv(filename, sep=' ', usecols=[122]).to_numpy().reshape(-1,)
mjd = pd.read_csv(filename, sep=' ', usecols=[123]).to_numpy().reshape(-1,)
fiber = pd.read_csv(filename, sep=' ', usecols=[124]).to_numpy().reshape(-1,)

N_sources = len(fiber)

lya_cont = np.zeros(N_sources)
lya_cont_err = np.zeros(N_sources)
lya_F = np.zeros(N_sources)
lya_F_err = np.zeros(N_sources)
lya_z = np.zeros(N_sources)

fread = fits.open('/home/alberto/almacen/prolly_useful_files/spAllLine-v5_13_0.fits')
lineinfo = fread[1].data[np.where(fread[1].data['LINEWAVE'] == 1215.67)]

for src in range(N_sources):
    print(f'{src} / {N_sources}', end='\r')

    where = (
        (lineinfo['MJD'] == mjd[src])
        & (lineinfo['PLATE'] == plate[src])
        & (lineinfo['FIBERID'] == fiber[src])
    )

    lya_F[src] = lineinfo['LINEAREA'][where]
    lya_F_err[src] = lineinfo['LINEAREA_ERR'][where]
    lya_cont[src] = lineinfo['LINECONTLEVEL'][where]
    lya_cont_err[src] = lineinfo['LINECONTLEVEL_ERR'][where]
    lya_z[src] = lineinfo['LINEZ'][where]

neg_cont = np.where(lya_cont < 0)
lya_cont_corrected = np.copy(lya_cont)
lya_cont_corrected[neg_cont] = np.abs(lya_cont_err[neg_cont])

data = {
    'LyaEW': np.abs(lya_F) / lya_cont_corrected,
    'LyaF': np.abs(lya_F),
    'LyaF_err': lya_F_err,
    'LyaCont': lya_cont_corrected,
    'LyaCont_err': lya_cont_err,
    'Lya_z': lya_z,
    'mjd': mjd,
    'plate': plate,
    'fiberid': fiber
}

pd.DataFrame(data).to_csv('csv/Lya_fts_test.csv')