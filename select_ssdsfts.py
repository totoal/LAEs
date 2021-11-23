from astropy.io import fits
import pandas as pd
import numpy as np

filename = ('/home/alberto/cosmos/JPAS_mocks_sep2021/'
    'JPAS_mocks_classification_01sep_model11/Fluxes/Qso_jpas_mock_flam_train.cat')

plate = pd.read_csv(filename, sep=' ', usecols=[122]).to_numpy().reshape(-1, )
mjd = pd.read_csv(filename, sep=' ', usecols=[123]).to_numpy().reshape(-1, )
fiber = pd.read_csv(filename, sep=' ', usecols=[124]).to_numpy().reshape(-1, )

N_sources = len(fiber)

lya_cont = np.zeros(N_sources)
lya_cont_err = np.zeros(N_sources)
lya_F = np.zeros(N_sources)

fread = fits.open('fits/spAllLine-v5_13_0.fits')
lineinfo = fread[1].data[np.where(fread[1].data['LINEWAVE'] == 1215.67)]

for src in range(N_sources):
    print(src)

    where = (
        (lineinfo['MJD'] == mjd[src])
        & (lineinfo['PLATE'] == plate[src])
        & (lineinfo['FIBERID'] == fiber[src])
    )

    lya_F[src] = lineinfo['LINEAREA'][where]
    lya_cont[src] = lineinfo['LINECONTLEVEL'][where]
    lya_cont_err[src] = lineinfo['LINECONTLEVEL_ERR'][where]

neg_cont = np.where(lya_cont < 0)
lya_cont[neg_cont] = lya_cont_err[neg_cont]

data = {
    'LyaEW': lya_F / lya_cont,
    'LyaCont': lya_cont,
    'LyaCont_err': lya_cont_err
}

pd.DataFrame(data).to_csv('Lya_fts.csv')