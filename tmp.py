from my_functions import *
from colorplot import *
import numpy as np
import matplotlib.pyplot as plt

nf_cat = load_noflag_cat('catalogDual_pz.pkl')

nb_ind = 12
bb_ind = -3
idx = (nf_cat['MAG'][:, nb_ind] < 90) & (nf_cat['MAG'][:, bb_ind] < 90)

nb_m = nf_cat['MAG'][idx, nb_ind]
bb_m = nf_cat['MAG'][idx, bb_ind]
nb_e = nf_cat['ERR'][idx, nb_ind]
bb_e = nf_cat['ERR'][idx, bb_ind]

bbnb = bb_m - nb_m

# Define binning
m_min = 14
m_max = 26
m_bin_n = 75
x_e = np.linspace(m_min, m_max, m_bin_n)

ref_m = np.copy(bb_m)
nb_bin_e = m_err_bin(nb_m, nb_e, x_e, ref_m)
bb_bin_e = m_err_bin(bb_m, bb_e, x_e, ref_m)
bbnb_bin_e = np.sqrt(nb_bin_e**2 + bb_bin_e**2) # Median error

bbnb_bin_m = m_dev_bin(bbnb, x_e, ref_m)

m_bias = np.nanmedian(bbnb[(ref_m > 14) & (ref_m < 20)])

x_m = np.linspace(14,26,100)

Sigma = 1
err_curve = Sigma*np.interp(x_m, x_e, bbnb_bin_e) + m_bias
dev_curve = Sigma*np.interp(x_m, x_e, bbnb_bin_m) + m_bias

bbcut = x_e[np.nanargmin(np.abs(m_err_bin(bb_m, bb_e, x_e, bb_m) - 0.24))]
nbcut = x_e[np.nanargmin(np.abs(m_err_bin(nb_m, bb_e, x_e, nb_m) - 0.24))]

err_curve_bbnb = Sigma*np.interp(ref_m, x_e, bbnb_bin_e) + m_bias

n_candidates, = np.where((bbnb > err_curve_bbnb) & (nb_m < nbcut))
print(len(n_candidates))

plt.figure(figsize=(10,3))
plt.scatter(bb_m, bb_m-nb_m, marker='.')
plt.plot(bb_m, bb_m - nbcut, c='green', label='NB cut')
plt.axvline(bbcut, c='black', label='BB cut')
plt.plot(x_m, err_curve, c='red', label='Median error')
plt.plot(x_m, dev_curve, c='orange', label='MAD of magnitudes')
plt.ylim((-1, 3))
plt.xlim((14,24))
plt.xlabel('BB (mag)')
plt.ylabel('BB - NB (mag)')
plt.legend()
filename = './figures/errortest_BB'
plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0, transparent=True)