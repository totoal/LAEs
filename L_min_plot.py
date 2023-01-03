import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import numpy as np
from my_functions import *

w_central = central_wavelength()
w_lya = 1215.67


def L_lim(nb_c, ew0_lim, survey, sigma=3):
    if survey == 'minijpas':
        detec_lim = np.vstack(
            (
                pd.read_csv('csv/5sigma_depths_NB.csv', header=None),
                pd.read_csv('csv/5sigma_depths_BB.csv', header=None)
            )
        )[:, 1]
    elif survey == 'jnep':
        detec_lim = pd.read_csv(
            'csv/jnep.TileImage.csv', sep=',', header=1
        )['DEPTH3ARC5S'].to_numpy().flatten()
    else:
        raise(f'Survey {survey} not recognized.')

    flambda_lim = mag_to_flux(detec_lim[nb_c], w_central[nb_c]) * sigma

    z = w_central[nb_c] / w_lya - 1
    Fline_lim = ew0_lim * flambda_lim * (1 + z)
    dL = cosmo.luminosity_distance(z).to(u.cm).value
    L_lim = np.log10(Fline_lim * 4*np.pi * dL**2)

    return L_lim


filter_tags = load_filter_tags()
data_tab = Table.read('fits/FILTERs_table.fits', format='fits')
# cmap = data_tab['color_representation']

# PLOT
fig, ax = plt.subplots(figsize=(4, 3.5))

NB_idx = np.arange(1, 20)
BB_idx = np.arange(20, 60)

L_min_minijpas = L_lim(np.arange(1, 60), 30, 'minijpas')
L_min_jnep = L_lim(np.arange(1, 60), 30, 'jnep')

for nb in NB_idx:
    ax.plot(
        w_central[nb], L_min_minijpas[nb], ls='', marker='o', markersize=7,
        markeredgecolor='r', markerfacecolor='none',
        label='miniJPAS' if nb == 12 else ''
    )
for nb in NB_idx:
    ax.plot(
        w_central[nb], L_min_jnep[nb], ls='', marker='^', markersize=7,
        markeredgecolor='g', markerfacecolor='none',
        label='J-NEP' if nb == 13 else ''
    )

ax.set_xlabel(r'$\lambda_\mathrm{NB}$ [$\mathrm{\AA}$]', fontsize=15)
ax.set_ylabel(r'$\log L_{\mathrm{Ly}\alpha}^\mathrm{min}$ [erg$\,$s$^{-1}$]', fontsize=15)
ax.tick_params(direction='in', which='both', labelsize=11)
ax.set_ylim(43.3, 44.5)
ax.set_xlim(3500, 6100)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.set_axisbelow(False)

ax.legend()

filename = 'figures/NB_L_lims.pdf'
plt.savefig(filename, bbox_inches='tight', facecolor='w', edgecolor='w')
# plt.show()
plt.close()
