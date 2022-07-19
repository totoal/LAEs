from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
from my_functions import central_wavelength, nb_fwhm, plot_JPAS_source, load_filter_tags
from load_jpas_catalogs import load_minijpas_jnep
import os

w_lya = 1215.67

tile_dict = {
    2241: 'AEGIS001',
    2243: 'AEGIS002',
    2406: 'AEGIS003',
    2470: 'AEGIS004',
    2520: 'J-NEP'
}
filter_labels = load_filter_tags()
w_central = central_wavelength()
fwhm_Arr = nb_fwhm(np.arange(60))

# Exposure times for NB and BB in seconds
bb_exp_time = 30
nb_exp_time = 120


def plot_jspectra_images(pm_flx, pm_err, tile_id, x_im, y_im, nb_sel, src):
    if tile_id == 2520:
        survey_name = 'jnep'
    else:
        survey_name = 'minijpas'

    filenamer = f'/home/alberto/almacen/images_fits/{survey_name}/{tile_id}-{59}.fits'
    filenamenb = f'/home/alberto/almacen/images_fits/{survey_name}/{tile_id}-{nb_sel + 1}.fits'

    box_side = 16
    y_range = slice(x_im - box_side, x_im + box_side + 1)
    x_range = slice(y_im - box_side, y_im + box_side + 1)
    im_r = fits.open(filenamer)[1].data[x_range, y_range]
    im_nb = fits.open(filenamenb)[1].data[x_range, y_range]

    # Normalize by the bandwidth
    im_r = im_r / fwhm_Arr[-2] * bb_exp_time
    im_nb = im_nb / fwhm_Arr[nb_sel] * nb_exp_time

    # Get max and min of the images to establish common scale
    im_max = np.max([im_r.max(), im_nb.max()])
    im_min = np.min([im_r.min(), im_nb.min()])

    fig = plt.figure(figsize=(7, 6))
    ax = plot_JPAS_source(pm_flx, pm_err, e17scale=True)

    # Draw line on the selected NB
    ax.axvline(w_central[nb_sel], color='r', linestyle='--')

    wh = 0.23
    ax1 = fig.add_axes([1 - 2 * wh, 1 - wh - 0.1, wh, wh])
    ax2 = fig.add_axes([1 - wh, 1 - wh - 0.1, wh, wh])

    ax1.tick_params(axis='both', bottom=False, top=False,
                    labelbottom=False, labeltop=False,
                    right=False, left=False,
                    labelright=False, labelleft=False)
    ax2.tick_params(axis='both', bottom=False, top=False,
                    labelbottom=False, labeltop=False,
                    right=False, left=False,
                    labelright=False, labelleft=False)

    ax1.imshow(im_r, cmap='binary', vmin=im_min, vmax=im_max)
    ax2.imshow(im_nb, cmap='binary', vmin=im_min, vmax=im_max)

    # Add circumference showing aperture 3arcsec diameter
    aper_r_px = 1.5 / 0.23
    circ1 = plt.Circle((box_side, box_side),
                       radius=aper_r_px, ec='r', fc='none')
    circ2 = plt.Circle((box_side, box_side),
                       radius=aper_r_px, ec='r', fc='none')
    ax1.add_patch(circ1)
    ax2.add_patch(circ2)

    tile_name = tile_dict[tile_id]
    title = f'{tile_name}-{src}'
    ax.set_title(title, fontsize=15, loc='left')
    ax1.set_title('rSDSS')
    ax2.set_title(filter_labels[nb_sel])

    # plt.show(block=True)
    dirname = '/home/alberto/almacen/selected_LAEs'
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f'{dirname}/{tile_name}-{src}.png',
                bbox_inches='tight', facecolor='w',
                edgecolor='w')
    plt.close()


if __name__ == '__main__':
    with open('npy/selection.npy', 'rb') as f:
        selection = pickle.load(f)

    print('Loading catalogs...')
    pm_flx, pm_err = load_minijpas_jnep()[:2]

    N_sel = len(selection['src'])
    for n in range(N_sel):
        print(f'Plotting {n + 1} / {N_sel}')
        src = selection['src'][n].astype(int)
        tile = selection['tile_id'][n].astype(int)
        x_im = selection['x_im'][n].astype(int)
        y_im = selection['y_im'][n].astype(int)
        nb = selection['nb_sel'][n].astype(int)

        plot_jspectra_images(
            pm_flx[:, src], pm_err[:, src], tile, x_im, y_im, nb, src)
