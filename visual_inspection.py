from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
from my_functions import plot_JPAS_source, load_filter_tags
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

def plot_jspectra_images(pm_flx, pm_err, tile_id, x_im, y_im, nb_sel, src):
    filenamer = f'/home/alberto/almacen/images_fits/minijpas/{tile_id}-{59}.fits'
    filenamenb = f'/home/alberto/almacen/images_fits/minijpas/{tile_id}-{nb_sel + 1}.fits'
    
    y_range = slice(x_im - 10, x_im + 10 + 1)
    x_range = slice(y_im - 10, y_im + 10 + 1)
    im_r = fits.open(filenamer)[1].data[x_range, y_range]
    im_nb = fits.open(filenamenb)[1].data[x_range, y_range]

    fig = plt.figure(figsize=(7, 6))
    ax = plot_JPAS_source(pm_flx, pm_err, e17scale=True)
    
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

    ax1.imshow(im_r, cmap='binary')
    ax2.imshow(im_nb, cmap='binary')

    tile_name = tile_dict[tile_id]
    ax.set_title(tile_name, fontsize=15, loc='left')
    ax1.set_title('rSDSS')
    ax2.set_title(filter_labels[nb_sel])
    
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner:
    mngr.window.setGeometry(50, 100, 640, 545)

    # plt.tight_layout()
    # plt.show(block=True)
    dirname = '/home/alberto/almacen/selected_LAEs'
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f'{dirname}/{tile_name}-{src}.png',
                bbox_inches='tight', facecolor='w',
                edgecolor='w')

if __name__ == '__main__':
    with open('npy/selection.npy', 'rb') as f:
        selection = pickle.load(f)

    print('Loading catalogs...')
    pm_flx, pm_err = load_minijpas_jnep()[:2]

    N_sel = len(selection['src'])
    for n in range(N_sel):
        print(f'Plotting {n} / {N_sel}')
        src = selection['src'][n].astype(int)
        tile = selection['tile_id'][n].astype(int)
        x_im = selection['x_im'][n].astype(int)
        y_im = selection['y_im'][n].astype(int)
        nb = selection['nb_sel'][n].astype(int)

        plot_jspectra_images(pm_flx[:, src], pm_err[:, src], tile, x_im, y_im, nb, src)