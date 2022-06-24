from astropy.io import fits
import matplotlib.pyplot as plt
import pickle
from my_functions import plot_JPAS_source
from load_jpas_catalogs import load_minijpas_jnep

w_lya = 1215.67

def plot_jspectra_images(pm_flx, pm_err, tile_id, x_im, y_im, nb_sel):
    filenamer = f'/home/alberto/almacen/images_fits/minijpas/{tile_id}-{59}.fits'
    filenamenb = f'/home/alberto/almacen/images_fits/minijpas/{tile_id}-{nb_sel + 1}.fits'
    
    x_range = slice(x_im - 5, x_im + 5 + 1)
    y_range = slice(y_im - 5, y_im + 5 + 1)
    im_r = fits.open(filenamer)[1].data[x_range, y_range]
    im_nb = fits.open(filenamenb)[1].data[x_range, y_range]

    fig = plt.figure(figsize=(15, 5))
    ax = plot_JPAS_source(pm_flx, pm_err)
    
    wh = 0.23
    ax1 = fig.add_axes([1 - 2 * wh, 1 - wh, wh, wh])
    ax2 = fig.add_axes([1 - wh, 1 - wh, wh, wh])

    ax1.tick_params(axis='both', bottom=False, top=False,
                        labelbottom=False, labeltop=False)
    ax2.tick_params(axis='both', bottom=False, top=False,
                        labelbottom=False, labeltop=False)

    ax1.imshow(im_r, cmap='binary')
    ax2.imshow(im_nb, cmap='binary')
    
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner:
    mngr.window.setGeometry(50, 100, 640, 545)

    # plt.tight_layout()
    plt.show(block=False)

if __name__ == '__main__':
    # with open('npy/selection.npy', 'rb') as f:
    #     selection = pickle.load(f)

    print('Loading catalogs...')
    pm_flx, pm_err = load_minijpas_jnep()[:2]

    n = 2
    # src = selection['src'][n]
    src = 69
    # tile = selection['tile_id'][n]
    # x_im = selection['x_im'][n]
    # y_im = selection['y_im'][n]
    # nb = selection['nb_sel'][n]

    nb = 15
    y_im = 387
    x_im = 6346
    tile = 2406
    plot_jspectra_images(pm_flx[:, src], pm_err[:, src], tile, x_im, y_im, nb)