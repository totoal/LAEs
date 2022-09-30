from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from my_functions import *
from load_jpas_catalogs import load_minijpas_jnep, load_sdss_xmatch
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


def plot_jspectra_images(pm_flx, pm_err, tile_id, x_im, y_im, nb_sel, src, zspec, spec=None,
                         g_band=None):
    if tile_id == 2520:
        survey_name = 'jnep'
    else:
        survey_name = 'minijpas'

    # Lya redshift
    z_src = z_NB(nb_sel)[0]

    # Relevant spectral features
    spec_fts = {
        'Ly lim': 912 * (1 + z_src),
        'CIV': 1549.48 * (1 + z_src),
        r'Ly $\beta$': 1025.18 * (1 + z_src),
        'CIII': 1908.73 * (1 + z_src),
        'MgII': 2799.12 * (1 + z_src)
    }

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

    fig = plt.figure(figsize=(8, 3))
    ax = plot_JPAS_source(pm_flx, pm_err, e17scale=True)

    text_h = pm_flx.max() * 1e17

    # Draw line on the selected NB
    ax.axvline(w_central[nb_sel], color='r', linestyle='--')
    ax.text(w_central[nb_sel] + 0.1, text_h,
            r'Ly$\alpha$', fontsize=8, color='dimgray')
    # Draw other important features
    for name, w_value in spec_fts.items():
        if w_value > 9500:
            continue
        ax.axvline(w_value, color='dimgray', linestyle=':')
        ax.text(w_value + 0.1, text_h, name,
                color='dimgray', fontsize=8, in_layout=True)

    ax.set_xlim(3000, 9600)

    #### Plot SDSS spectrum if available ####
    if g_band is not None and spec is not None:
        # Normalizing factor:
        norm = pm_flx[-3] / g_band
        spec_flx = spec['FLUX'] * norm
        spec_w = 10 ** spec['LOGLAM']

        ax.plot(spec_w, spec_flx, c='dimgray', zorder=-99, alpha=0.7)

    #########################################

    wh = 0.25
    ax1 = fig.add_axes([1 - 1.5 * wh, 1 - wh - 0.1, wh, wh])
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
                       radius=aper_r_px, ec='yellow', fc='none')
    circ2 = plt.Circle((box_side, box_side),
                       radius=aper_r_px, ec='yellow', fc='none')
    ax1.add_patch(circ1)
    ax2.add_patch(circ2)

    tile_name = tile_dict[tile_id]
    title = f'{tile_name}-{src}  z = {z_src:0.2f}, zspec = {zspec:0.2f}'
    ax.set_title(title, fontsize=15, loc='left')
    ax1.set_title('rSDSS')
    ax2.set_title(filter_labels[nb_sel])

    # plt.show(block=True)
    dirname = '/home/alberto/almacen/selected_LAEs'
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f'{dirname}/{tile_name}-{src}.png',
                bbox_inches='tight', facecolor='w',
                edgecolor='w', dpi=500)
    plt.close()

def nanomaggie_to_flux(nmagg, wavelength):
    mAB = -2.5 * np.log10(nmagg * 1e-9)
    flx = mag_to_flux(mAB, wavelength)
    return flx


if __name__ == '__main__':
    selection = pd.read_csv('csv/selection.csv')
    sel_x_im = selection['x_im']
    sel_y_im = selection['y_im']

    print('Loading catalogs...')
    pm_flx, pm_err, x_im, y_im, tile_id, number = load_minijpas_jnep(selection=True)
    N_sel = len(selection['src'])

    sdss_xm_num, sdss_xm_tid, sdss_xm_spObjID = load_sdss_xmatch() 

    # Directory of the spectra .fits files
    fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/miniJPAS_Xmatch'

    for n in range(N_sel):
        print(f'Plotting {n + 1} / {N_sel}')

        try:
        # Look for the source in the SDSS Xmatch
            where_mjj = np.where((sel_x_im[n] == x_im) & (sel_y_im[n] == y_im))[0][0]
            this_number = int(number[where_mjj])
            this_tile_id = int(tile_id[where_mjj])

            this_spObjID = sdss_xm_spObjID.to_numpy()[(this_number == sdss_xm_num)
                                            & (this_tile_id == sdss_xm_tid)][0]

            # Disgregate SpObjID in mjd, tile, fiber
            spObj_binary = np.binary_repr(this_spObjID)
            plate = int(spObj_binary[::-1][50:64][::-1], 2)
            mjd = int(spObj_binary[::-1][24:38][::-1], 2) + 50000
            fiber = int(spObj_binary[::-1][38:50][::-1], 2)

            spec_name = f'spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits'
            print(spec_name)
            spec_bool = True
            spec = Table.read(f'{fits_dir}/{spec_name}', hdu=1, format='fits')
            g_band = Table.read(f'{fits_dir}/{spec_name}', hdu=2, format='fits')['SPECTROFLUX']
            g_band = nanomaggie_to_flux(np.array(g_band)[0][1], 4750)

        except:
            spec_bool = False
            print('No spectrum')


        src = selection['src'][n].astype(int)
        tile = selection['tile_id'][n].astype(int)
        this_x_im = selection['x_im'][n].astype(int)
        this_y_im = selection['y_im'][n].astype(int)
        nb = selection['nb_sel'][n].astype(int)
        zspec = selection['SDSS_zspec'][n]

        if not spec_bool:
            plot_jspectra_images(
                pm_flx[:, src], pm_err[:, src], tile, this_x_im, this_y_im, nb, src, zspec)
        else:
            plot_jspectra_images(
                pm_flx[:, src], pm_err[:, src], tile, this_x_im, this_y_im, nb, src, zspec, spec, g_band)
