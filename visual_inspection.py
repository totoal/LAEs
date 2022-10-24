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


def plot_jspectra_images(pm_flx, pm_err, cont_est, cont_err,
                         tile_id, number, x_im, y_im, nb_sel, other_lines,
                         plot_text, n_src, dirname, spec=None, g_band=None):

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
    # im_max = np.max([im_r.max(), im_nb.max()])
    # im_min = np.min([im_r.min(), im_nb.min()])

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
    # Draw line on other lines selected
    for nb in other_lines:
        print(nb)
        ax.axvline(w_central[nb], ls='--', c='orange', zorder=-90)

    # Zero line
    ax.axhline(0, linewidth=1, ls='-', c='k', zorder=-99)

    ax.set_xlim(3000, 9600)

    # Plot the continuum
    ax.errorbar(w_central[1:40], cont_est[1:40] * 1e17,
                yerr=cont_err[1:40] * 1e17, c='k', ls='-')

    #### Plot SDSS spectrum if available ####
    if g_band is not None and spec is not None:
        # Normalizing factor:
        norm = pm_flx[-3] / g_band
        spec_flx = spec['MODEL'] * norm
        spec_w = 10 ** spec['LOGLAM']

        ax.plot(spec_w, spec_flx, c='dimgray', zorder=-99, alpha=0.7)

    #########################################

    wh = 0.25
    ax1 = fig.add_axes([1 - 1.5 * wh - 0.1, 0.9, wh, wh])
    ax2 = fig.add_axes([1 - wh - 0.1, 0.9, wh, wh])

    ax1.tick_params(axis='both', bottom=False, top=False,
                    labelbottom=False, labeltop=False,
                    right=False, left=False,
                    labelright=False, labelleft=False)
    ax2.tick_params(axis='both', bottom=False, top=False,
                    labelbottom=False, labeltop=False,
                    right=False, left=False,
                    labelright=False, labelleft=False)

    # ax1.imshow(im_r, cmap='binary', vmin=im_min, vmax=im_max)
    # ax2.imshow(im_nb, cmap='binary', vmin=im_min, vmax=im_max)
    ax1.imshow(im_r, cmap='binary')
    ax2.imshow(im_nb, cmap='binary')

    # Add circumference showing aperture 3arcsec diameter
    aper_r_px = 1.5 / 0.23
    circ1 = plt.Circle((box_side, box_side),
                       radius=aper_r_px, ec='yellow', fc='none')
    circ2 = plt.Circle((box_side, box_side),
                       radius=aper_r_px, ec='yellow', fc='none')
    ax1.add_patch(circ1)
    ax2.add_patch(circ2)

    tile_name = tile_dict[tile_id]
    ax1.set_title('rSDSS')
    ax2.set_title(filter_labels[nb_sel])
    
    ypos = ax.get_ylim()[1] * 1.05
    for iii, [xpos, string] in enumerate(plot_text):
        if spec is None:
            if iii == 1:
                continue
        ax.text(xpos, ypos, string)

    # plt.show(block=True)
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f'{dirname}/{n_src}-{tile_name}-{src}.png',
                bbox_inches='tight', facecolor='w',
                edgecolor='w', dpi=500)
    plt.close()

def plot_paper(pm_flx, pm_err, cont_est, cont_err,
               tile_id, number, x_im, y_im, nb_sel, other_lines,
               plot_text, n_src, dirname, redshift, spec=None, g_band=None):
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
    # im_max = np.max([im_r.max(), im_nb.max()])
    # im_min = np.min([im_r.min(), im_nb.min()])

    fig = plt.figure(figsize=(6, 2.4))

    ax = plot_JPAS_source(pm_flx, pm_err, e17scale=True, fs=11)

    # Zero line
    ax.axhline(0, linewidth=1, ls='-', c='k', zorder=-99)

    ax.set_xlim(3000, 9600)

    # Plot the continuum
    ax.errorbar(w_central[1:40], cont_est[1:40] * 1e17,
                yerr=cont_err[1:40] * 1e17, c='k', ls='-')

    #### Plot SDSS spectrum if available ####
    if g_band is not None and spec is not None:
        # Normalizing factor:
        norm = pm_flx[-3] / g_band
        spec_flx = spec['MODEL'] * norm
        spec_w = 10 ** spec['LOGLAM']

        ax.plot(spec_w, spec_flx, c='dimgray', zorder=-99, alpha=0.7)
    
    ylim = list(ax.get_ylim())
    ylim[1] = np.max([ylim[1], spec_flx.max() * 1.1])

    ax.set_ylim(ylim)

    text_h = ax.get_ylim()[1] * 1.05
    # Draw line on the selected NB
    ax.axvline(w_central[nb_sel], color='r', linestyle='--')
    ax.text(w_central[nb_sel] + 0.1, text_h,
            r'Ly$\alpha$', fontsize=8, color='dimgray')
    # Draw other important features
    for name, w_value in spec_fts.items():
        if w_value > 9500:
            continue
        ax.axvline(w_value, color='dimgray', linestyle=':')
        if name == 'Ly lim':
            if redshift < 4:
                continue
            this_text_h = text_h * 1.065
        else:
            this_text_h = text_h
        ax.text(w_value - 100, this_text_h, name,
                color='dimgray', fontsize=8, in_layout=True)
    # Draw line on other lines selected
    for nb in other_lines:
        print(nb)
        ax.axvline(w_central[nb], ls='--', c='orange', zorder=-90)

    #########################################

    wh = 0.25
    ax1 = fig.add_axes([1 - 1.5 * wh - 0.1, 0.61, wh, wh])
    ax2 = fig.add_axes([1 - wh - 0.1, 0.61, wh, wh])

    ax1.tick_params(axis='both', bottom=False, top=False,
                    labelbottom=False, labeltop=False,
                    right=False, left=False,
                    labelright=False, labelleft=False)
    ax2.tick_params(axis='both', bottom=False, top=False,
                    labelbottom=False, labeltop=False,
                    right=False, left=False,
                    labelright=False, labelleft=False)

    # ax1.imshow(im_r, cmap='binary', vmin=im_min, vmax=im_max)
    # ax2.imshow(im_nb, cmap='binary', vmin=im_min, vmax=im_max)
    ax1.imshow(im_r, cmap='binary')
    ax2.imshow(im_nb, cmap='binary')

    # Add circumference showing aperture 3arcsec diameter
    aper_r_px = 1.5 / 0.23
    circ1 = plt.Circle((box_side, box_side),
                       radius=aper_r_px, ec='yellow', fc='none')
    circ2 = plt.Circle((box_side, box_side),
                       radius=aper_r_px, ec='yellow', fc='none')
    ax1.add_patch(circ1)
    ax2.add_patch(circ2)

    tile_name = tile_dict[tile_id]
    ax1.set_xlabel('rSDSS', fontsize=9)
    ax2.set_xlabel(filter_labels[nb_sel], fontsize=9)

    ax.tick_params(labelsize=9, direction='in', which='both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f'{dirname}/{n_src}-{tile_name}-{src}.png',
                bbox_inches='tight', facecolor='w',
                edgecolor='w', dpi=500, pad_inches=0)
    plt.close()

def nanomaggie_to_flux(nmagg, wavelength):
    mAB = -2.5 * np.log10(nmagg * 1e-9)
    flx = mag_to_flux(mAB, wavelength)
    return flx


if __name__ == '__main__':
    selection = pd.read_csv('csv/selection.csv')
    sel_x_im = selection['x_im']
    sel_y_im = selection['y_im']
    puri = selection['puri']

    print('Loading catalogs...')
    pm_flx, pm_err, x_im, y_im, tile_id, number, starprob, spCl,\
            photoz, photoz_chi_best, photoz_odds = load_minijpas_jnep(selection=True)
    N_sel = len(selection['src'])

    # Estimate the continuum to plot it
    cont_est_lya, cont_err_lya = estimate_continuum(pm_flx, pm_err, IGM_T_correct=False)

    sdss_xm_num, sdss_xm_tid, sdss_xm_spObjID = load_sdss_xmatch() 

    times_selected = np.load('tmp/times_selected.npy')

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
        this_x_im = selection['x_im'][n].astype(int)
        this_y_im = selection['y_im'][n].astype(int)
        nb = selection['nb_sel'][n].astype(int)
        other_lines = selection['other_lines'][n]
        z_src = z_NB(nb)[0]
        NB_snr = pm_flx[nb, src] / pm_err[nb, src]

        oth_raw_list = other_lines[1:-1].split()
        if len(oth_raw_list) == 0:
            oth_list = []
        else:
            oth_list = [int(item[:-1]) for item in oth_raw_list[:-1]] + [int(oth_raw_list[-1])]

        # Text to write besides the plot for info
        z_NB_name = '$z_\mathrm{NB}$'
        z_spec_name = '$z_\mathrm{spec}$'
        Log_LLya_name = r'$\log L_{\mathrm{Ly}\alpha}$'
        EW_name = r'EW$_{\mathrm{Ly}\alpha, 0}$'
        photoz_name = r'$z_\mathrm{phot}$'

        # Direct info from the catalogs and method
        ts = times_selected[src] * 0.2
        if ts >= 50:
            ts_color = 'green'
        else:
            ts_color = 'red'
        text_plot_0 = (f'#{n}\n'
                       f'\n{z_NB_name} = {z_src:0.2f}'
                       f'\n{Log_LLya_name} = {selection["L_lya"][n]:0.2f}'
                       f'\n{EW_name} = {selection["EW_lya"][n]:0.2f} $\AA$'
                       f'\nstarprob = {starprob[where_mjj]}'
                       f'\nTimes selected = {ts:0.1f} %'
                       f'\npuri = {puri[n]:0.2f}')

        # SDSS spectroscopic info
        text_plot_1 = (f'{z_spec_name} = {selection["SDSS_zspec"][n]:0.2f}'
                       f'\nspCl = {selection["SDSS_spCl"][n]}')

        # Photo_z
        text_plot_2 = (f'{photoz_name} = {photoz[where_mjj]:0.2f}'
                       f'\nodds = {photoz_odds[where_mjj]:0.2f}'
                       f'\n$\chi^2$ = {photoz_chi_best[where_mjj]:0.2f}')

        # r band and NB S/N
        text_plot_3 = (f'$r$ = {selection["r"][n]:0.2f}'
                      f'\nNB S/N = {NB_snr:0.1f}')

        text_plot = [[3000, text_plot_0],
                     [5050, text_plot_1],
                     [6100, text_plot_2],
                     [9000, text_plot_3],]

        dirname = '/home/alberto/almacen/Selected_LAEs/with_spec_info'
        if not spec_bool:
            # plot_jspectra_images(*args)
            pass
        else:
            args = (pm_flx[:, src], pm_err[:, src],
                    cont_est_lya[:, src], cont_err_lya[:, src],
                    this_tile_id, this_number, this_x_im, this_y_im, nb,
                    oth_list, text_plot, n, dirname)

            plot_jspectra_images(*args, spec, g_band)

            dirname = '/home/alberto/almacen/Selected_LAEs/paper_no_other_lines'
            args = (pm_flx[:, src], pm_err[:, src],
                    cont_est_lya[:, src], cont_err_lya[:, src],
                    this_tile_id, this_number, this_x_im, this_y_im, nb,
                    oth_list, text_plot, n, dirname)
            plot_paper(*args, z_src, spec, g_band)
            
        ####

        dirname = '/home/alberto/almacen/Selected_LAEs/no_spec_info'
        args = (pm_flx[:, src], pm_err[:, src],
                cont_est_lya[:, src], cont_err_lya[:, src],
                this_tile_id, this_number, this_x_im, this_y_im, nb,
                oth_list, text_plot, n, dirname)

        plot_jspectra_images(*args)