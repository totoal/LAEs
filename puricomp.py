import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from my_functions import *
import seaborn as sns

w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))

def main(mag_cut):
    # Load QSO catalog
    filename = ('/home/alberto/cosmos/JPAS_mocks_sep2021/'
        'JPAS_mocks_classification_01sep_model11/Fluxes/Qso_jpas_mock_flam_train.cat')

    my_filter_order = np.arange(60)
    my_filter_order[[-4, -3, -2, -1]] = np.array([1, 12, 28, 43])
    my_filter_order[1:-4] += 1
    my_filter_order[12:-4] += 1
    my_filter_order[28:-4] += 1
    my_filter_order[43:-4] += 1

    pm_flx = pd.read_csv(
        filename, sep=' ', usecols=range(2, 2 + 60)
    ).to_numpy().T[my_filter_order]
    pm_err = pd.read_csv(
        filename, sep=' ', usecols=range(2 + 60, 2 + 60 + 60)
    ).to_numpy().T[my_filter_order]
    zspec = pd.read_csv(filename, sep=' ', usecols=[127]).to_numpy().reshape(-1, )

    # Apply errors
    np.random.seed(22)
    pm_flx += pm_err * np.random.normal(size=pm_err.shape)

    Lya_fts = pd.read_csv('csv/Lya_fts.csv')
    EW_lya = Lya_fts.LyaEW

    w_lya = 1215.67 # A
    N_sources = pm_flx.shape[1]

    mag = flux_to_mag(pm_flx, w_central.reshape(-1, 1))
    mag[np.isnan(mag)] = 99.

    zspec_dist = cosmo.luminosity_distance(zspec).to(u.cm).value
    L = EW_lya * Lya_fts.LyaCont * 1e-17 * 4*np.pi * zspec_dist**2
    L = np.log10(L)
    L[np.isnan(L)] = -99

    # Lya search
    cont_est_lya, cont_err_lya = estimate_continuum(pm_flx, pm_err)

    # Other lines
    cont_est_other, cont_err_other = estimate_continuum(pm_flx, pm_err, IGM_T_correct=False)

    ew0lya_min = 0
    ew0lya_max = 70
    ew0lya_step = 8
    ew0oth_min = 0
    ew0oth_max = 50
    ew0oth_step = 11

    other_select_list = []
    for ew0min in np.linspace(ew0oth_min, ew0oth_max, ew0oth_step):
        print(ew0min)
        line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other, ew0min)
        other_lines = identify_lines(line_other, pm_flx, pm_err, first=False)

        other_select_list.append(other_lines)

    lya_select_list = []
    lya_z_nb = []
    for ew0min in np.linspace(ew0lya_min, ew0lya_max, ew0lya_step):
        print(ew0min)
        line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0min)
        lya_lines, lya_cont_lines = identify_lines(line, pm_flx, pm_err, first=True)
        z_nb_Arr = np.ones(N_sources) * -1 
        z_nb_Arr[np.where(np.array(lya_lines) != -1)] =\
            z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])
        
        lya_select_list.append(lya_lines)
        lya_z_nb.append(z_nb_Arr)

    select_grid = np.zeros((ew0lya_step, ew0oth_step))
    rightz_grid = np.zeros((ew0lya_step, ew0oth_step))

    target = (
        (mag[-2] < mag_cut) & (EW_lya / (1 + zspec) > 20) & (zspec > 2.12) & (zspec < 4.5)
    )

    for i in range(ew0lya_step):
        print(i)
        for j in range(ew0oth_step):
            nice_lya = nice_lya_select(
                lya_select_list[i], other_select_list[j], pm_flx, cont_est_other, lya_z_nb[i]
            )
            nice_z = np.abs(lya_z_nb[i] - zspec) < 0.12

            select_grid[i, j] = len(np.where((mag[-2] < mag_cut) & nice_lya)[0])
            rightz_grid[i, j] = len(np.where((target & nice_lya & nice_z))[0])

    N_target = len(np.where(target)[0])
    purity = rightz_grid / select_grid
    completeness = rightz_grid / N_target

    fig = plt.figure(figsize=(8, 8))

    width = 0.5
    height = 0.5
    spacing = 0.04
    cbar_width = 0.05

    # Define axes
    ax00 = fig.add_axes([0, height + 2 * spacing, width, height])
    ax01 = fig.add_axes([width + spacing, height + 2 * spacing, width, height], sharey=ax00)
    ax10 = fig.add_axes([0, 0, width, height], sharex=ax00)
    ax11 = fig.add_axes([width + spacing, 0, width, height], sharex=ax01, sharey=ax10)
    axcbar0 = fig.add_axes([2 * width + 2 * spacing, height + 2 * spacing, cbar_width, height])
    axcbar1 = fig.add_axes([2 * width + 2 * spacing, 0, cbar_width, height])

    # Plot stuff in the rectangles
    vmax = np.max([np.max(rightz_grid), np.max(select_grid)])
    cmap = 'Spectral'

    sns.heatmap(rightz_grid, ax=ax00, vmin=0, vmax=vmax, cbar_ax=axcbar0, cmap=cmap)
    sns.heatmap(select_grid, ax=ax01, vmin=0, vmax=vmax, cbar_ax=axcbar0, cmap=cmap)

    sns.heatmap(purity, ax=ax10, vmin=0, vmax=1, cbar_ax=axcbar1)
    sns.heatmap(completeness, ax=ax11, vmin=0, vmax=1, cbar=False)

    ax00.invert_yaxis()
    ax10.invert_yaxis()

    # Axes ticks
    ax00.tick_params(bottom=False, labelbottom=False)
    ax01.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax11.tick_params(left=False, labelleft=False)

    xticks = range(ew0oth_step)
    yticks = range(ew0lya_step)
    xtick_labels = ['{0:0.0f}'.format(n) for n in np.linspace(ew0oth_min, ew0oth_max, ew0oth_step)]
    ytick_labels = ['{0:0.0f}'.format(n) for n in np.linspace(ew0lya_min, ew0lya_max, ew0lya_step)]

    ax00.set_yticks(yticks)
    ax00.set_yticklabels(ytick_labels)
    ax10.set_yticks(yticks)
    ax10.set_yticklabels(ytick_labels)
    ax10.set_xticks(xticks)
    ax10.set_xticklabels(xtick_labels)
    ax11.set_xticks(xticks)
    ax11.set_xticklabels(xtick_labels)

    # Set titles
    ax00.set_title('Selected w/ correct z', fontsize=15)
    ax01.set_title('All selected', fontsize=15)
    ax10.set_title('Purity', fontsize=15)
    ax11.set_title('Completeness', fontsize=15)

    plt.savefig(
        'output/puri-comp_magcut-'+str(mag_cut)+'.pdf' , bbox_inches='tight' , dpi=600
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    combined = (completeness + purity) / 2

    sns.heatmap(combined, ax=ax)

    xticks = range(ew0oth_step)
    yticks = range(ew0lya_step)
    xtick_labels = ['{0:0.0f}'.format(n) for n in np.linspace(ew0oth_min, ew0oth_max, ew0oth_step)]
    ytick_labels = ['{0:0.0f}'.format(n) for n in np.linspace(ew0lya_min, ew0lya_max, ew0lya_step)]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)

    ax.invert_yaxis()

    ax.set_title(r'(Purity + Completeness) / 2', fontsize=15)

    plt.savefig(
        'output/puri_plus_comp_magcut-'+str(mag_cut)+'.pdf' , bbox_inches='tight' , dpi=600
    )

if __name__ == '__main__':
    main(20)
    main(21)
    main(22)