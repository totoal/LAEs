import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 15})
import seaborn as sns

tile_dict = {
    'minijpasAEGIS001': 2241,
    'minijpasAEGIS002': 2243,
    'minijpasAEGIS003': 2406,
    'minijpasAEGIS004': 2470,
    'jnep': 2520
}

def puricomp2d_plot(puri, comp, L_bins, r_bins, dirname, survey_name, select,
                    L_Arr=None, L_Arr_e=None, mag=None):

    # L and r from the selected sources in this {survey_name}
    mask_this_tile = (select['tile_id'] == tile_dict[survey_name])
    selec_r = select['r'][mask_this_tile]
    selec_L = select['L_lya'][mask_this_tile]
    selec_L_err = select['L_lya_err'][mask_this_tile]

    fig = plt.figure(figsize=(5, 5))

    width = 1
    height = 1
    cbar_width = 0.06

    # ADD AXES
    ax0 = fig.add_axes([0, 0, width, height])
    axc0 = fig.add_axes([width + 0.02, 0, cbar_width, height])
    ax1 = fig.add_axes([width + 0.02 + 0.12 + cbar_width,
                       0, width, height], sharey=ax0)
    axc1 = fig.add_axes([width * 2 + 0.02 * 2 + 0.12 +
                        cbar_width, 0, cbar_width, height])

    # Mask puri and comp where at least one of them is zero or nan
    mask_puricomp = ~(np.isfinite(puri) & np.isfinite(comp)
                      & (puri > 0) & (comp > 0))
    puri[mask_puricomp] = np.nan
    comp[mask_puricomp] = np.nan

    # PLOT STUFF
    cmap = 'Spectral'
    sns.heatmap(puri.T, ax=ax0, vmin=0, vmax=1,
                cbar_ax=axc0, cmap=cmap, rasterized=True)
    sns.heatmap(1 / comp.T, ax=ax1, vmin=0, vmax=5,
                cbar_ax=axc1, cmap=cmap, rasterized=True)

    # If L_Arr is not None, plot the selected sources
    # Change units to plot:
    def L_to_bins(L_Arr):
        return np.interp(L_Arr, L_bins, np.arange(len(L_bins)))
    def r_to_bins(mag):
        return np.interp(mag, r_bins, np.arange(len(r_bins)))

    L_Arr_b = L_to_bins(selec_L)
    mag_b = r_to_bins(selec_r)
    L_err_Arr_b = (L_to_bins(selec_L + selec_L_err)
                   - L_to_bins(selec_L - selec_L_err)) * 0.5

    for ax in [ax0, ax1]:
        ax.errorbar(L_Arr_b, mag_b, fmt='s', xerr=L_err_Arr_b,
                    color='k', capsize=3, linestyle='')

    # TICKS
    xticks = range(len(L_bins))[1::7]  # Only odd xticks
    yticks = range(len(r_bins))[1::15]  # Only odd yticks
    xtick_labels = ['{0:0.1f}'.format(n)
                    for n in L_bins][1::7]  # Only odd xticks
    ytick_labels = ['{0:0.1f}'.format(n)
                    for n in r_bins][1::15]  # Only odd yticks
    comp_cbar_ticks = np.arange(6)
    comp_cbar_tick_labels = [0, 1, 2, 3, 4, '$>$5']

    ax0.set_yticks(yticks)
    ax0.set_yticklabels(ytick_labels, rotation='horizontal')
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xtick_labels, rotation='vertical')
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(axis='y', direction='in', labelsize=16)
    ax0.tick_params(axis='x', direction='in', labelsize=16)
    axc1.set_yticks(comp_cbar_ticks)
    axc1.set_yticklabels(comp_cbar_tick_labels)

    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels, rotation='horizontal')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels, rotation='vertical')
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(axis='y', direction='in', labelsize=16, labelleft=False)
    ax1.tick_params(axis='x', direction='in', labelsize=16)

    axc0.tick_params(labelsize=16)
    axc1.tick_params(labelsize=16)

    # SPINES
    ax0.spines[:].set_visible(True)
    ax1.spines[:].set_visible(True)

    # TITLES
    ax0.set_title('Purity', fontsize=25)
    ax1.set_title('N correction', fontsize=25)

    # AXES LABELS
    ax0.set_xlabel(r'Measured $\log L_{\mathrm{Ly}\alpha}$ (erg s$^{-1}$)', fontsize=22)
    ax1.set_xlabel(r'Measured $\log L_{\mathrm{Ly}\alpha}$ (erg s$^{-1}$)', fontsize=22)
    ax0.set_ylabel('$r$ (magAB)', fontsize=22)
    # ax1.set_ylabel('$r$ (magAB)', fontsize=22)

    # AXES LIMITS
    ax0.set_xlim(90, 160)
    ax0.set_ylim(199, 20)
    ax1.set_xlim(90, 160)
    ax1.set_ylim(199, 20)

    plt.savefig(f'{dirname}/PuriComp2D_{survey_name}.pdf',
                bbox_inches='tight', facecolor='white',)
    plt.close()

    ####################
    # Alternative Plot #
    ####################
    fig, ax = plt.subplots(figsize=(5, 5))

    ax_cbar = fig.add_axes([0.92, 0.1, 0.05, 0.79])
    
    correction = puri.T / comp.T
    # correction[~np.isfinite(correction)] = 0.
    sns.heatmap(correction, ax=ax, vmin=0, vmax=2, cbar_ax=ax_cbar, cmap=cmap,
                rasterized=True)

    # Change units to plot:
    def L_to_bins(L_Arr):
        return np.interp(L_Arr, L_bins, np.arange(len(L_bins)))
    def r_to_bins(mag):
        return np.interp(mag, r_bins, np.arange(len(r_bins)))

    L_Arr_b = L_to_bins(selec_L)
    mag_b = r_to_bins(selec_r)
    L_err_Arr_b = (L_to_bins(selec_L + selec_L_err)
                   - L_to_bins(selec_L - selec_L_err)) * 0.5

    ax.errorbar(L_Arr_b, mag_b, fmt='s', xerr=L_err_Arr_b,
                color='k', capsize=3, linestyle='')

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, rotation='horizontal')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation='vertical')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='y', direction='in', labelsize=16)
    ax.tick_params(axis='x', direction='in', labelsize=16)
    ax_cbar.tick_params(labelsize=12)
    ax.spines[:].set_visible(True)
    ax.set_xlabel(r'$\log L_{\mathrm{Ly}\alpha}$ (erg s$^{-1}$)', fontsize=22)
    ax.set_ylabel('$r$ (magAB)', fontsize=22)
    ax.set_xlim(90, 160)
    ax.set_ylim(199, 20)

    plt.savefig(f'{dirname}/PuriComp2D_{survey_name}_alt.pdf',
                bbox_inches='tight', facecolor='white',)
    plt.close()


def load_puricomp1d(dirname):
    comp_list = [
        np.load(f'{dirname}/comp1d_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/comp1d_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/comp1d_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/comp1d_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/comp1d_jnep.npy'),
    ]
    comp_den_list = [
        np.load(f'{dirname}/comp_denominator_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/comp_denominator_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/comp_denominator_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/comp_denominator_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/comp_denominator_jnep.npy'),
    ]

    comp_qso_list = [
        np.load(f'{dirname}/comp_qso_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/comp_qso_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/comp_qso_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/comp_qso_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/comp_qso_jnep.npy')
    ]
    comp_den_qso_list = [
        np.load(f'{dirname}/comp_qso_denominator_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/comp_qso_denominator_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/comp_qso_denominator_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/comp_qso_denominator_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/comp_qso_denominator_jnep.npy'),
    ]

    comp_sf_list = [
        np.load(f'{dirname}/comp_sf_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/comp_sf_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/comp_sf_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/comp_sf_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/comp_sf_jnep.npy')
    ]
    comp_den_sf_list = [
        np.load(f'{dirname}/comp_sf_denominator_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/comp_sf_denominator_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/comp_sf_denominator_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/comp_sf_denominator_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/comp_sf_denominator_jnep.npy'),
    ]

    puri_list = [
        np.load(f'{dirname}/puri1d_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/puri1d_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/puri1d_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/puri1d_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/puri1d_jnep.npy')
    ]
    puri_den_list = [
        np.load(f'{dirname}/puri_denominator_minijpasAEGIS001.npy'),
        np.load(f'{dirname}/puri_denominator_minijpasAEGIS002.npy'),
        np.load(f'{dirname}/puri_denominator_minijpasAEGIS003.npy'),
        np.load(f'{dirname}/puri_denominator_minijpasAEGIS004.npy'),
        np.load(f'{dirname}/puri_denominator_jnep.npy'),
    ]

    puricomp_bins = np.load(f'{dirname}/puricomp_bins.npy')

    return comp_list, comp_qso_list, comp_sf_list, puri_list,\
        comp_den_list, comp_den_qso_list, comp_den_sf_list, puri_den_list, puricomp_bins

survey_list = [f'minijpasAEGIS00{i}' for i in np.arange(1, 5)] + ['jnep']

if __name__ == '__main__':
    # PURICOMP 2D
    LF_name = 'LF_r17-24_nb16-20_ew30_ewoth100_nb_1.0'
    LF_dirname = f'Luminosity_functions/{LF_name}'
    dirname = f'/home/alberto/cosmos/LAEs/Luminosity_functions/{LF_name}'

    L_bins = np.load(f'{dirname}/puricomp2d_L_bins.npy')
    r_bins = np.load(f'{dirname}/puricomp2d_r_bins.npy')

    dirname = './figures'

    for survey_name in survey_list:
        puri2d = np.load(f'{LF_dirname}/puri2d_{survey_name}.npy')
        comp2d = np.load(f'{LF_dirname}/comp2d_{survey_name}.npy')
        select = np.load(f'{LF_dirname}/selection.npy', allow_pickle=True)

        puricomp2d_plot(puri2d, comp2d, L_bins, r_bins, dirname, survey_name, select)