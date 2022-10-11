import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def puricomp2d_plot(puri, comp, L_bins, r_bins, dirname, survey_name,
                    L_Arr=None, L_Arr_e=None, mag=None):
    fig = plt.figure(figsize=(5, 5))

    width = 1
    height = 1
    cbar_width = 0.06

    # ADD AXES
    ax0 = fig.add_axes([0, 0, width, height])
    axc0 = fig.add_axes([width + 0.02, 0, cbar_width, height])
    ax1 = fig.add_axes([width + 0.02 + 0.15 + cbar_width,
                       0, width, height], sharey=ax0)
    axc1 = fig.add_axes([width * 2 + 0.02 * 2 + 0.15 +
                        cbar_width, 0, cbar_width, height])

    # Mask puri and comp where at least one of them is zero or nan
    mask_puricomp = ~(np.isfinite(puri) & np.isfinite(comp)
                      & (puri > 0) & (comp > 0))
    puri[mask_puricomp] = np.nan
    comp[mask_puricomp] = np.nan

    # PLOT STUFF
    cmap = 'Spectral'
    sns.heatmap(puri.T, ax=ax0, vmin=0, vmax=1, cbar_ax=axc0, cmap=cmap)
    sns.heatmap(comp.T, ax=ax1, vmin=0, vmax=5, cbar_ax=axc1, cmap=cmap)

    # If L_Arr is not None, plot the selected sources
    if (L_Arr is not None) and (L_Arr_e is not None) and (mag is not None):
        # Change units to plot:
        def L_to_bins(L_Arr):
            return np.interp(L_Arr, L_bins, np.arange(len(L_bins)))
        def r_to_bins(L_Arr):
            return np.interp(mag, r_bins, np.arange(len(r_bins)))

        L_Arr_b = L_to_bins(L_Arr)
        mag_b = r_to_bins(mag)

        for ax in [ax0, ax1]:
            ax.errorbar(L_Arr_b, mag_b, xerr=L_Arr_e, fmt='s',
                        color='k', capsize=3, linestyle='')

    # TICKS
    xticks = range(len(L_bins))[1::2]  # Only odd xticks
    yticks = range(len(r_bins))[1::2]  # Only odd yticks
    xtick_labels = ['{0:0.1f}'.format(n)
                    for n in L_bins][1::2]  # Only odd xticks
    ytick_labels = ['{0:0.1f}'.format(n)
                    for n in r_bins][1::2]  # Only odd yticks

    ax0.set_yticks(yticks)
    ax0.set_yticklabels(ytick_labels, rotation='horizontal')
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xtick_labels, rotation='vertical')
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(axis='y', direction='in', labelsize=16)
    ax0.tick_params(axis='x', direction='in', labelsize=16)

    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels, rotation='horizontal')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels, rotation='vertical')
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(axis='y', direction='in', labelsize=16, labelleft=False,
                    length=9)
    ax1.tick_params(axis='x', direction='in', labelsize=16, length=9)

    axc0.tick_params(labelsize=16)
    axc1.tick_params(labelsize=16)

    # SPINES
    ax0.spines[:].set_visible(True)
    ax1.spines[:].set_visible(True)

    # TITLES
    ax0.set_title('Purity', fontsize=25)
    ax1.set_title('Completeness', fontsize=25)

    # AXES LABELS
    ax0.set_xlabel(r'$\logL_{\mathrm{Ly}\alpha}$ (erg s$^{-1}$)', fontsize=22)
    ax1.set_xlabel(r'$\logL_{\mathrm{Ly}\alpha}$ (erg s$^{-1}$)', fontsize=22)
    ax0.set_ylabel('$r$ (magAB)', fontsize=22)
    # ax1.set_ylabel('$r$ (magAB)', fontsize=22)

    # AXES LIMITS
    ax0.set_xlim(8, 22)
    ax1.set_xlim(8, 22)

    plt.savefig(f'{dirname}/PuriComp2D_{survey_name}.pdf',
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


def puricomp1d_plot(dirname, save_dirname, surname=''):
    comp_list, comp_qso_list, comp_sf_list, puri_list,\
        comp_den_list, comp_den_qso_list, comp_den_sf_list, puri_den_list, puricomp_bins = \
        load_puricomp1d(dirname)

    # Define the survey list in order
    survey_list = [f'AEGIS00{i}' for i in range(1, 4 + 1)] + ['J-NEP']

    # Bin centers
    bc = [puricomp_bins[i: i + 2].sum() * 0.5 for i in range(len(puricomp_bins) - 1)]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot the individual comps
    for i, comp in enumerate(comp_list):
        ax.plot(bc, comp, ls=':', alpha=0.6, marker='^', markersize=10,
                color=f'C{i + 2}', label=survey_list[i])

    # Total comp
    total_comp_num = (np.array(comp_list) *
                      np.array(comp_den_list)).sum(axis=0)
    total_comp_den = np.array(comp_den_list).sum(axis=0)
    total_comp = total_comp_num / total_comp_den
    ax.plot(bc, total_comp, ls='-', marker='s', color='black', label='Total')

    # Total SF and QSO comps
    # total_comp_num = (np.array(comp_qso_list) * np.array(comp_den_qso_list)).sum(axis=0)
    # total_comp_den = np.array(comp_den_qso_list).sum(axis=0)
    # total_qso_comp = total_comp_num / total_comp_den
    # ax.plot(bc, total_qso_comp, ls='--', color='C0', linewidth=2,
    #         label='Only QSO')
    # total_comp_num = (np.array(comp_sf_list) * np.array(comp_den_sf_list)).sum(axis=0)
    # total_comp_den = np.array(comp_den_sf_list).sum(axis=0)
    # total_sf_comp = total_comp_num / total_comp_den
    # ax.plot(bc, total_sf_comp, ls='--', color='C1', linewidth=2,
    #         label='Only SF')

    # Fraction sf/qso
    # total_comp_num_qso = (np.array(comp_qso_list) * np.array(comp_den_qso_list)).sum(axis=0)
    # total_comp_num_sf = (np.array(comp_sf_list) * np.array(comp_den_sf_list)).sum(axis=0)
    # sf_qso_frac = total_comp_num_sf / total_comp_num_qso
    # sf_qso_frac[~np.isfinite(sf_qso_frac)] = 0
    # ax.plot(bc, sf_qso_frac, color='m', label='SF / QSO')

    ax.legend(loc=0, fontsize=10)
    ax.set_xlabel(r'$\log L$ (erg$\,$s$^{-1}$)', fontsize=12)
    ax.set_ylabel('Completeness', fontsize=12)

    ax.set_ylim(0, 1)

    plt.savefig(f'{save_dirname}/Comp1D.pdf',
                bbox_inches='tight', facecolor='white',)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot the individual puris
    for i, puri in enumerate(puri_list):
        ax.plot(bc, puri, ls='--', alpha=0.6, marker='s', label=survey_list[i])

    # Total puri
    total_puri_num = (np.array(puri_list) *
                      np.array(puri_den_list)).sum(axis=0)
    total_puri_den = np.array(puri_den_list).sum(axis=0)
    total_puri = total_puri_num / total_puri_den
    total_puri[~np.isfinite(total_puri)] = 0.
    ax.plot(bc, total_puri, ls='-', marker='s', color='black', label='Total')

    ax.legend(loc=0, fontsize=10)
    ax.set_xlabel(r'$\log L$ (erg$\,$s$^{-1}$)', fontsize=12)
    ax.set_ylabel('Purity', fontsize=12)

    ax.set_ylim(0, 1)

    plt.savefig(f'{save_dirname}/Puri1D.pdf',
                bbox_inches='tight', facecolor='white',)
    plt.close()

    ### Combined plot PuriComp1D ###

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.1)

    for i, puri in enumerate(puri_list):
        axs[0].plot(bc, puri, ls='--', alpha=0.6, marker='s', markersize=10,
                    color=f'C{i + 2}', label=survey_list[i])

    for i, comp in enumerate(comp_list):
        axs[1].plot(bc, comp, ls='--', alpha=0.6, marker='s', markersize=10,
                    color=f'C{i + 2}')

    axs[0].plot(bc, total_puri, ls='-', marker='s', color='black',
                markersize=10, label='Total')
    axs[1].plot(bc, total_comp, ls='-', marker='s', color='black',
                markersize=10)

    # Font size
    fs = 15

    axs[0].legend(loc=0, fontsize=14)

    axs[1].set_xlabel(r'$\log L$ (erg$\,$s$^{-1}$)', fontsize=20)
    axs[0].set_ylabel('Purity', fontsize=20)
    axs[1].set_ylabel('Completeness', fontsize=20)

    for ax in axs:
        ax.set_ylim(0, 1)
        ax.set_xlim(42.25, 45.5)
        ax.tick_params(labelsize=fs, direction='in', length=6)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

    # fig.tight_layout()

    plt.savefig(f'{save_dirname}/PuriComp1D{surname}.pdf',
                bbox_inches='tight', facecolor='white',)
    plt.close()


survey_list = [f'minijpasAEGIS00{i}' for i in np.arange(1, 5)] + ['jnep']

if __name__ == '__main__':
    # PURICOMP 2D
    LF_dirname = 'npy'

    L_bins = np.load('npy/puricomp2d_L_bins.npy')
    r_bins = np.load('npy/puricomp2d_r_bins.npy')

    dirname = './figures'

    for survey_name in survey_list:
        puri2d = np.load(f'{LF_dirname}/puri2d_{survey_name}.npy')
        comp2d = np.load(f'{LF_dirname}/comp2d_{survey_name}.npy')

        puricomp2d_plot(puri2d, comp2d, L_bins, r_bins, dirname, survey_name)