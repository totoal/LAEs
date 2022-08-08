import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def puricomp_plot(puri, comp, L_bins, r_bins, dirname, survey_name):
    fig = plt.figure(figsize=(5, 5))

    width = 1
    height = 1
    spacing = 0.06
    cbar_width = 0.06

    # ADD AXES
    ax0 = fig.add_axes([0, 0, width, height])
    axc0 = fig.add_axes([width + 0.02, 0, cbar_width, height])
    ax1 = fig.add_axes([width + 0.02 + 0.15 + cbar_width, 0, width, height], sharey=ax0)
    axc1 = fig.add_axes([width * 2 + 0.02 + 0.15 + spacing + cbar_width, 0, cbar_width, height])

    # Mask puri and comp where at least one of them is zero or nan
    mask_puricomp = ~(np.isfinite(puri) & np.isfinite(comp) & (puri > 0) & (comp > 0))
    puri[mask_puricomp] = np.nan
    comp[mask_puricomp] = np.nan

    # PLOT STUFF
    cmap = 'Spectral'
    sns.heatmap(puri.T, ax=ax0, vmin=0, vmax=1, cbar_ax=axc0, cmap=cmap)
    sns.heatmap(comp.T, ax=ax1, vmin=0, vmax=5, cbar_ax=axc1, cmap=cmap)

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


survey_list = [f'minijpasAEGIS00{i}' for i in np.arange(1, 5)] + ['jnep']

if __name__ == '__main__':
    # PURICOMP 2D
    L_bins = np.load('npy/puricomp2d_L_bins.npy')
    r_bins = np.load('npy/puricomp2d_r_bins.npy')

    dirname = './figures'

    for survey_name in survey_list:
        puri2d = np.load(f'npy/puri2d_{survey_name}.npy')
        comp2d = np.load(f'npy/comp2d_{survey_name}.npy')

        puricomp_plot(puri2d, comp2d, L_bins, r_bins, dirname, survey_name)

    # PURICOMP 1D
    dirname = './Luminosity_functions/LF_r17-24_z2.5-3.8_ew15_ewoth400_nb'
    