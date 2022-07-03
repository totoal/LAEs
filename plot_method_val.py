import numpy as np
import matplotlib.pyplot as plt
from my_functions import double_schechter

if __name__ == '__main__':
    out = np.load('npy/method_val_out.npy', allow_pickle=True)
    L_binning = np.log10(np.load('npy/L_nb_err_binning.npy'))
    L_bin_c = [L_binning[i : i + 2].sum() * 0.5 for i in range(len(L_binning) - 1)]

    frac_list = [1, 0.9, 0.75, 0.5, 0.3]
    fracs = []
    for x in frac_list:
        for y in frac_list:
            fracs.append([x, y])

    for i, (a, b, c) in enumerate(out):
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(L_bin_c, a, marker='x', facecolor='k', label='miniJPAS')
        ax.scatter(L_bin_c, b, marker='o', facecolor='none', edgecolor='k', label='Mock')

        Lx = np.linspace(10 ** 42, 10 ** 46, 10000)
        phistar1 = 3.33e-6
        Lstar1 = 44.65
        alpha1 = -1.35

        phistar2 = -3.45
        Lstar2 = 42.93
        alpha2 = -1.93

        Phi_center = double_schechter(
            Lx, phistar1, 10 ** Lstar1, alpha1, 10 ** phistar2, 10 ** Lstar2, alpha2,
            scale1=fracs[i][0] * 0.5, scale2=fracs[i][1]
        ) * Lx * np.log(10)

        ax.plot(
            np.log10(Lx), Phi_center, ls='-.', alpha=0.5,
            label='Spinoso2020 (2.2 < z < 3.25)'
            )

        ax.set_yscale('log')
        ax.set_title(f'R$^2$ = {c:0.2f}   ({fracs[i][0]}, {fracs[i][1]})', fontsize=15)
        ax.set_xlim(42, 45.5)
        ax.set_ylim(1e-8, 1e-4)
        ax.legend(fontsize=15)
    
        plt.savefig(f'/home/alberto/Desktop/fig{i}', bbox_inches='tight', facecolor='white')
        plt.close()