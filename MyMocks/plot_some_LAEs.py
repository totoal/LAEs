import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_LAE(sp_flx, w_Arr, z_Arr, n=4):
    fig, axes = plt.subplots(n, n, figsize=(7, 7))

    # Randomly select n spectra to plot
    selection = np.random.permutation(np.arange(len(sp_flx)))[:n**2]
    for i, ax in enumerate(axes.flat):
        sel = selection[i]
        ax.plot(w_Arr / (1 + z_Arr[sel]), sp_flx[sel] * 1e17)

    plt.show()


if __name__ == '__main__':
    cat_name = 'LAE_0.1deg_z2-4.25_train_minijpas_VUDS_0'
    path_name = '/home/alberto/almacen/Source_cats'
    w_Arr = np.load(f'{path_name}/{cat_name}/w_Arr.npy')
    z_Arr = pd.read_csv(f'{path_name}/{cat_name}/data1.csv').to_numpy()[:, -3]
    sp_flx = pd.read_csv(f'{path_name}/{cat_name}/SEDs1.csv').to_numpy()

    plot_LAE(sp_flx, w_Arr, z_Arr)