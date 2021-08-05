import numpy as np
import matplotlib.pyplot as plt

mock = np.load('Source_cat_LAE_1deg.npy', allow_pickle = True).item()

grid_dim = 6


fig, ax = plt.subplots(grid_dim, grid_dim, sharex = True)
for i in range(grid_dim):
    for j in range(grid_dim):
        idx = np.random.randint(0,33)
        ax[i, j].plot(mock['w_Arr'], mock['SEDs'][idx])
plt.show()
