import glob

import pandas as pd

import numpy as np

from my_utilities import *

def main():
    files = glob.glob('QSO_Spectra/')
    N_sources = len(files)

    pm_SEDs = np.empty((60, N_sources))

    tcurves = np.load('../npy/tcurves.npy', allow_pickle=True).item()

    w_min  = 2500   # Minimum wavelength
    w_max  = 10000  # Maximum wavelegnth
    N_bins = 10000  # Number of bins
    w_Arr = np.linspace(w_min, w_max, N_bins)

    for src in N_sources:
        pm_SEDs[:, src] = JPAS_synth_phot(spec_flx, w_Arr, tcurves)
 