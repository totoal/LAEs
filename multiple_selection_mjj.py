from LAEs.load_mocks import ensemble_mock
from LAEs.load_jpas_catalogs import load_minijpas_jnep
from LAEs.minijpas_LF_and_puricomp import add_errors
from LAEs.my_functions import *

import threading
import time

import numpy as np

w_central = central_wavelength()
nb_fwhm_Arr = nb_fwhm(range(60))
w_lya = 1215.67


def this_selection(pm_flx, pm_err, zspec, times_selected, times_nicez,
                   ew0_cut=30, ew_other=100, nb_min=1, nb_max=24):
    N_sources = pm_flx.shape[1]
    # Lya search
    cont_est_lya, cont_err_lya = estimate_continuum(pm_flx, pm_err, IGM_T_correct=True,
                                                    N_nb_min=nb_min, N_nb_max=nb_max)
    line = is_there_line(pm_flx, pm_err, cont_est_lya, cont_err_lya, ew0_cut)
    lya_lines, lya_cont_lines, _ = identify_lines(line, pm_flx, cont_est_lya, first=True,
                                                  return_line_width=True)
    lya_lines = np.array(lya_lines)

    # Other lines
    cont_est_other, cont_err_other = estimate_continuum(pm_flx, pm_err, IGM_T_correct=False)
    line_other = is_there_line(pm_flx, pm_err, cont_est_other, cont_err_other,
        ew_other, obs=True, sigma=5)
    other_lines = identify_lines(line_other, cont_est_other, pm_err)

    # Compute z
    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])

    z_min = (w_central[nb_min] - nb_fwhm_Arr[nb_min] * 0.5)/ w_lya - 1
    z_max = (w_central[nb_max] + nb_fwhm_Arr[nb_max] * 0.5)/ w_lya - 1

    z_cut = (z_min < z_Arr) & (z_Arr < z_max)

    snr = np.empty(N_sources)
    for src in range(N_sources):
        l = lya_lines[src]
        snr[src] = pm_flx[l, src] / pm_err[l, src]

    nice_lya_mask = z_cut & (snr > 6)

    this_nice_lya = nice_lya_select(
        lya_lines, other_lines, pm_flx, pm_err, cont_est_lya, z_Arr, mask=nice_lya_mask
    )

    # Compute z
    z_Arr = np.zeros(N_sources)
    z_Arr[np.where(np.array(lya_lines) != -1)] =\
        z_NB(np.array(lya_cont_lines)[np.where(np.array(lya_lines) != -1)])
    if zspec is not None:
        nice_z = np.abs(z_Arr - zspec) < 0.16
        times_nicez += nice_z.astype(int)

    times_selected += this_nice_lya.astype(int)
    # tmp

if __name__ == '__main__':
    print('Loading mock...')
    field_name = 'minijpasAEGIS001'
    name_qso = 'QSO_100000_0'
    name_qso_bad = f'QSO_double_train_minijpas_DR16_D_0'
    name_gal = f'GAL_LC_lines_0'
    name_sf = f'LAE_12.5deg_z2-4.25_train_minijpas_0'

    pm_flx, pm_err, tile_id, pmra_sn, pmdec_sn, parallax_sn, starprob, _,\
        spCl, zsp, photoz, photoz_chi_best, photoz_odds, N_minijpas, x_im, y_im,\
        ra, dec = load_minijpas_jnep()
    N_sources = len(zsp)

    # Initialize output arrays
    times_selected = np.zeros(N_sources).astype(int) # Arr with the number counts
    times_nicez = np.copy(times_selected)

    # print('Adding errors...')
    # pm_flx_0, pm_err = add_errors(pm_flx, apply_err=True, survey_name='minijpasAEGIS001')

    ###############

    N_parallel = 5
    N_iter = 500

    print('\nSelection start\n')
    t0 = time.time()
    initial_count = threading.activeCount()
    for i in range(N_iter):
        while threading.activeCount() - initial_count >= N_parallel:
            time.sleep(1)
        
        print(f'{i + 1} / {N_iter}')

        this_pm_flx = pm_flx + pm_err * np.random.normal(size=pm_err.shape)
        args = (this_pm_flx, pm_err, zsp, times_selected, times_nicez)
        this_thread = threading.Thread(target=this_selection, args=args)
        this_thread.start()

    # Wait until the last thread is finished
    while threading.activeCount() - initial_count != 0:
        time.sleep(1) 

    # Save the arrays
    np.save('tmp/times_selected.npy', times_selected)
    np.save('tmp/times_nicez.npy', times_nicez)

    print('\nDone in {0}m {1:0.1f}s'.format(*divmod(time.time() - t0, 60)))