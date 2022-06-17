import numpy as np
from scipy.integrate import quad

from my_functions import central_wavelength, load_tcurves, load_filter_tags

'''
alpha & beta: two auxiliary functions for the three-filter continuum estimate method
'''
def alpha(w_Arr, t):
    return quad(w_Arr**2 * t, w_Arr) / quad(w_Arr*t, w_Arr)

def beta(w_Arr, t, w_EL):
    return np.interp(w_EL, w_Arr, t) * w_EL / quad(t*w_Arr, w_Arr)

## Main function
def three_filter_method(NB, BB_LC, BB_LU,
                        NB_err, BB_LC_err, BB_LU_err,
                        t_NB, w_NB,
                        t_BB_LC, t_BB_LU, w_BB_LC, w_BB_LU,
                        w_EL):
    '''
    Classical 3-filter method that fits the spectral continuum to a straight line
    INPUT arguments: flambda of NB, Line Contaminated BB, Line Uncontaminated BB and
    their associated errors; transmission curves and wavelength arrays of these 3
    filters.
    RETURNS: Integrated line flux, A and B fit parameters with their errors.

    Fit: flambda_cont = A * lambda + B

    Details in Appendix A of Spinoso et al. 2020
    '''

    a_LU = alpha(w_BB_LU, t_BB_LU)
    a_LC = alpha(w_BB_LC, t_BB_LC)
    a_NB = alpha(w_NB, t_NB)
    b_LC = beta(w_BB_LC, t_BB_LC, w_EL)
    b_NB = beta(w_NB, t_NB, w_EL)
    c_a = (a_LU - a_LC) / (a_NB - a_LU)

    A_denominator = (a_NB - a_LU - b_NB/b_LC * (a_LC - a_LU))

    F_EL = ((BB_LC - BB_LU) + c_a * (NB - BB_LU)) / (b_LC + c_a * b_NB)
    A = (NB - BB_LU - b_NB/b_LC * (BB_LC - BB_LU)) / A_denominator
    B = BB_LU - a_LU * A

    A_err =\
    (NB_err**2 + BB_LU_err**2 + (b_NB/b_LC)**2*(BB_LC_err**2 + BB_LU_err**2))**0.5\
                        / A_denominator
    B_err = (BB_LU_err**2 + a_LU**2 * A_err**2)**0.5

    return F_EL, A, B, A_err, B_err

def cont_est_3FM(pm_flx, pm_err, NB_Arr):
    '''
    INPUT
    pm_flx: matrix of band fluxes (60 x N_sources)
    pm_err: errors of pm_flx
    NB_Arr: Array of NBs at the position of which to estimate the continuum

    RETURNS
    cont_est: A matrix of dimension 56 x N_sources with the continuum estimate for all the
        NB positions of NB_Arr. The continum at the non-requested NBs is 0.
    cont_err: Errors of cont_est. The error at the non-requested NBs is 99.
    '''
    tcurves = load_tcurves(load_filter_tags())
    w_central = central_wavelength()
    N_sources = pm_flx.shape[1]

    cont_est_lya = np.zeros((56, N_sources))
    cont_err_lya = np.ones((56, N_sources)) ** 99.

    for nb_c in NB_Arr:
        NB = pm_flx[nb_c]
        NB_err = pm_err[nb_c]
        t_NB = np.array(tcurves['t'][nb_c])
        w_NB = np.array(tcurves['w'][nb_c])
        w_EL = w_central[nb_c]
        if 5 <= nb_c < 18: # g band range
            BB_LC = pm_flx[-3]
            BB_LC_err = pm_err[-3]
            t_BB_LC = np.array(tcurves['t'][-3])
            w_BB_LC = np.array(tcurves['w'][-3])
            BB_LU = pm_flx[-2]
            BB_LU_err = pm_err[-2]
            t_BB_LU = np.array(tcurves['t'][-2])
            w_BB_LU = np.array(tcurves['w'][-2])
        elif 19 <= nb_c < 33: # r band range
            BB_LC = pm_flx[-2]
            BB_LC_err = pm_err[-2]
            t_BB_LC = np.array(tcurves['t'][-2])
            w_BB_LC = np.array(tcurves['w'][-2])
            BB_LU = pm_flx[-1]
            BB_LU_err = pm_err[-1]
            t_BB_LU = np.array(tcurves['t'][-1])
            w_BB_LU = np.array(tcurves['w'][-1])
        else:
            cont_est_lya[nb_c] = 0.
            cont_err_lya[nb_c] = 99.

        _, A, B, A_err, B_err = three_filter_method(
            NB, BB_LC, BB_LU, NB_err, BB_LC_err, BB_LU_err, t_NB, w_NB, t_BB_LC, t_BB_LU,
            w_BB_LC, w_BB_LU, w_EL
        )

        cont_est_lya[nb_c] = A * w_EL + B
        cont_err_lya[nb_c] = (w_EL**2 * A_err**2 + B_err**2) ** 0.5

    return cont_est_lya, cont_err_lya