import numpy as np
from scipy.integrate import simpson
from my_functions import IGM_TRANSMISSION

'''
alpha & beta: two auxiliary functions for the three-filter continuum estimate method
'''
def alpha(w_Arr, t):
    return simpson(w_Arr**2 * t, w_Arr) / simpson(w_Arr*t, w_Arr)

def beta(w_Arr, t, w_EL):
    return np.interp(w_EL, w_Arr, t) * w_EL / simpson(t*w_Arr, w_Arr)

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

def NB_3fm(pm_data, pm_err, nb_c, tcurves, w_central, bb_dist=10, N_nb=4):
    '''
    Builds synthetic photometry for virtual broad bands made out of narrow bands which
    serve as arguments for the three_filter_method function.

    INPUTS: pm_data (60xN_sources), pm_err (60xN_sources), nb_c central NB, bb_dist
    separation in Angstroms of the virtual BBs, N_nb number of NBs to make the virtual
    BB.
    '''
    # IGM_T_Arr = IGM_TRANSMISSION(np.array(w_central[nb_c-N_nb : nb_c])).reshape(-1, 1)
    # pm_data[nb_c-N_nb : nb_c] *= IGM_T_Arr
    # pm_err[nb_c-N_nb : nb_c] *= IGM_T_Arr

    NB = pm_data[nb_c]
    BB_LC = np.average(
        pm_data[nb_c-N_nb : nb_c+N_nb+1], axis=0,
        weights=pm_err[nb_c-N_nb : nb_c+N_nb+1] ** -2
    )
    BB_LU = np.average(
        pm_data[nb_c-N_nb+bb_dist : nb_c+N_nb+1+bb_dist], axis=0,
        weights=pm_err[nb_c-N_nb+bb_dist : nb_c+N_nb+1+bb_dist] ** -2
    )
    NB_err = pm_err[nb_c]
    BB_LC_err = np.sum(pm_err[nb_c-N_nb : nb_c+N_nb+1] ** -2, axis=0) ** -0.5
    BB_LU_err = np.sum(
        pm_err[nb_c-N_nb+bb_dist : nb_c+N_nb+1+bb_dist] ** -2, axis=0) ** -0.5

    w_Arr = np.array(tcurves['w'][nb_c])
    t_NB = np.array(tcurves['t'][nb_c])
    t_BB_LC = np.zeros(len(w_Arr))
    t_BB_LU = np.zeros(len(w_Arr))

    for i in range(2 * N_nb + 1):
        t_BB_LC += np.interp(
            w_Arr, tcurves['w'][nb_c-N_nb+i], tcurves['t'][nb_c-N_nb+i]
        )
        t_BB_LU += np.interp(
            w_Arr, tcurves['w'][nb_c-N_nb+bb_dist+i], tcurves['t'][nb_c-N_nb+bb_dist+i]
        )

    w_NB = w_BB_LU = w_BB_LC = w_Arr

    w_EL = w_central[nb_c]

    return three_filter_method(NB, BB_LC, BB_LU, NB_err, BB_LC_err, BB_LU_err, t_NB,
                               w_NB, t_BB_LC, t_BB_LU, w_BB_LC, w_BB_LU, w_EL)
