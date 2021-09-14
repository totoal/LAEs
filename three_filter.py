import numpy as np
from scipy.integrate import simps

'''
alpha & beta: two auxiliary functions for the three-filter continuum estimate method
'''
def alpha(w_Arr, t):
    return simps(w_Arr**2 * t, w_Arr) / simps(w_Arr*t, w_Arr)

def beta(w_Arr, t, w_EL):
    return np.interp(w_EL, w_Arr, t) * w_EL / simps(t*w_Arr, w_Arr)

# Main function #
def three_filter_method(NB, BB_LC, BB_LU,
                        NB_err, BB_LC_err, BB_LU_err,
                        t_NB, w_NB,
                        t_BB_LC, t_BB_LU, w_BB_LC, w_BB_LU,
                        w_EL):

    a_LU = alpha(w_BB_LU, t_BB_LU)
    a_LC = alpha(w_BB_LC, t_BB_LC)
    a_NB = alpha(w_NB, t_NB)
    b_LC = beta(w_BB_LC, t_BB_LC, w_EL)
    b_NB = beta(w_NB, t_NB, w_EL)
    c_a = (a_LU-a_LC)/(a_NB-a_LU)

    A_denominator = (a_NB - a_LU - b_NB/b_LC * (a_LC - a_LU))

    F_EL = ((BB_LC - BB_LU) + c_a * (NB - BB_LU)) / (b_LC + c_a * b_NB)
    A = (NB - BB_LU - b_NB/b_LC * (BB_LC - BB_LU)) / A_denominator
    B = BB_LU - a_LU * A

    A_err =\
    (NB_err**2 + BB_LU_err**2 + (b_NB/b_LC)**2*(BB_LC_err**2 + BB_LU_err**2))**0.5\
                        / A_denominator
    B_err = (BB_LU_err**2 + a_LU**2 * A_err**2)**0.5

    return F_EL, A, B, A_err, B_err
