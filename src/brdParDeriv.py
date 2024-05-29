import jax
import jax.numpy as jnp
import numpy as np

'''
import my own Module
'''
from brdModel import calSatPos_16para
from brdModel import calSatPos_17para_case1,calSatPos_17para_case2,calSatPos_17para_case3
from brdModel import calSatPos_18para_case1,calSatPos_18para_case2,calSatPos_18para_case3
from brdModel import calSatPos_19para_case1,calSatPos_19para_case2,calSatPos_19para_case3
from brdModel import calSatPos_20para_case1,calSatPos_20para_case2,calSatPos_20para_case3
from brdModel import calSatPos_21para_case1,calSatPos_21para_case2,calSatPos_21para_case3
from brdModel import calSatPos_22para_case1,calSatPos_22para_case2,calSatPos_22para_case3


'''
This function: calculate the partial derivatives of satellite position vectors
with respect to broadcast ephemeris parameters at a single epoch
'''
def formMatrixSingleEpoch(x_ecef, x_brd, tk, toe, npara, para_case):
    # in: (1)x,y,z from sp3
    # (2) tk, toe
    # (3) npara(the number of broadcast eph parameters)
    # (4) the case for special parameter scheme

    # out: (1) A matrix at a single epoch
    # (2) B vector at a single epoch
    if npara == 16:
        x_cal_pos = calSatPos_16para(x_brd, tk, toe)
        f = jax.jacrev(calSatPos_16para)
        dfdx = f(x_brd, tk, toe)

    elif npara == 17:
        if para_case == 'case1':
            x_cal_pos = calSatPos_17para_case1(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_17para_case1)
            dfdx = f(x_brd, tk, toe)

        elif para_case == 'case2':
            x_cal_pos = calSatPos_17para_case2(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_17para_case2)
            dfdx = f(x_brd, tk, toe)

        elif para_case == 'case3':
            x_cal_pos = calSatPos_17para_case3(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_17para_case3)
            dfdx = f(x_brd, tk, toe)

    elif npara == 18:
        if para_case == 'case1':
            x_cal_pos = calSatPos_18para_case1(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_18para_case1)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case2':
            x_cal_pos = calSatPos_18para_case2(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_18para_case2)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case3':
            x_cal_pos = calSatPos_18para_case3(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_18para_case3)
            dfdx = f(x_brd, tk, toe)

    elif npara == 19:
        if para_case == 'case1':
            x_cal_pos = calSatPos_19para_case1(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_19para_case1)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case2':
            x_cal_pos = calSatPos_19para_case2(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_19para_case2)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case3':
            x_cal_pos = calSatPos_19para_case3(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_19para_case3)
            dfdx = f(x_brd, tk, toe)

    elif npara == 20:
        if para_case == 'case1':
            x_cal_pos = calSatPos_20para_case1(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_20para_case1)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case2':
            x_cal_pos = calSatPos_20para_case2(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_20para_case2)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case3':
            x_cal_pos = calSatPos_20para_case3(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_20para_case3)
            dfdx = f(x_brd, tk, toe)

    elif npara == 21:
        if para_case == 'case1':
            x_cal_pos = calSatPos_21para_case1(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_21para_case1)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case2':
            x_cal_pos = calSatPos_21para_case2(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_21para_case2)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case3':
            x_cal_pos = calSatPos_21para_case3(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_21para_case3)
            dfdx = f(x_brd, tk, toe)

    elif npara == 22:
        if para_case == 'case1':
            x_cal_pos = calSatPos_22para_case1(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_22para_case1)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case2':
            x_cal_pos = calSatPos_22para_case2(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_22para_case2)
            dfdx = f(x_brd, tk, toe)
        elif para_case == 'case3':
            x_cal_pos = calSatPos_22para_case3(x_brd, tk, toe)
            f = jax.jacrev(calSatPos_22para_case3)
            dfdx = f(x_brd, tk, toe)

    A_mat = dfdx
    B_mat = x_ecef[0:3] - x_cal_pos

    return np.array(A_mat, dtype=np.float64), np.array(B_mat, dtype=np.float64)
    


'''
This function: accumulate the partial derivatives of satellite position vectors
with respect to broadcast ephemeris parameters at many epochs
'''
def formMatrixManyEpochs(x_ecef_arr, x_brd, tk_arr, toe, npara, para_case):
    #pass
    
    # obtain the row and column size length
    rows = tk_arr.shape[0]

    final_A_mat = np.zeros((3*rows, npara-1))
    final_B_mat = np.zeros(3*rows)
    
    
    for index in range(rows):
        single_A_mat, single_B_mat = formMatrixSingleEpoch(x_ecef_arr[index,0:3], x_brd, tk_arr[index], toe, npara, para_case)
        final_A_mat[3*index:3*index+3,:] = single_A_mat
        final_B_mat[3*index:3*index+3] = single_B_mat
        
    #return final_A_mat, final_B_mat
    return final_A_mat, final_B_mat




    
    
    
