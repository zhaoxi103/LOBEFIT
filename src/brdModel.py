import jax
import jax.numpy as jnp
import numpy as np

from brdConstant import MU_GPS, OMGE

'''
16-parameter model
'''
def calSatPos_16para(x, tk, toe):
    # 16 parameters, toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        #diff = (Mk - (Ek - e * jnp.sin(Ek))) / (1.0 - e * jnp.cos(Ek))
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

'''
17-parameter model,case1~3
'''
def calSatPos_17para_case1(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Adot = x[15]

    A = sqra * sqra + Adot * tk
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        #diff = (Mk - (Ek - e * jnp.sin(Ek))) / (1.0 - e * jnp.cos(Ek))
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])
    
def calSatPos_17para_case2(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    ndot = x[15]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        #diff = (Mk - (Ek - e * jnp.sin(Ek))) / (1.0 - e * jnp.cos(Ek))
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

def calSatPos_17para_case3(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    rdot = x[15]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        #diff = (Mk - (Ek - e * jnp.sin(Ek))) / (1.0 - e * jnp.cos(Ek))
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk + rdot*tk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

'''
18-parameter model, case1~3
'''    
def calSatPos_18para_case1(x, tk, toe):
    # 18 parameters, toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Crc3 = x[15]
    Crs3 = x[16]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc3 * jnp.cos(3.0 * Phik) + Crs3 * jnp.sin(3.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])
    
def calSatPos_18para_case2(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Cuc3 = x[15]
    Cus3 = x[16]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc3 * jnp.cos(3.0 * Phik) + Cus3 * jnp.sin(3.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

def calSatPos_18para_case3(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    ndot = x[15]
    ndotdot = x[16]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk + 0.5*ndotdot*tk*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

'''
19-parameter model,case1~3
'''
def calSatPos_19para_case1(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Adot = x[15]
    Adotdot = x[16]
    ndot = x[17]

    A = sqra * sqra + Adot*tk + 0.5*Adotdot*tk*tk
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])
    
def calSatPos_19para_case2(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Adot = x[15]
    ndot = x[16]
    ndotdot = x[17]

    A = sqra * sqra + Adot*tk
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk + 0.5*ndotdot*tk*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])
    
def calSatPos_19para_case3(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    rdot = x[15]
    Crc1 = x[16]
    Crs1 = x[17]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc1 * jnp.cos(Phik) + Crs1 * jnp.sin(Phik) + rdot*tk
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

'''
20-parameter model, case1~3
'''
def calSatPos_20para_case1(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Crc3 = x[15]
    Crs3 = x[16]
    Cuc1 = x[17]
    Cus1 = x[18]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc3 * jnp.cos(3.0 * Phik) + Crs3 * jnp.sin(3.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc1 * jnp.cos(Phik) + Cus1 * jnp.sin(Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

def calSatPos_20para_case2(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Crc3 = x[15]
    Crs3 = x[16]
    Adot = x[17]
    Adotdot = x[18]

    A = sqra * sqra + Adot*tk + 0.5*Adotdot*tk*tk
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc3 * jnp.cos(3.0 * Phik) + Crs3 * jnp.sin(3.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

def calSatPos_20para_case3(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    Cuc3 = x[15]
    Cus3 = x[16]
    rdot = x[17]
    udot = x[18]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + rdot*tk
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc3 * jnp.cos(3.0 * Phik) + Cus3 * jnp.sin(3.0 * Phik) + udot*tk
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

'''
21-parameter model, case1~3
'''
def calSatPos_21para_case1(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    rdot = x[15]
    Crc1 = x[16]
    Crs1 = x[17]
    Cuc3 = x[18]
    Cus3 = x[19]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc1 * jnp.cos(Phik) + Crs1 * jnp.sin(Phik) + rdot*tk
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc3 * jnp.cos(3.0 * Phik) + Cus3 * jnp.sin(3.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

def calSatPos_21para_case2(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    ndot = x[15]
    Crc1 = x[16]
    Crs1 = x[17]
    Cuc3 = x[18]
    Cus3 = x[19]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc1 * jnp.cos(Phik) + Crs1 * jnp.sin(Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc3 * jnp.cos(3.0 * Phik) + Cus3 * jnp.sin(3.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

def calSatPos_21para_case3(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    ndot = x[15]
    Crc3 = x[16]
    Crs3 = x[17]
    Cuc1 = x[18]
    Cus1 = x[19]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc3 * jnp.cos(3.0 * Phik) + Crs3 * jnp.sin(3.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc1 * jnp.cos(Phik) + Cus1 * jnp.sin(Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

'''
22-parameter model, case1~3
'''
def calSatPos_22para_case1(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    rdot = x[15]
    udot = x[16]
    Crc1 = x[17]
    Crs1 = x[18]
    Cuc3 = x[19]
    Cus3 = x[20]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc1 * jnp.cos(Phik) + Crs1 * jnp.sin(Phik) + rdot*tk
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc3 * jnp.cos(3.0 * Phik) + Cus3 * jnp.sin(3.0 * Phik) +udot*tk
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])


def calSatPos_22para_case2(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    ndot = x[15]
    ndotdot = x[16]
    Crc1 = x[17]
    Crs1 = x[18]
    Cuc3 = x[19]
    Cus3 = x[20]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk + 0.5*ndotdot*tk*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc1 * jnp.cos(Phik) + Crs1 * jnp.sin(Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc3 * jnp.cos(3.0 * Phik) + Cus3 * jnp.sin(3.0 * Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])

def calSatPos_22para_case3(x, tk, toe):
    # toe is fixed here
    sqra = x[0]
    #e = x[1]
    ex = x[1]
    ey = x[2]
    # i0 = x[2]
    # omega0 = x[3]
    # w = x[4]
    # M0 = x[5]
    i0 = x[3]
    omega0 = x[4]
    u0 = x[5]
    dn = x[6]
    idot = x[7]
    omegadot = x[8]
    Crc = x[9]
    Crs = x[10]
    Cuc = x[11]
    Cus = x[12]
    Cic = x[13]
    Cis = x[14]
    ndot = x[15]
    ndotdot = x[16]
    Crc3 = x[17]
    Crs3 = x[18]
    Cuc1 = x[19]
    Cus1 = x[20]

    A = sqra * sqra
    n0 = jnp.sqrt(MU_GPS / A / A / A)

    w = jnp.arctan2(ey, ex)
    e = jnp.sqrt(ex*ex + ey*ey)
    M0 = u0 - w

    # d_toe = tk
    Mk = M0 + tk * (n0 + dn + ndot*tk + 0.5*ndotdot*tk*tk)
    Ek = Mk
    nofe = 0
    while True:
        nofe += 1
        diff = Mk - (Ek - e * jnp.sin(Ek))
        Ek = Ek + diff
        if abs(diff) <= 1.0e-9 or nofe >= 15:
            break

    vk = jnp.arctan2(jnp.sqrt(1.00 - jnp.square(e)) * jnp.sin(Ek) , (jnp.cos(Ek) - e))
    Phik = vk + w

    delta_rk = Crc * jnp.cos(2.0 * Phik) + Crs * jnp.sin(2.0 * Phik) + Crc3 * jnp.cos(3.0 * Phik) + Crs3 * jnp.sin(3.0 * Phik)
    delta_uk = Cuc * jnp.cos(2.0 * Phik) + Cus * jnp.sin(2.0 * Phik) + Cuc1 * jnp.cos(Phik) + Cus1 * jnp.sin(Phik)
    delta_ik = Cic * jnp.cos(2.0 * Phik) + Cis * jnp.sin(2.0 * Phik)

    uk = Phik + delta_uk
    rk = A * (1.00 - e * jnp.cos(Ek)) + delta_rk
    ik = i0 + delta_ik + idot * tk

    xk = rk * jnp.cos(uk)
    yk = rk * jnp.sin(uk)
    omegak = omega0 + (omegadot - OMGE) * tk - OMGE * toe

    pos_x = xk * jnp.cos(omegak) - yk * jnp.cos(ik) * jnp.sin(omegak)
    pos_y = xk * jnp.sin(omegak) + yk * jnp.cos(ik) * jnp.cos(omegak)
    pos_z = yk * jnp.sin(ik)

    return jnp.asarray([pos_x, pos_y, pos_z])
  
