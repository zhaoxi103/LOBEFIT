import jax
import jax.numpy as jnp
from jax import config
import numpy as np

import os
import sys
import argparse

'''
import my own Module
'''
from time_convert import doy_ymd, doy_mjd, mjd_gpswk ,sod_hms, sow_wkd, calGAST
from brdConstant import MU_GPS, OMGE, nMaxIter
from brdParDeriv import formMatrixManyEpochs



class opttype:
    def __init__(self):
        self.MJD = 99999
        self.sow = 0
        self.sod = 0
        self.npara = 16
        self.inputfile = "XXXX.sp3"
        self.indicator = "K01"
        self.inter_proc = 1
        self.fitNum = 0
        #self.predNum = 0
        self.URER = 0.5823
        self.URETN = 0.5749
        self.para_case = 'case1'
        self.inputtype = 'PV'
        self.outputPath = '/home/users'
            
        
        

'''
Extract the epochs and corresponding satellite orbits used for fitting
'''
def extract_orbit_and_time(time_arr, orbit_arr, input_time, m):
    # Find the index of the input time 
    index = np.where((time_arr[:, 0] == input_time[0]) & (time_arr[:, 1] == input_time[1]))[0][0]

    
    if (index < 0) or (index >= len(time_arr)):
        return None, None
    elif index + m > len(time_arr):  
        return time_arr[index:], orbit_arr[index:]
    else:
        return time_arr[index:index + m], orbit_arr[index:index + m]

 
'''
obtain initial broadcast ephemeris parameters before iteration for fitting
'''
def initial_fiteph(r0, rp0, npara):
    '''
    in---
    r0:satellite position vector
    rp0:satellite velocity vector
    npara: the number of parameters
    out---
    x_brd:the initial value of broadcast ephemeris parameter
    '''
    
    x_brd = np.zeros(npara-1)  
    
    norm = np.linalg.norm
    cross = np.cross
    nr = norm(r0)
    nv2 = norm(rp0)**2
    rrp = r0.dot(rp0)
    a_1 = 2 / nr - nv2 / MU_GPS
    a = 1 / a_1
    ecos = 1 - nr / a
    esin = rrp / np.sqrt(MU_GPS*a)
    e = np.sqrt(ecos**2 + esin**2)
    E = np.arctan2(esin, ecos)
    M = E - e * np.sin(E)
    pz = np.cos(E)/nr*r0[2] - np.sqrt(a / MU_GPS)*np.sin(E)*rp0[2]
    qz = np.sin(E)/nr*r0[2] + np.sqrt(a/MU_GPS)*(np.cos(E) - e)*rp0[2]
    qz /= np.sqrt(1 - e*e)
    omega = np.arctan2(pz, qz)
    rcrp = cross(r0, rp0)
    bigomega = np.arctan2(rcrp[0], -rcrp[1])
    i = np.arccos(rcrp[2] / np.sqrt(MU_GPS*a*(1 - e*e)))
    
    x_brd[0] = np.sqrt(a)
    # x_brd[1] = e
    x_brd[1] = e*np.cos(omega)
    x_brd[2] = e*np.sin(omega)
    # x_brd[2] = i
    # x_brd[3] = bigomega
    # x_brd[4] = omega
    # x_brd[5] = M
    x_brd[3] = i
    x_brd[4] = bigomega
    x_brd[5] = omega + M
    
    return x_brd
        
'''
obtain complete broadcast ephemeris fitting process
'''    
def ephfit(time_arr, orbit_arr, optClass):
    '''
    in---
    time_arr:epochs for fitting
    orbit_arr:satellite position vectors for fitting
    optClass: Class for option from reading configuration file
    out---
    toe: toe for output
    x_brd0:the estimated value of broadcast ephemeris parameter
    RTN_lst: RTN fitting errors and UREs
    '''    
    
    #(1) extract several sp3 orbit
    fitNum = optClass.fitNum
    option_start_time = np.array([optClass.MJD, optClass.sow])
    chosen_time_arr, chosen_orbit_arr = extract_orbit_and_time(time_arr, orbit_arr, option_start_time, fitNum)
    
    #(2) toe and tk_arr
    toe = chosen_time_arr[-1, 1]
    tk_arr = chosen_time_arr[:, 1] - toe
    
    #(3) obtain the initial broadcast ephmeris value
    npara = optClass.npara
    x_ecef_toe = chosen_orbit_arr[-1,:]
    x_brd0 = initial_fiteph(x_ecef_toe[0:3], x_ecef_toe[3:6], npara)
    
    
        
    #(4) begin to fit with iteration
    x_brd0 = jnp.array(x_brd0, dtype=jnp.float64)
    chosen_orbit_arr = jnp.array(chosen_orbit_arr,dtype=jnp.float64)
    deltaV = np.ones(npara-1)
    para_case = optClass.para_case
    for i in range(nMaxIter):
        if np.sum(np.square(deltaV)) >= 1e-7:
            # 
            A_mat, B_mat = formMatrixManyEpochs(chosen_orbit_arr, x_brd0, tk_arr, toe, npara, para_case)
            #A_mat, B_mat = parallel_formMatrixManyEpochs(chosen_orbit_arr, x_brd0, tk_arr, toe, npara)
            #
            A_t = A_mat.T  # The transpose of A
            deltaV = np.linalg.inv(A_t @ A_mat) @ A_t @ B_mat
            x_brd0 = x_brd0 + deltaV
        else:
            break
    
    # recovery for real para
    x_brd0 = np.array(x_brd0, dtype=np.float64)
    final_w = np.arctan2(x_brd0[2], x_brd0[1])
    final_e = np.sqrt(x_brd0[1]*x_brd0[1] + x_brd0[2]*x_brd0[2])
    final_M0 = x_brd0[5] - final_w

    x_brd0[1] = final_e
    x_brd0[2] = x_brd0[3]
    x_brd0[3] = x_brd0[4]
    x_brd0[4] = final_w
    x_brd0[5] = final_M0

    #(5) precision
    first_mjd = chosen_time_arr[0, 0]
    # first_sec = chosen_time_arr[0, 1]
    S0 = calGAST(first_mjd)
    S = S0 + OMGE * toe
    Rs = np.array([[np.cos(S), np.sin(S), 0], [-np.sin(S), np.cos(S), 0], [0, 0, 1]])
    
    
    RTN_lst = []
    for k in range(fitNum):
        gp = np.array(chosen_orbit_arr[k, :3],dtype=np.float64)
        gv = np.array(chosen_orbit_arr[k, 3:6],dtype=np.float64)
        gr = np.linalg.norm(gp)
        GA = gp / gr
        grrp3 = np.cross(gp, gv)
        mgrrp = np.linalg.norm(grrp3)
        GC = grrp3 / mgrrp
        GB = np.cross(GC, GA)
     
        G = np.array([[GA[0], GB[0], GC[0]],
            [GA[1], GB[1], GC[1]],
            [GA[2], GB[2], GC[2]]])
        GR = G @ Rs
        dxyz = B_mat[k*3 : k*3+3]
        RTN = GR @ dxyz
        URE = np.sqrt((optClass.URER*RTN[0])**2 + (optClass.URETN*RTN[1])**2 + (optClass.URETN*RTN[2])**2)
        UREN = np.array([chosen_time_arr[k,0], chosen_time_arr[k,1], RTN[1], RTN[0], RTN[2], URE])
        RTN_lst.append(UREN)
        
    return toe, x_brd0, RTN_lst

'''
output the Result 
'''  
def outputResult(toe_sow, brd_para, rtnLst, opt):
    #(1) the estimated value
    npara = opt.npara
    para_case = opt.para_case
    outputPath = opt.outputPath
    eph_para_file = os.path.join(outputPath,"eph_para")
    with open(eph_para_file, "w") as fp:
        fp.write("%-100s%20d\n" % ("The numbers of parameters used for this resulution is: ", npara))
        fp.write("%-100s%20.12e\n" % ("toe: reference time ephemeris", toe_sow))
        fp.write("%-100s%20.12e\n" % ("sqra: square root of semi-major axis", brd_para[0]))
        fp.write("%-100s%20.12e\n" % ("dty: eccentricity", brd_para[1]))
        fp.write("%-100s%20.12e\n" % (
            "inc0: inclination angle at reference time", brd_para[2]))
        fp.write("%-100s%20.12e\n" % (
            "Omega0: longitude of ascending node of orbit plane at weekly epoch", brd_para[3]))
        fp.write("%-100s%20.12e\n" % ("w: argument of perigee", brd_para[4]))
        fp.write("%-100s%20.12e\n" % ("ma: mean anomaly at reference time", brd_para[5]))
        fp.write("%-100s%20.12e\n" % ("dn: mean motion correction term", brd_para[6]))
        fp.write("%-100s%20.12e\n" % (
            "idot: rate of inclination angle", brd_para[7]))
        fp.write("%-100s%20.12e\n" % (
            "Omegadot: rate of right ascension", brd_para[8]))
        fp.write("%-100s%20.12e\n" % (
            "crc: amplitude of the cosine harmonic correction term to the orbit radius", brd_para[9]))
        fp.write("%-100s%20.12e\n" % (
            "crs: amplitude of the sine harmonic correction term to the orbit radius", brd_para[10]))
        fp.write("%-100s%20.12e\n" % (
            "cuc: amplitude of the cosine harmonic correction term to the argument of latitude", brd_para[11]))
        fp.write("%-100s%20.12e\n" % (
            "cus: amplitude of the sine harmonic correction term to the argument of latitude", brd_para[12]))
        fp.write("%-100s%20.12e\n" % (
            "cic: amplitude of the cosine harmonic correction term to the angle of inclination", brd_para[13]))
        fp.write("%-100s%20.12e\n" % (
            "cis: amplitude of the sine harmonic correction term to the angle of inclination", brd_para[14]))
        
        if npara == 17:
            if para_case == 'case1':
                fp.write("%-100s%20.12e\n" % ("Adot: newly introduced parameter", brd_para[15]))
            elif para_case == 'case2':
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[15]))
            elif para_case == 'case3':
                fp.write("%-100s%20.12e\n" % ("rdot: newly introduced parameter", brd_para[15]))
                
        elif npara == 18:
            if para_case == 'case1':
                fp.write("%-100s%20.12e\n" % ("crc3: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("crs3: newly introduced parameter", brd_para[16]))
            elif para_case == 'case2':
                fp.write("%-100s%20.12e\n" % ("cuc3: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("cus3: newly introduced parameter", brd_para[16]))
            elif para_case == 'case3':
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("ndotdot: newly introduced parameter", brd_para[16]))
        
        elif npara == 19:
            if para_case == 'case1':
                fp.write("%-100s%20.12e\n" % ("Adot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("Adotdot: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[17]))
            elif para_case == 'case2':
                fp.write("%-100s%20.12e\n" % ("Adot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("ndotdot: newly introduced parameter", brd_para[17]))
            elif para_case == 'case3':
                fp.write("%-100s%20.12e\n" % ("rdot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("Crc1: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("Crs1: newly introduced parameter", brd_para[16]))
        
        elif npara == 20:
            if para_case == 'case1':
                fp.write("%-100s%20.12e\n" % ("crc3: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("crs3: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("cuc1: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("cus1: newly introduced parameter", brd_para[18]))
            elif para_case == 'case2':
                fp.write("%-100s%20.12e\n" % ("crc3: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("crs3: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("Adot: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("Adotdot: newly introduced parameter", brd_para[18]))
            elif para_case == 'case3':
                fp.write("%-100s%20.12e\n" % ("cuc3: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("cus3: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("rdot: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("udot: newly introduced parameter", brd_para[18]))
                
        elif npara == 21:
            if para_case == 'case1':
                fp.write("%-100s%20.12e\n" % ("rdot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("crc1: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("crs1: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("cuc3: newly introduced parameter", brd_para[18]))
                fp.write("%-100s%20.12e\n" % ("cus3: newly introduced parameter", brd_para[19]))
            elif para_case == 'case2':
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("crc1: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("crs1: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("cuc3: newly introduced parameter", brd_para[18]))
                fp.write("%-100s%20.12e\n" % ("cus3: newly introduced parameter", brd_para[19]))
            elif para_case == 'case3':
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("crc3: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("crs3: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("cuc1: newly introduced parameter", brd_para[18]))
                fp.write("%-100s%20.12e\n" % ("cus1: newly introduced parameter", brd_para[19]))
                
        elif npara == 22:
            if para_case == 'case1':
                fp.write("%-100s%20.12e\n" % ("rdot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("udot: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("crc1: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("crs1: newly introduced parameter", brd_para[18]))
                fp.write("%-100s%20.12e\n" % ("cuc3: newly introduced parameter", brd_para[19]))
                fp.write("%-100s%20.12e\n" % ("cus3: newly introduced parameter", brd_para[20]))
            elif para_case == 'case2':
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("ndotdot: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("crc1: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("crs1: newly introduced parameter", brd_para[18]))
                fp.write("%-100s%20.12e\n" % ("cuc3: newly introduced parameter", brd_para[19]))
                fp.write("%-100s%20.12e\n" % ("cus3: newly introduced parameter", brd_para[20]))
            elif para_case == 'case3':
                fp.write("%-100s%20.12e\n" % ("ndot: newly introduced parameter", brd_para[15]))
                fp.write("%-100s%20.12e\n" % ("ndotdot: newly introduced parameter", brd_para[16]))
                fp.write("%-100s%20.12e\n" % ("crc3: newly introduced parameter", brd_para[17]))
                fp.write("%-100s%20.12e\n" % ("crs3: newly introduced parameter", brd_para[18]))
                fp.write("%-100s%20.12e\n" % ("cuc1: newly introduced parameter", brd_para[19]))
                fp.write("%-100s%20.12e\n" % ("cus1: newly introduced parameter", brd_para[20]))
                
    #(2) fitting errors and UREs
    fitting_error_file = os.path.join(outputPath,"URE_file")
    with open(fitting_error_file, "w") as fp:
        for count in rtnLst:
            fp.write(
                f"{count[0]:15.6f} {count[1]:15.6f} {count[2]:15.6f} {count[3]:15.6f} {count[4]:15.6f} {count[5]:15.6f}\n")
    #pass


'''
get the input parameter from a single line 
'''                
def GetIniKeyString(title, key, filename):
    flag = 0
    sTitle = "[%s]" % title
    with open(filename, "r") as fp:
        for sLine in fp:
            if sLine.startswith("//"):
                continue
            if sLine.startswith("#"):
                continue
            wTmp = sLine.find('=')
            if wTmp != -1 and flag == 1:
                if sLine.startswith(key):
                    sLine = sLine.strip()
                    if sLine[-1] == '\n':
                        sLine = sLine[:-1]
                    buf = sLine[wTmp + 1:].lstrip()
                    return buf
            else:
                if sLine.startswith(sTitle):
                    flag = 1    
                    
'''
retrieves input parameters from the configuration file
'''                      
def readConfig(cfgPath, opt):

    _startTime = GetIniKeyString("config", "Start_Time", cfgPath)
    #_endTime = GetIniKeyString("config", "End_Time", cfgPath)
    _indicator = GetIniKeyString("config", "Indicator", cfgPath)
    _inputfile = GetIniKeyString("config", "Input_File", cfgPath)
    _inputtype = GetIniKeyString("config", "Input_Type", cfgPath)
    _interval = GetIniKeyString("config", "Interval", cfgPath)
    _paranum = GetIniKeyString("config", "Para_Num", cfgPath)
    _paracase = GetIniKeyString("config", "Para_Case", cfgPath)
    _epochnum = GetIniKeyString("config", "Epoch_FitNum", cfgPath)
    #_prednum = GetIniKeyString("config", "Epoch_PredNum", cfgPath)
    _URER = GetIniKeyString("config", "URER", cfgPath)
    _URETN = GetIniKeyString("config", "URETN", cfgPath)
    _outputPath = GetIniKeyString("config", "OutPath", cfgPath)
    
    if not os.path.exists(_outputPath):
        try:
            os.makedirs(_outputPath)
        except Exception as e:
            print(f"Failed to create path '{_outputPath}': {e}")
    
    str_tmp = _startTime.split()
    year = int(str_tmp[0])
    month = int(str_tmp[1])
    day = int(str_tmp[2])
    hour = int(str_tmp[3])
    minute = int(str_tmp[4])
    second = float(str_tmp[5])
    
    year, doy = doy_ymd(year,month,day)
    mjd = doy_mjd(year, doy)
    gpsweek, weekd = mjd_gpswk(mjd)
    sod = sod_hms(hour, minute, second)
    sow = sow_wkd(weekd,sod)

    opt.MJD = mjd
    opt.sow = sow
    opt.sod = sod
    opt.inputfile = _inputfile
    opt.indicator = _indicator
    opt.inter_proc = int(_interval)
    opt.npara = int(_paranum)
    opt.fitNum = int(_epochnum)
    opt.URER = float(_URER)
    opt.URETN = float(_URETN)
    opt.para_case = _paracase
    opt.inputtype = _inputtype
    opt.outputPath = _outputPath
    
    #return opt

'''
read satellite orbit data to the array 
'''      
def readsp3File(sp3File,LeoName,sp3Type,sampleTime):
    with open(sp3File,'r') as inp:
        allLines = inp.readlines()
        
    x_sp3_pool = []
    time_pool = []
    
    if sp3Type == 'PV' or sp3Type == 'P':
    
        for singleLine in allLines:
            if singleLine.startswith('*'):
                str_tmp = singleLine.split()
                year = int(str_tmp[1])
                month = int(str_tmp[2])
                day = int(str_tmp[3])
                hour = int(str_tmp[4])
                minute = int(str_tmp[5])
                second = float(str_tmp[6])

                year, doy = doy_ymd(year,month,day)
                mjd = doy_mjd(year, doy)
                gpsweek, weekd = mjd_gpswk(mjd)
                sod = sod_hms(hour,minute,second)
                sow = sow_wkd(weekd,sod)
                time_pool.append([mjd,sow])
            elif singleLine.startswith('P'+LeoName):
                x_sp3_single = []
                str_tmp = singleLine.split()
                x_sp3_single.append(float(str_tmp[1]))
                x_sp3_single.append(float(str_tmp[2]))
                x_sp3_single.append(float(str_tmp[3]))
            elif singleLine.startswith('V'+LeoName):
                str_tmp = singleLine.split()
                x_sp3_single.append(float(str_tmp[1]))
                x_sp3_single.append(float(str_tmp[2]))
                x_sp3_single.append(float(str_tmp[3]))

                x_sp3_pool.append([x * 1000 for x in x_sp3_single])  #transfer to m,m/s
            else:
                continue
    
    if sp3Type == 'P':
        #begin to calculate velocity using high-order differentiation
        print(len(x_sp3_pool))
        for k in range(len(x_sp3_pool)):
            if k == 0 or k == 1:
                continue
            elif k==len(x_sp3_pool)-1 or k==len(x_sp3_pool)-2: 
                x_sp3_pool[k,3:6] = x_sp3_pool[k-1,3:6]
            else:
                x_sp3_pool[k,3:6] = (x_sp3_pool[k-2,0:3] - 8*x_sp3_pool[k-1,0:3] + 8*x_sp3_pool[k+1,0:3] - x_sp3_pool[k+2,0:3])/(12*sampleTime)
        
        x_sp3_pool[0,3:6] = x_sp3_pool[2,3:6]
        x_sp3_pool[1,3:6] = x_sp3_pool[2,3:6]
        

           
    return np.array(time_pool, dtype=np.float64), np.array(x_sp3_pool, dtype=np.float64)
            
            
            
'''   
main function
(1) read config file
(2) read sp3 file to memory
(3) fit
(4) output the result
'''

if __name__ == '__main__':

# <very start. parse the command parameters>
    parser = argparse.ArgumentParser(description='parse the command parameters')
    parser.add_argument('-ini', type=str, help='the complete path and file name of the input configuration file')

    args = parser.parse_args()

    if args.ini is not None:
        cfgPath = args.ini
    else:
        print("Please provide the configuration file!\n")
        exit(1)

# </very start. parse the command parameters>

    config.update('jax_enable_x64', True)
    opt = opttype()
    
    #(1) read config and sp3 file 
    readConfig(cfgPath, opt)
    
    myTimePool, mySp3Pool = readsp3File(opt.inputfile,opt.indicator,opt.inputtype,opt.inter_proc)
    
    #(2) eph fit
    toeSow, x_final_brd, RTN_lst = ephfit(myTimePool, mySp3Pool, opt)
    
    #(3) output the result
    outputResult(toeSow, x_final_brd, RTN_lst, opt)
    
    
    
