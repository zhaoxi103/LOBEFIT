#!/usr/bin/python3

'''
 This py module is used for converting different time
'''

import numpy as np
from math import *
import sys
import os

Month_doy = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
Month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
Doys = [366, 365, 365, 365]
Jdst = 34012
Yearst = 1952

def doy_ymd(*timeinp):
	'''
	This function is used for converting doy to ymd or ymd to doy
	timeinp: arg of time(year doy/year mm dd)
	'''
	ninp = len(timeinp)
	if ninp == 3:
		year, mm, dd = timeinp
		doy = Month_doy[mm - 1] + dd
		doy = doy + 1 if (year - 1988) % 4 == 0 and mm > 2 else doy
		return year, doy
	elif ninp == 2:
		year, doy = timeinp
		month = Month[:]
		month[1] = month[1] + 1 if (year - 1988) % 4 == 0 else month[1]
		for imon in range(len(month)):
			if doy > month[imon]:
				doy = doy - month[imon]
			else:
				break
		return year, imon + 1, doy
	else:
		print('Wrong input time: ' + str(timeinp))
		sys.exit()


def doy_mjd(*timeinp):
	'''
	This function is used for converting doy to mjd or mjd to doy
	timeinp: arg of time(mjd/year doy)
	'''
	ninp = len(timeinp)
	iy = 1
	if ninp == 1:
		mjd = timeinp[0]
		year, doy = Yearst, mjd + 1 - Jdst
		while doy > Doys[iy - 1]:
			doy = doy - Doys[iy - 1]
			year = year + 1
			iy = year % 4 + 1
		return year, doy
	elif ninp == 2:
		year, doy = timeinp
		nyear = year - Yearst
		leap = floor((nyear + 3) / 4)
		dd = nyear * 365
		leapy = year % 4 + 1
		mjd = Jdst - 1 + dd + leap + doy
		mjd = mjd - 1 if dd < 29 and leapy == 1 and nyear > 0 else mjd
		return mjd
	else:
		print('Wrong input time: ' + str(timeinp))


def mjd_gpswk(*timeinp):
	'''
	This function is used for converting the Mjd to GPS week or GPS week to Mjd
	timeinp: arg of time(mjd/week, weekd)
	'''
	ninp = len(timeinp)
	if ninp == 1:
		mjd = timeinp[0]
		nwk = int((mjd - 44244) / 7)
		nwkd = mjd - nwk * 7 - 44244
		return nwk, nwkd
	elif ninp == 2:
		nwk, nwkd = timeinp
		mjd = nwk * 7 + 44244 + nwkd
		return mjd
	else:
		print('Wrong input time: ' + str(timeinp))
		sys.exit()


def sod_hms(*timeinp):
	'''
	This function is used for converting Sod to Hms or Hms to Sod
	timeinp: arg of time(sod/hh, mm, ss)
	'''
	ninp = len(timeinp)
	if ninp == 1:
		sod = timeinp[0]
		hh = floor(sod / 3600)
		mm = floor((sod - hh * 3600) / 60)
		ss = sod - hh * 3600 - mm * 60
		return hh, mm, ss
	elif ninp == 3:
		hh, mm, ss = timeinp
		sod = hh * 3600 + mm * 60 + ss
		return sod
	else:
		print('Wrong input time: ' + str(timeinp))
		sys.exit()


def sow_wkd(*timeinp):
	'''
	This function is used for converting Sow(second of week) to Wkd(week day) or Wkd to Sow
	timeinp: arg of time(sow/weekd, sod)
	'''
	ninp = len(timeinp)
	if ninp == 1:
		sow = timeinp[0]
		weekd = floor(sow / 86400)
		sod = sow - weekd * 86400
		return weekd, sod
	elif ninp == 2:
		weekd, sod = timeinp
		sow = weekd * 86400 + sod
		return sow
	else:
		print('Wrong input time: ' + str(timeinp))
		sys.exit()


def dmjd_ymdhms(*timeinp):
	'''
	This function is used for converting Dmjd(mjd.sod) tp Ymdhms(year, mon, doy, hh, mm, ss) or Ymdhms to Dmjd
	timeinp: arg of time(mjd.sod/year, mon, doy, hh, mm, ss)
	'''
	ninp = len(timeinp)
	if ninp == 1:
		dmjd = timeinp[0]
		mjd = floor(dmjd)
		sod = (dmjd - mjd) * 86400
		year, doy = doy_mjd(mjd)
		year, mon, day = doy_ymd(year, doy)
		hh, mm, ss = sod_hms(sod)
		ss = round(ss) if ss < 10e-5 else ss
		return year, mon, day, hh, mm, ss
	elif ninp == 6:
		year, mon, day, hh, mm, ss = timeinp
		year, doy = doy_ymd(year, mon, day)
		mjd = doy_mjd(year, doy)
		sod = sod_hms(hh, mm, ss)
		dmjd = mjd + sod / 86400.0
		return dmjd
	else:
		print('Wrong input time: ' + str(timeinp))
		sys.exit()


def ymdhms_dif(timeinp1, timeinp2):
	'''
	This function is used for calculating the time difference between two time
	timeinp1: [year, month, day, hour, minute, second]
	timeinp2: [year, month, day, hour, minute, second]
	return: dif sec
	'''
	year1, doy1 = doy_ymd(timeinp1[0], timeinp1[1], timeinp1[2])
	mjd1 = doy_mjd(year1, doy1)
	sod1 = sod_hms(timeinp1[3], timeinp1[4], timeinp1[5])
	year2, doy2 = doy_ymd(timeinp2[0], timeinp2[1], timeinp2[2])
	mjd2 = doy_mjd(year2, doy2)
	sod2 = sod_hms(timeinp2[3], timeinp2[4], timeinp2[5])
	sod_dif = round((mjd1 - mjd2) * 86400 + (sod1 - sod2))
	return sod_dif

def ymdhms_inc(timeinp, addsec):
	'''
	This function is used for calculating the time after increase
	timeinp: [year, month, day, hour, minute, second]
	addsec: seclen
	return:  [year, month, day, hour, minute, second]
	'''
	year, doy = doy_ymd(timeinp[0], timeinp[1], timeinp[2])
	mjd = doy_mjd(year, doy)
	sod = sod_hms(timeinp[3], timeinp[4], timeinp[5])
	sod = sod + addsec
	dayinc = sod // 86400
	sodinc = sod % 86400
	mjdinc = mjd + dayinc
	year1, doy1 = doy_mjd(mjdinc)
	year1, mon1, dd1 = doy_ymd(year1, doy1)
	hh1, mm1, ss1 = sod_hms(sodinc)
	return year1, mon1, dd1, hh1, mm1, ss1

def calGMST(D):
    '''
	This function is used for calculating GMST
	D: Julian day
	return: GMST
	'''
    T = D / 36525.0
    GMST = 6.697374558 + 2400.051336*T + 0.000025862*T*T
    GMST = np.mod(GMST, 24)
    GMST = GMST * 15
    return GMST

def calGAST(MJD):
    '''
	This function is used for calculating GMST
	D: modified Julian day
	return: GAST
	'''
    fjd = MJD + 2400000.5
    TJD = fjd - 2451545.0 # Julian day of 2000-1-1.5 
    T0 = TJD / 36525.0
    THETAm = calGMST(TJD)
    EPSILONm = 23.4392911111111111 - 0.0130041666667* T0 - 1.638e-07 * T0 * T0 + 5.0361e-07 * T0 * T0 * T0
    
    L = 280.4665 + 36000.7698 * T0
    dL = 218.3165 + 481267.8813 * T0
    OMEGA = 125.04452 - 1934.136261 * T0
    dPSI = -17.20 * np.sin(OMEGA) - 1.32 * np.sin(2 * L) - 0.23 * np.sin(2 * dL) + 0.21 * np.sin(2 * OMEGA)
    dEPSILON = 9.20 * np.cos(OMEGA) + 0.57 * np.cos(2 * L) + 0.10 * np.cos(2 * dL) - 0.09 * np.cos(2 * OMEGA)
     
    dPSI /= 3600
    dEPSILON /= 3600
    
    GAST = (THETAm + dPSI * np.cos(EPSILONm + dEPSILON))
    GAST = np.mod(GAST, 360)
    return GAST