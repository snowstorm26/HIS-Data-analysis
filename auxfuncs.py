import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob
import cdflib

# some necessary variables
micros_to_s = 1e-6 #conversion factor from microseconds to seconds
ns_to_s = 1e-9 #conversion factor from nanoseconds to seconds
m_to_AU = 1/(1.496e+11) #conversion factor from meters to astronomical units
mev_to_J = 1e6*1.6022e-19 #conversion factor from megaelectronvolts to joules. 
ms_to_AUd = 86400/(1.496e+11) #conversion factor from m/s to AU/day
ms_to_Gmd = 86400*1e-9 #conversion factor from m/s to Gm/day
u_to_kg = 1.6603145e-27  #conversion factor from atomic units to kg
proton_mass = 1.67262192e-27 # mass of a proton [kg]
day_sec = 86400 # number of seconds during one day
epoch_origin_SOLO = dt.datetime(2000,1,1,11,58,56)
epoch_origin_fromtimestamp = dt.datetime(1970,1,1)
delta_origins=946724399.999365 #[sec]
light_speed = 299792458.0 #[m/s] it is the value copied from "astropy.constants" ("from astropy.constants import c")
light_speed_AUd = light_speed*ms_to_AUd # [AU/day] speed of light in AU/day

# Auxiliar functions:
#1) time conversion functions:
def epoch_to_unixseconds(epoch): # input is a list (or tuple, or array...) of epochs (format: nanoseconds from SIS data); output is a 1-dimensional array of epochs in seconds 
    """
    conversion from Solar Orbiter epoch (in J2000 nanoseconds) to unix time seconds
    TODO: export this function to SOLO_auxilliary_functions, check which time offset is correct (see below). 
    Check whether this is the same offset for all EPD sensors and SWA data!  
    """
    #timedelta_J2000unix_seconds=(dt.datetime(2000,1,1,11,58,56)-dt.datetime(1970,1,1)).total_seconds()#my old offset
    #timedelta_J2000unix_seconds = 946720723.723854#Mario's offset
    #delta_origins=946720723.723854 #seconds between the reference epoch of the files (which is 2000/01/01 at 10h58m35s and 902163 microseconds) and the reference epoch of the function "datetime.datetime.fromtimestamp(number_of_seconds)" (which is 1970/01/01 at 1h0m0s and 0 microseconds)
    #dtu=timedelta_J2000unix_seconds
    ns_convfactor=1e-9
    seconds_J2000=epoch*ns_convfactor
    seconds_unix = []
    for i_ep in range(len(epoch)):
        if epoch[i_ep] <= 6568631839650079:
            timedelta_J2000unix_seconds = delta_origins - 3600
        elif epoch[i_ep] >= 6701688191893978 and epoch[i_ep] <= 6889175787893203:
            timedelta_J2000unix_seconds = delta_origins - 3600
        else: timedelta_J2000unix_seconds = delta_origins
        
        seconds_unix.append(seconds_J2000[i_ep] + timedelta_J2000unix_seconds)
    
    return np.array(seconds_unix)


def unixseconds_to_date(seconds_unix): # input is a list (or tuple, or array...) of epochs as seconds; output is a list (or tuple, or array...) of epochs (format: datetime.datetime)
    """
    conversion from unix seconds to date (as python datetime object). As the underlying function "dt.datetime.fromtimestamp" only works on single values (and not on arrays), this function includes a loop.
    TODO: If this loop limits the performance of the analysis code, it might be worth to do the conversion once for all files and save the date in new files. Alternatively, there might be a python date conversion function that works on arrays, but this was not easy to find ... (maybe look for it in astropy?)
    """
    dates=[]
    for i in range(len(seconds_unix)):
        dates.append(dt.datetime.fromtimestamp(seconds_unix[i]))
    return dates # it is a list