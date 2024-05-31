# import packages, functions, libraries...
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob
import cdflib
import matplotlib.patches as patches
from scipy.odr import *

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

# Figure parameters
fig_size = (15,7)
title_size = 14
xlabel_size = 14
ylabel_size = 14
sub_ylabel_size = 13
xticks_size = 14
yticks_size = 14
legend_size = 12
legend_position = 'upper right'
box_size = 14
box_position=(0.2, 0.86)

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

def epoch_to_unixdays(epoch): # input is one date (not a list) (format: nanoseconds from SIS data); output is one epoch in day number format
    epoch_list = [epoch]
    return epoch_to_unixseconds(epoch_list)[0]

def unixseconds_to_date(seconds_unix): # input is a list (or tuple, or array...) of epochs as seconds; output is a list (or tuple, or array...) of epochs (format: datetime.datetime)
    """
    conversion from unix seconds to date (as python datetime object). As the underlying function "dt.datetime.fromtimestamp" only works on single values (and not on arrays), this function includes a loop.
    TODO: If this loop limits the performance of the analysis code, it might be worth to do the conversion once for all files and save the date in new files. Alternatively, there might be a python date conversion function that works on arrays, but this was not easy to find ... (maybe look for it in astropy?)
    """
    dates=[]
    for i in range(len(seconds_unix)):
        dates.append(dt.datetime.fromtimestamp(seconds_unix[i]))
    return dates # it is a list

def unixdays_to_date(days_unix): # input is one epoch (not a list) in day number format; output is one date (not a list) (format: datetime.datetime)
    return dt.datetime.fromtimestamp(days_unix * 86400)

def convert_timetuple_to_date(ttuple):
    year=ttuple[0]
    month=ttuple[1]
    day=ttuple[2]
    return year,month,day

 
#2) functions to load Level-2 files from the SOLO public archive (SOAR):
#-can be probably also used/generalized to load other cdf data 

def _load_datefile(year, month, day, infile_path, infile_corname, show_loaded_files='yes'):
    """
    In this function the input is a date and a (relative or absolute) path, and the output is the file, but if this file doesn't exist the output is a message saying that it doesn't exist. The calibration version of the file is recognized automatically. Make sure that there never exist two files of the same date with different calibration versions in the same path (in case of doubt always use the latest/highest version).  
    """
    print ("filepath test")
    
    #reconstruct filename (including relative path), except for calibration-version name extension
    if month<10:
        date_path="%i/0%i/"%(year,month)    
    else:
        date_path="%i/%i/"%(year,month)
    if month<10 and day<10:#day and month with 1 digit each one ==> we add one zero in the day an one more in the month, in the left
        infile_nameext='%i0%i0%i'%(year,month,day)
    elif month<10:#month with 1 digit ==> we add one zero in the left of the month
        infile_nameext='%i0%i%i'%(year,month,day)
    elif day<10:#day with 1 digit ==> we add one zero in the left of the day
        infile_nameext='%i%i0%i'%(year,month,day) 
    else: #day and month with 2 digits ==> no add zeros
        infile_nameext='%i%i%i'%(year,month,day) 

        
    
    
    #find existing files with reconstructed file name (including calibration version) name extension
    pathfiles_search=infile_path+date_path+infile_corname+infile_nameext
    print("pathfiles_search:", pathfiles_search)
    pathfiles_found=glob.glob(pathfiles_search+'*')
    #print("pathfiles_found",pathfiles_found, len(pathfiles_found))
    print(len(pathfiles_found))
    if len(pathfiles_found)==0:
        if show_loaded_files == 'yes':
            print("path+file starting with %s doesn't exist"%(pathfiles_search))
        else: gfkhgftyf=0
    else:
        pathfile_found=pathfiles_found[0]
        if show_loaded_files == 'yes':
            print("path+file %s found"%(pathfile_found))
        else: gfkhgftyf=0
        infile=cdflib.CDF(pathfile_found)
        return infile    


def load_datefiles(first_date, last_date, infile_path, infile_corname, show_loaded_files):
    """
    This function returns the existing files from first_date to last_date (and it ignores missing days).
    The input date variables should be fist_date and last_date, which are arrays whith this format: [yyyy,mm,dd] 
    The infile_path should be the relative or absolute path to the (local) data directory
    Example: files_of_days([2020,7,26], [2020,9,1], path="/user/home/SOLO_data/EPD/SIS/")

    """
    #Transform inputs in a legible way to the code below:
    """
    first_year, first_month, first_day=first_date[0], first_date[1], first_date[2] #we difference the list "first_date" into year, month and day
    last_year, last_month, last_day=last_date[0], last_date[1], last_date[2] #we difference the list "last_date" into year, month and day
    first_datetime=dt.datetime(first_year, first_month, first_day) #we convert the numbers of the year, month and day of the first date into a date that Python understands
    last_datetime=dt.datetime(last_year, last_month, last_day) #we convert the numbers of the year, month and day of the last date into a date that Python understands
    """
    
    #convert date tuples into python datetime objects        
    first_year, first_month, first_day=convert_timetuple_to_date(first_date)
    last_year, last_month, last_day=convert_timetuple_to_date(last_date)
    first_datetime=dt.datetime(first_year, first_month, first_day)  
    last_datetime=dt.datetime(last_year, last_month, last_day) 
    
    #load input data cdf-files   
    delta_date=last_datetime-first_datetime #We calculate the period of time between last and first date
    n_days=delta_date.days+1 #We calculate the number of days of above period of time
    files=[]
    k=0
    for i in range(n_days):
        day_=first_datetime + dt.timedelta(days=i) #day_ is the day number i of the range of days selected, considering first_date as day number 0
        infile=_load_datefile(day_.year,day_.month,day_.day, infile_path, infile_corname, show_loaded_files) #we use the above function "_load_datefile" to obtain the file corresponding to that day
        if infile is None: #If the file corresponding this day doesn't exist...
            k=k+1 #Number of missing days (for example I put this because if not it returns an error)
        else: #If the file corresponding this day exist...
            files.append(infile) #We add the file of the day i to the list called "files"
            #Thus "files" list is only made by files which exist.
    return [files, n_days, first_datetime] #The function returns the list "files" which contains only the files of the selected period of time and that exist in the corresponding path or folder. 


#3) functions to resample epochs, fluxes and uncertainties of fluxes
# three functions to resample the epochs, fluxes (or rates) and uncertainties of fluxes (or rates). They are mostly used in time series.    
def resample_time(times,sampling_factor=2):
    f=sampling_factor
    Nx=np.shape(times)[0]
    l=int(Nx/f)
    Nxx=l*f
    a=times[0:Nxx].reshape(l, f)
    b=a.T[0]
    return b


def resample_flux(fluxes,sampling_factor=2):
    f=sampling_factor
    Nx=np.shape(fluxes)[0]
    l=int(Nx/f)
    Nxx=l*f
    Ny=np.shape(fluxes)[1]
    a=fluxes[0:Nxx].reshape(l, f, Ny)
    b=np.mean(a,axis=1) # in the original one that Nils sent me, here it used "sum", but I've changed it by "mean". This is because I want to calculate the average of counts or intensities during each resampled time step, instead of the sum. 
    return b

def resample_fluxerr(fluxeserr,sampling_factor=2):
    f=sampling_factor
    Nx=np.shape(fluxeserr)[0]
    l=int(Nx/f)
    Nxx=l*f
    Ny=np.shape(fluxeserr)[1]
    a=fluxeserr[0:Nxx].reshape(l, f, Ny)
    #return a
    b=np.sqrt(np.sum(a**2,axis=1)) / f # in the original one that Nils sent me, it was't divided by the sampling factor, but I've done it (because in the above function "resample_flux", I calculate the mean during each resampled time step). This is not totally correct because there can be several masked points in the original data that shouldn't be taken into account in the division. 
    return b

def resample_vector(onedimensional_array, sampling_factor=2):
    f=sampling_factor
    Nx=np.shape(onedimensional_array)[0]
    l=int(Nx/f)
    Nxx=l*f
    a=onedimensional_array[0:Nxx].reshape(l, f) 
    b=np.mean(a,axis=1)  
    return b


#4) properties of ion species that the sensor SOLO/EPD/SIS measures
def elements_properties(elem):
    """
    Function that returns parameters or details of the selected ion specie.
    - Input variable: the ID of an ion specie. It can be: 'H', 'He3', 'He4', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca' or 'Fe'
    - Output variables: mass [kg], number of nucleons and name (in lowercase letters, capital letters or only the first letter is capital; this is fot titles, labels,...)
    """
    if elem=='H':
        mass = 1.00784 * u_to_kg #kg
        number_of_nucleons = 1
        name, Name, NAME = 'hydrogen', 'Hydrogen', 'HYDROGEN'
        symbol = 'H'
    elif elem=='He3':
        mass = 3.0160293 * u_to_kg #kg
        number_of_nucleons = 3
        name, Name, NAME = 'helium 3', 'Helium 3', 'HELIUM 3'
        symbol = r'$^{3}$He'
    elif elem=='He4':
        mass = 4.002603254 * u_to_kg #kg
        number_of_nucleons = 4
        name, Name, NAME = 'helium 4', 'Helium 4', 'HELIUM 4'
        symbol = r'$^{4}$He'
    elif elem=='C':
        mass = 12.0107 * u_to_kg 
        number_of_nucleons = 12
        name, Name, NAME = 'carbon', 'Carbon', 'CARBON'
        symbol = 'C'
    elif elem=='N':
        mass = 14.0067 * u_to_kg
        number_of_nucleons = 14
        name, Name, NAME = 'nitrogen', 'Nitrogen', 'NITROGEN'
        symbol = 'N'
    elif elem=='O':
        mass = 15.999 * u_to_kg
        number_of_nucleons = 16
        name, Name, NAME = 'oxygen', 'Oxygen', 'OXYGEN'
        symbol = 'O'
    elif elem=='Ne':
        mass = 20.1797 * u_to_kg 
        number_of_nucleons = 20
        name, Name, NAME = 'neon', 'Neon', 'NEON'
        symbol = 'Ne'
    elif elem=='Mg':
        mass = 24.305 * u_to_kg
        number_of_nucleons = 24
        name, Name, NAME = 'magnesium', 'Magnesium', 'MAGNESIUM'
        symbol = 'Mg'
    elif elem=='Si':
        mass = 28.0855 * u_to_kg
        number_of_nucleons = 28
        name, Name, NAME = 'silicon', 'Silicon', 'SILICON'
        symbol = 'Si'
    elif elem=='S':
        mass = 32.065 * u_to_kg
        number_of_nucleons = 32
        name, Name, NAME = 'sulphur', 'Sulphur', 'SULPHUR'
        symbol = 'S'
    elif elem=='Ca':
        mass = 40.078 * u_to_kg 
        number_of_nucleons = 40
        name, Name, NAME = 'calcium', 'Calcium', 'CALCIUM'
        symbol = 'Ca'
    elif elem=='Fe':
        mass = 55.845 * u_to_kg
        number_of_nucleons = 56
        name, Name, NAME = 'iron', 'Iron','IRON'
        symbol = 'Fe'
    
    return {'mass': mass, 'number_of_nucleons': number_of_nucleons, 'name': name, 'Name': Name, 'NAME': NAME, 'symbol': symbol}

#5) data limits in time axis
def select_date_index(vector_to_cut, cutoff_date):
    """
    In the array or list of dates, this function returns the index of the input date called "cutoff_date". This index corresponds to the flux or rate value of this date. 
    The format of ''cutoff_date'' should be: (year, month, day, hour) or (year, month, day, hour, min)
    
    How this function works: it is reading the array or list of epochs from the beginning and stops in the epoch that agrees with "cutoff_date". 
    """
    vc = vector_to_cut
    
    for i in range(len(vc)):
        
        yy = vc[i].year
        mm = vc[i].month
        dd = vc[i].day
        hh = vc[i].hour
        
        # If we do not use minutes in "cutoff_date":
        if len(cutoff_date)==4 and yy==cutoff_date[0] and mm==cutoff_date[1] and dd==cutoff_date[2] and hh==cutoff_date[3]:
            date_index = i
            break
        # If we use minutes in "cutoff_date":
        elif len(cutoff_date)==5 and yy==cutoff_date[0] and mm==cutoff_date[1] and dd==cutoff_date[2] and hh==cutoff_date[3]:
            minmin = vc[i].minute
            if minmin==cutoff_date[4]:
                date_index = i
                break

        elif len(cutoff_date)!=5 and len(cutoff_date)!=4: print('cutoff_date wrong length, length of date tuple is: ',len(cutoff_date))

    return date_index

#6) maximum of a timeseries of intensities or rates, and its uncertainty
def x_cross(xl,yl,xu,yu,y_cross):
    """
    There are 2 points in the graph ((xl,yl) and (xu,yu)) connected by a straight line. There is also a horizontal line (y=y_cross) which cuts the other line. This fuction calculates the value x of the crossing point between the 2 lines (so the point is (x_cross(xl,yl,xu,yu,y_cross), y_cross)). 
    """
    a_slope = (yu-yl)/(xu-xl)
    b_xintercept = yu - a_slope * xu
    
    return (y_cross - b_xintercept) / a_slope

def x_unc(time, Flux_j, d_Flux_j): #either rates or fluxes can be used
    """
    This function calculates the x and y values of the maximum of the vector "Rate_j". This is used in "time_speed_maximum" function to calculate the x and y values of the maximum of a time series for one "elem" and one channel. 
    
    - Input:
        - time: vector of epoch corresponding to one "elem" and one channel. It should have been cut with "boundaries" function. Type: vector. Units: [days]. 
        - Rate_j: vector of rate corresponding to one "elem" and one channel. It should have been cut with "boundaries" function. Type: vector. Units: [number of counts].
    
    - Output: 
         - 'x1' and 'x2' = lower and upper boundaries of x uncertainty, respectively. Type: scalar. Units: [day].
         - 'x_max':  is the epoch which corresponds to the maximum value of rate or intensity (is one of the points of speed that we use in speed vs time graph). Type: scalar. Units: [day].
         - 'y_max': maximum value of speed array. Type: scalar. Units: [number of counts].
         - 'dy_max': y_max - sqrt(y_max). It is the lower uncertainty of y. Type: scalar. Units: [number of counts].
         - 'ind_ymax': index of the "time" or "Rate_j" corresponding to the "Rate_j" maximum. It can be used for flux (or rate) and for epoch. Type: scalar. Format: integer. Units: (no units). 
    """
    tt, yy, dyy = time, Flux_j, d_Flux_j #change name
    y_max = max(yy) # y_max is the maximum value of speed array
    ind_ymax = np.argmax(yy) # index of y_max
    dy_max_low = d_Flux_j[ind_ymax]
    y_max_low = y_max-dy_max_low
    x_max = tt[ind_ymax]
    
    # cut the arrays of speeds and times in the y_max index (included y_max): 
    yy1=yy[0:ind_ymax+1][::-1] # yy before y_max and reversed
    tt1=tt[0:ind_ymax+1][::-1] # tt before y_max and reversed
    if min(yy1)>=y_max_low:
        #x1=tt1[np.argmin(yy1)]
        x1=tt[0]
    else: 
        k=0
        ind1=0
        while yy1[k]>y_max_low:
            ind1=k
            k=k+1
        x1=x_cross(tt1[ind1], yy1[ind1], tt1[ind1+1], yy1[ind1+1], y_max_low)
    # cut the arrays of speeds and times in the y_max index (included y_max): 
    yy2=yy[ind_ymax:] # yy after y_max
    tt2=tt[ind_ymax:] # tt after y_max
    if min(yy2)>=y_max_low:
        #x2=tt2[np.argmin(yy2)]
        x2=tt[len(tt)-1]
    else: 
        k=0
        while yy2[k]>y_max_low:
            ind2=k
            k=k+1
        x2=x_cross(tt2[ind2], yy2[ind2], tt2[ind2+1], yy2[ind2+1], y_max_low)
    
    return {'x1': x1, 'x2': x2, 'x_max': x_max, 'y_max': y_max, 'dy_max_low': dy_max_low, 'y_max_low': y_max_low, 'ind_max': ind_ymax} 
  # type:  {scalar,   scalar,    scalar,         scalar,          scalar,           scalar}
  # units :{day,      day,       day,            counts,          counts,           (no units)}

#7) fuction to fit the velocity dispersion: v=s/(t-t_0)
def speed_fit_odr(t0_s, t):
    """
    It is the function we use for the fit. v=s/(t-t_0)
    -The measurements of "t" and "vspeed" (which are the time of the maximum calculated above (the time when the largest number of particles reaches the probe) and the speeds of these particles (calculated by the quadratic average of the channels' energies), respectively) should be fitted to this fuction "vspeed". 
    -The parameters we want to calculate are "t_0" and "s" (the moment of the impulsive event and the distance between the probe and the impulsive event following the Parker spiral, respectively). 
    """
    return t0_s[1]/(t-t0_s[0]) # t0==t0_s[0]; s=t0_s[1]


#8) functions to fit the spectra
def flux_sp(slor0, x_Enuc):
    """
    - Input variables:
    
        - "slor0": it is a list or tuple with two items. The first item is the initial value os the slope and the second one is the ordered at the origin. Therefore slor0 = [slope, ordered at the origin]. What slope and what ordered at the origin? Those that result from making the base 10 logarithm of the present function (see the function "log_flux_sp"). 
        
        - "x_Enuc": x axis, which is the 1-dimensional array or list with the values of energy per nucleon corresponding to the energy channels, of one ion specie. 
        
    - Return: y axis values using the x_Enuc as x axis, and slor0[0] and slor0[1] as parameters. This values correspond to the intensities given by the model (fit) of the spectrum. 
        
    """
    
    return 10**slor0[1] * x_Enuc**slor0[0]

def log_flux_sp(slor0, x_log_Enuc):
    """
    This fuction is like "flux_sp" but in logarithmic scale. 
    
    - Input variables:
    
        - 'x_log_Enuc': x axis, which is the 1-dimensional array or list with the values of log10(energy per nucleon) corresponding to the energy channels, of one ion specie. 
        
        - "slor0": it is a list or tuple with two items. The first item is the initial value os the slope and the second one is the ordered at the origin. Therefore slor0 = [slope, ordered at the origin]. 
                
    - Return: y axis values using the x_log_Enuc as x axis, and slor0[0] and slor0[1] as parameters. This values correspond to the logarithm in base 10 of the intensities, given by the model (fit) of the spectrum.  
        
    """
    return  slor0[0] * x_log_Enuc + slor0[1] 
    
def fit_ODR(xodr, yodr, dxodr, dyodr, init_values):

    # fit with scipy.odr
    model = Model(speed_fit_odr)
    mydata = RealData(x=xodr, y=yodr, sx=dxodr, sy=dyodr)
    myodr = ODR(mydata, model, beta0=init_values)
    myoutput = myodr.run()

    # parameters from the fit
    s = myoutput.beta[1] # path length
    ds = np.sqrt(np.diag(myoutput.cov_beta))[1] # uncertainty of the path length
    t0_days = myoutput.beta[0] + init_values[2] # injection time (format: number of day)
    t0 = unixdays_to_date(t0_days)
    dt0_days = np.sqrt(np.diag(myoutput.cov_beta))[0] # uncertainty of t0_days. Units: [days]
    chi2red = myoutput.res_var # reduced chi squared

    # x and y axes of the fit curve (model)
    xfit_days = np.linspace(min(xodr),max(xodr), 200) + init_values[2] # [days]
    xfit = unixseconds_to_date(xfit_days * 86400) # [date] x axis of the fit curve (the model)
    yfit = speed_fit_odr([t0_days-init_values[2], s], xfit_days-init_values[2]) # y axis of the fit curve (the model)

    # box of annotations
    t0_annotation = t0.replace(microsecond = 0) #we delete microseconds
    dt0_annotation = round(dt0_days*24*60)
    s_annotation = round(s, 3)
    ds_annotation = round(ds, 3)
    chi2red_annotation = round(chi2red, 3) 
    annotation_string = r"""
    time of injection: {0} $\pm$ {3} minutes
    path length {1} $\pm$ {4} AU
    {5} = {2}""".format(t0_annotation, s_annotation, chi2red_annotation, dt0_annotation, ds_annotation, r'$\chi^2_{red}$')

    return {'xfit': xfit, 'yfit': yfit, 's': s, 'ds': ds, 't0': t0, 't0_days': t0_days, 'dt0_days': dt0_days, 'chi2red': chi2red, 'annotation_string': annotation_string}



#9) arch length 
def archlength(r, theta):
    """
    It returns the archlength of Archimedes spiral given the angle (theta [rad]) and the radial distance (r). 
    """
    b=r/theta
    return b/2*(theta*np.sqrt(1+theta**2) + np.log(theta + np.sqrt(1+theta**2)))

    
#10) fit spectrum
def flux_sp(slor0, x_Enuc):
    """
    - Input variables:
    
        - "slor0": it is a 1-dimensional array, list or tuple with two items. The first item is the initial value of the slope and the second one is the ordered at the origin. Therefore slor0 = [slope, ordered at the origin]. What are slope and ordered at the origin? Those that result from making the base 10 logarithm of the present function. 
        
        - "x_Enuc": x axis, which is the 1-dimensional array, list or tuple with the values of energy per nucleon corresponding to the energy channels, of one ion specie. 
        
    - Return: y axis values using the x_Enuc as x axis, and slor0[0] and slor0[1] as parameters. This values correspond to the intensities given by the model (fit) of the spectrum. 
        
    """
    
    return 10**slor0[1] * x_Enuc**slor0[0]

def fit_straight_line(x, y, dx, dy, init_values):
    
	# fit with scipy.odr
	model = Model(flux_sp)
	mydata = RealData(x=x, y=y, sx=dx, sy=dy)
	myodr = ODR(mydata, model, beta0=[init_values[0], init_values[1]])
	myoutput = myodr.run()
	# parameters from the fit
	ordered_at_the_origin = myoutput.beta[1]
	d_ordered_at_the_origin = np.sqrt(np.diag(myoutput.cov_beta))[1]
	slope = myoutput.beta[0]
	d_slope = np.sqrt(np.diag(myoutput.cov_beta))[0]
	chi2red = myoutput.res_var
	# model curve
	x_fit = np.linspace(min(x), max(x), 200)
	y_fit = flux_sp([slope, ordered_at_the_origin], x_fit)
	
	return {'slope': slope, 'd_slope': d_slope, 'ordered_at_the_origin': ordered_at_the_origin, 'd_ordered_at_the_origin': d_ordered_at_the_origin, 'chi2red': chi2red, 'xfit': x_fit, 'yfit': y_fit}
    
    
