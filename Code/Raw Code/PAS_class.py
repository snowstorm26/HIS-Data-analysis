import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from SIS_auxfuncs import epoch_to_unixseconds, unixseconds_to_date, convert_timetuple_to_date, _load_datefile,  load_datefiles, select_date_index, resample_time, resample_vector



#input that needs to be adapted by user:
#infile_path_PASdatab="/media/marior/Seagate Expansion Drive/MASTER/TFM/SOARdata/SWA/PAS/"#local (absolute) path to SWA/PAS data
infile_path_PASdata="/users/nils.janitzek/Projects/SOLO/SWA/PAS_Moments/Data/"#local (absolute) path to SWA/PAS data


class PAS(object):
    """
    Class to load and analyze SOLO/SWA/PAS data. So far only Level-2 proton moment data is included (calculated on ground). The data products have self-explinatory names. More info can be found in the SWA instrument paper (Owen et al. (2020)) and the public SWA data documentation within the Solar Orbiter archive.
    TODO: 
    - method to exclude invalid values (obvious out-of-range values and too bad quality - once the mode and quality data products are fully understood). 
    - method to resample data (if needed due too larger value fluctuations) 
    """
    def __init__(self, first_date, last_date, show_loaded_files):

        #convert date tuples into python datetime objects
        first_year, first_month, first_day=convert_timetuple_to_date(first_date)
        last_year, last_month, last_day=convert_timetuple_to_date(last_date)
        first_datetime=dt.datetime(first_year, first_month, first_day)  
        last_datetime=dt.datetime(last_year, last_month, last_day) 
        
        #load measurement data as cdf-files 
        Files_of_days = load_datefiles(first_date, last_date, infile_path=infile_path_PASdata,infile_corname='solo_L2_swa-pas-grnd-mom_', show_loaded_files=show_loaded_files)
        self.cdf_files = Files_of_days[0] #self 
        self.n_days_total = Files_of_days[1] #self 
        self.first_datetime = Files_of_days[2] #self 
        self.n_days=len(self.cdf_files) #self #number of days (of existing files) or number of files 
        
        #extract data products from the cdfs
        usecs=[]
        dates=[]
        self.original_time_resolution=[]
        numdens=[]
        vmeans_r=[]
        vmeans_t=[]
        vmeans_n=[]
        vmeans_rtn=[]
        temps=[]
        qfs=[]
        mode_infos=[]
        i=0
        while i<self.n_days:
            c=self.cdf_files[i]
            epoch=c.varget("EPOCH")
            usec=epoch_to_unixseconds(epoch)
            date=unixseconds_to_date(usec)
            time_resolution_day = usec[1] - usec[0]
            self.original_time_resolution.append(time_resolution_day) # time resolution of the original data in seconds
            dens=c.varget("N")
            velmean_rtn=c.varget("V_RTN")
            vmean_r=velmean_rtn.T[0]
            vmean_t=velmean_rtn.T[1]
            vmean_n=velmean_rtn.T[2]
            #TODO: make here a check for invalid (out-of-range) values!
            vmean_rtn=np.sqrt(vmean_r**2+vmean_t**2+vmean_n**2)
            temp=c.varget("T")
            qf=c.varget("quality_factor")
            mode_info=c.varget("INFO")

            usecs.append(usec)
            dates.append(date)
            numdens.append(dens)
            vmeans_r.append(vmean_r)
            vmeans_t.append(vmean_t)
            vmeans_n.append(vmean_n)
            vmeans_rtn.append(vmean_rtn)
            temps.append(temp)
            qfs.append(qf)
            mode_infos.append(mode_info)
            i=i+1
        
        self.usec=np.concatenate(usecs) # time [seconds]. 1-dimensional array of floats
        self.Epoch=np.concatenate(dates) # time [dates]. 1-dimensional array of dates with format datetime.datetime
        self.numdens=np.concatenate(numdens) # number density [particles cm^-3]. 1-dimensional array of floats
        self.vmean_r=np.concatenate(vmeans_r) # radial speed [km/s] in RTN frame: "Sun-Spacecraft direction, positive antisunward". 1-dimensional array of floats
        self.vmean_t=np.concatenate(vmeans_t) # tangential speed [km/s] in RTN frame: "Completes Right Handed Set". 1-dimensional array of floats
        self.vmean_n=np.concatenate(vmeans_n) # normal speed [km/s] in RTN frame: "Projection of Solar North on plane perpendicular to X". 1-dimensional array of floats 
        self.vmean_rtn=np.concatenate(vmeans_rtn) # magnitude of the speed [km/s]. 1-dimensional array of floats
        self.temp=np.concatenate(temps) # temperature [eV]. 1-dimensional array of floats
        self.quality=np.concatenate(qfs) # quality factor [unitless]. 1-dimensional array of floats
        self.mode_info=np.concatenate(mode_infos) # info [unitless]. 1-dimensional array of integers
    
    def timeseries(self, sampling_factor, limits): 
        """
        Prepare data to do time series. This method cuts the arrays in the limits defined by the input variable "limits". It also resample the cut arrays with a time resolution of "original_time_resolution * sampling_factor". The new value of each new time step is the average of the original resampled values of that time step. 
        
        TODO: the time resolution sometimes changes (I have seen for one day (one file) time resolutions of 0.25, 4 and 12 seconds). The sampling factor used in this method is multiplied directly by the original time resolution. This creates a ney time resolution which is "original_time_resolution * sampling_factor". Therefore if the original time resolution changes, the new time resolution is also different depending on the time. Hence, I should do someting here. Anyway this is not a great problem for the plot of PAS data in the paper of the analysis of the double impulsive and gradual event of March 5-10, 2022, because we use very short time steps (maybe the original time resolution), so the change of time resolution is unobservable. 
        TODO: I should check the other sensors and instruments (MAG, EPD/EPT and EPD/SIS) to see if there are changes of time resolution as in SWA/PAS. I observed that EPD/SIS has 30 seconds always, EPD/EPT changes as SWA/PAS, and I do not know about MAG.
        
        
        # limits of the time series 
        ind1 = select_date_index(self.Epoch, cutoff_date=limits[0]) #index of the first date where you want to cut 
        ind2 = select_date_index(self.Epoch, cutoff_date=limits[1]) #index of the second date where you want to cut 
        self.ind1, self.ind2 = ind1, ind2
        
        # cut and resampled variables 
        self.usec_ts = resample_time(self.usec[ind1:ind2], sampling_factor)
        self.Epoch_ts = resample_time(self.Epoch[ind1:ind2], sampling_factor)
        self.numdens_ts = resample_vector(self.numdens[ind1:ind2], sampling_factor)
        self.vmean_r_ts = resample_vector(self.vmean_r[ind1:ind2], sampling_factor)
        self.vmean_t_ts = resample_vector(self.vmean_t[ind1:ind2], sampling_factor)
        self.vmean_n_ts = resample_vector(self.vmean_n[ind1:ind2], sampling_factor)
        self.vmean_rtn_ts = resample_vector(self.vmean_rtn[ind1:ind2], sampling_factor)
        self.temp_ts = resample_vector(self.temp[ind1:ind2], sampling_factor)
        #self.quality_ts = resample_vector(self.quality[ind1:ind2], sampling_factor)
        """
        
        # cut and resampled variables 
        self.usec_ts = resample_time(self.usec, sampling_factor)
        self.Epoch_ts = resample_time(self.Epoch, sampling_factor)
        self.numdens_ts = resample_vector(self.numdens, sampling_factor)
        self.vmean_r_ts = resample_vector(self.vmean_r, sampling_factor)
        self.vmean_t_ts = resample_vector(self.vmean_t, sampling_factor)
        self.vmean_n_ts = resample_vector(self.vmean_n, sampling_factor)
        self.vmean_rtn_ts = resample_vector(self.vmean_rtn, sampling_factor)
        self.temp_ts = resample_vector(self.temp, sampling_factor)
        self.quality_ts = resample_vector(self.quality, sampling_factor)
        
    
    def plot_timeseries(self, sampling_factor, limits, plot_prods=["dens","speed","temp","qual","mode"]):
        """
        Method to plot several proton-VDF-moment data products as time series.
        Input: first and last date are given as tuples with format"(yyyy,mm,dd)"
        TODO: Include further products (if needed).
        """
        
        self.timeseries(sampling_factor, limits) # create cut and resampled variables using the method "timeseries" 
        n=len(plot_prods)
        if n>1:
            fig, axs=plt.subplots(n,1,figsize=(25,10),sharex=True) #create figure and axes
            i=0
            axs[i].set_title("SOLO/SWA/PAS proton VDF moments")
            if "dens" in plot_prods:
                axs[i].plot(self.Epoch_ts,self.numdens_ts,marker='o',markersize=1,linestyle="None",color="k",label="n")
                axs[i].set_ylabel("number density\n[1/cm^3]")
                axs[i].legend(loc="upper right")
                i+=1
            if "speed" in plot_prods:
                axs[i].plot(self.Epoch_ts,self.vmean_rtn_ts,marker='o',markersize=1,linestyle="None",color="k",label="|v|")
                axs[i].set_ylabel("mean speed (RTN)\n[km/s]")
                axs[i].legend(loc="upper right")
                i+=1
            if "temp" in plot_prods:
                axs[i].plot(self.Epoch_ts,self.temp_ts,marker='o',markersize=1,linestyle="None",color="k",label="T")
                axs[i].set_ylabel("kin. temperature\n[eV]")
                axs[i].legend(loc="upper right")
                i+=1
            if "qual" in plot_prods:
                axs[i].plot(self.Epoch,self.quality,marker='o',markersize=1,linestyle="None",color="r",label="qual.")
                axs[i].set_ylabel("quality factor")
                axs[i].legend(loc="upper right")
                i+=1
            if "mode" in plot_prods:
                axs[i].plot(self.Epoch,self.mode_info,marker='o',markersize=1,linestyle="None",color="b",label="mode")
                axs[i].set_ylabel("PAS op. mode")
                axs[i].legend(loc="upper right")
                i+=1
            axs[i-1].set_xlabel("time")
                
            #if len(limits[0]) == 4:
            #    first_xlim = dt.datetime(limits[0][0], limits[0][1], limits[0][2], limits[0][3])
            #elif len(limits[0]) == 5:
            #    first_xlim = dt.datetime(limits[0][0], limits[0][1], limits[0][2], limits[0][3], limits[0][4])
            #if len(limits[1]) == 4:
            #    last_xlim = dt.datetime(limits[1][0], limits[1][1], limits[1][2], limits[1][3])
            #elif len(limits[1]) == 5:
            #    last_xlim = dt.datetime(limits[1][0], limits[1][1], limits[1][2], limits[1][3], limits[1][4])
            #axs[0].set_xlim(first_xlim, last_xlim)
            axs[0].set_xlim(self.Epoch_ts[0], self.Epoch_ts[len(self.Epoch_ts)-1])
            
            plt.show(block=False)
        else:
            print("Error: No plot could be created. Please include at least two of the following products in 'plot_prods': ['dens','speed','temp','qual','mode']")
            

