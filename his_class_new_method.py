import cdflib
import numpy as np
from numpy import append as ap
import scipy as sp
import datetime as dt


import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.path import Path
from datetime import datetime, timedelta
import os

from auxfuncs import epoch_to_unixseconds,unixseconds_to_date




#General plot definitions

my_cmap = cm.get_cmap("jet",1024*16)
my_cmap.set_under('w')

class HIS_PHA(object):
    def __init__(self,start_date=(2022,9,10),end_date=(2022,9,30),version="V02",inpath="Data/", load_prates=True):
        #Getting all the dates between start and end as tuple
        start_date_dt = datetime(*start_date)
        end_date_dt = datetime(*end_date)
        dates = []
        current_date = start_date_dt
        while current_date <= end_date_dt:
            dates.append((current_date.year, current_date.month, current_date.day))
            current_date += timedelta(days=1)
        print(dates)    
        #PHA data arrays
        self.epoch_period = []
        self.step_period = []
        self.eoq_period = []
        self.tof_period = []
        self.essd_period = []
        self.azim_period = []
        self.elev_period = []
        self.pr_period = []
        self.ueoq_period = []
        self.uelev_period = []
        self.uepoch_period = []
        self.udoy_period = []
        self.doy_period = []
        
        #Rates data arrays
        self.rates_period = []
        self.vels_He2_period = []
        self.uvels_He2_period = []
        

        for i in range(len(dates)):
            print(i)
            date = dates[i]
            #load file (includes 1 day of HIS PHA data (=raw data))
            dyear=date[0]
            dmon=date[1]
            dday=date[2]
            syear=str(dyear)
            if dmon<10:
                smon="0"+str(dmon)
            else:
                smon=str(dmon)
            if dday<10:
                sday="0"+str(dday)
            else:
                sday=str(dday)
            sdate=syear+smon+sday   
            self.date_str = sdate
            filename_core="solo_L2_swa-his-pha"
            filename=filename_core+"_"+sdate+"_"+version+".cdf"
            try:
                self.cdf_file=cdflib.CDF(inpath+filename) 
                print("loaded %s"%(filename))
                self.eoq = self.cdf_file.varget("PHA_EOQ")#in kV
                self.ueoq=np.unique(self.eoq)
                
                self.epoch = self.cdf_file.varget("EPOCH")
                self.uepoch,self.inverse=np.unique(self.epoch,return_inverse=True)
                self.uepoch_seconds = epoch_to_unixseconds(self.uepoch)
                self.utime = np.array(unixseconds_to_date(self.uepoch_seconds))
                self.udoy = [(t - dt.datetime(t.year, 1, 1)).days + 1 + t.hour / 24 + t.minute / 1440 + t.second / 86400 + t.microsecond / 86400000000 for t in self.utime]
                
                self.rdata=HIS_Rates(date=date, inpath=inpath)
                self.rates=self.rdata.rates
                if len(self.rates) == len(self.udoy):
                    if len(self.ueoq) != 64:
                        pass
                    else:
                        #load data products
                        self.epoch = self.cdf_file.varget("EPOCH")  
                        self.epoch_period.append(self.epoch)      
                        self.step = self.cdf_file.varget("PHA_EOQ_STEP")
                        self.step_period.append(self.step)
                        self.eoq = self.cdf_file.varget("PHA_EOQ")#in kV
                        self.eoq_period.append(self.eoq)
                        self.tof = self.cdf_file.varget("PHA_TOF")
                        self.tof_period.append(self.tof)
                        self.essd = self.cdf_file.varget("PHA_SSD_Energy")
                        self.essd_period.append(self.essd)
                        self.azim = self.cdf_file.varget("PHA_AZIMUTH")
                        self.azim_period.append(self.azim)
                        self.azim_rad = np.pi/180*self.azim

                        self.elev = self.cdf_file.varget("PHA_ELEVATION")
                        self.elev_period.append(self.elev)
                        self.elev_rad = np.pi/180*self.elev
                        self.pr = self.cdf_file.varget("PHA_PRIORITIZATION_RANGE") 
                        self.pr_period.append(self.pr)
                        self.eoq_delta = self.cdf_file.varget("EOQ_Delta")
                        self.qual = self.cdf_file.varget("QUALITY_FLAG")

                        #get unique data product arrays (to be used below in histograms, etc.)
                        self.ustep=np.unique(self.step)
                        self.ueoq=np.unique(self.eoq)
                        self.ueoq_period.append(self.ueoq)
                        self.uelev=np.unique(self.elev)
                        self.uelev_period.append(self.uelev)
                        self.uazim=np.unique(self.azim)
                        self.uazim_rad=np.unique(self.azim_rad)
                        self.uepoch,self.inverse=np.unique(self.epoch,return_inverse=True)
                        self.uepoch_period.append(self.uepoch)
                        self.uepoch_seconds = epoch_to_unixseconds(self.uepoch)
                        self.utime = np.array(unixseconds_to_date(self.uepoch_seconds))
                        
                        self.udoy = [(t - dt.datetime(t.year, 1, 1)).days + 1 + t.hour / 24 + t.minute / 1440 + t.second / 86400 + t.microsecond / 86400000000 for t in self.utime]
                        self.udoy = np.array(self.udoy)
                        self.udoy_period.append(self.udoy)
                        self.doy=self.udoy[self.inverse]
                        self.doy_period.append(self.doy)
                        
                        #derive data quantities for He2+ (=alphas)
                        self.pr_helium=5#priority range for helium (includes He2+ (=main species) and He1+)
                        self.steprange=np.arange(0,64,1)
                        self.uvels_He2=self.calc_vel(self.steprange,m=4,q=2)
                        self.uvels_He2_period.append(self.uvels_He2)
                        self.vels_He2=self.calc_vel(self.step,m=4,q=2)
                        self.vels_He2_period.append(self.vels_He2)
                        
                        self.umeanspeeds=None
                        self.umeanspeeds_uncor=None    
                        
                        if load_prates==True:
                            self.rdata=HIS_Rates(date=date, inpath=inpath)
                            self.repoch=self.rdata.epoch #should alwys be equal to unique(self.epoch)!
                            self.prange=self.rdata.prange
                            self.rates=self.rdata.rates
                            self.rates_period.append(self.rates)
                            self.rates_alpha=self.rdata.rates_alpha
                else:
                    print("PR and udoy not matching") 
                
                
            except FileNotFoundError:
                pass
            
        self.epoch_period = np.hstack(self.epoch_period)
        self.step_period = np.hstack(self.step_period)
        self.eoq_period = np.hstack(self.eoq_period)
        self.tof_period = np.hstack(self.tof_period)
        self.essd_period = np.hstack(self.essd_period)
        self.azim_period = np.hstack(self.azim_period)
        self.elev_period = np.hstack(self.elev_period)
        self.pr_period = np.hstack(self.pr_period)
        self.ueoq_period = np.hstack(self.ueoq_period)
        self.uelev_period = np.hstack(self.uelev_period)
        self.uepoch_period = np.hstack(self.uepoch_period)
        self.udoy_period = np.hstack(self.udoy_period)
        self.doy_period = np.hstack(self.doy_period)
        
        self.uvels_He2_period = np.unique(self.uvels_He2_period)
        self.vels_He2_period = np.hstack(self.vels_He2_period)
        self.uvels_He2_period = np.hstack(self.uvels_He2_period)
    def calc_vel(self,step,m,q):
        """
        Method to calculate speed from Epq-steps for a given ion species with mass m (in amu) and charge q (in e)
        """
        Eoq=self.ueoq[step]#in kV
        m_amu=1.660539e-27 #in kg
        e=1.602177e-19 #in C
        E=1e3*Eoq*q*e #in J
        v=np.sqrt(2*E/(m*m_amu))/1000.#in km/s
        return v
    
    def get_he2_data(self,save_to_file=False): 
        """
        Method to get ET-matrix for a given E/q-step and priority range. 
        Input: If uepoch_ind="all" is selected, the ET-matrix histogram is accumulated over the whole day (=file content). Otherwise the histogram is accumulated over time stamp of the given index uepoch_ind=i. 
        """
        sq_list = []
        dateipfad = 'he2+_boxes.txt'
        for i in range(63):
            with open(dateipfad, 'r') as datei:
                for index, zeile in enumerate(datei):
                    if index == i:
                        sq = zeile.strip().split(',')
                        sq = list(map(int,sq)) 
            sq_list.append(sq)   
        
        C = []
        matrices = []
        
        for j in range(8):
            for k in range(16):
                print(k)
                vels = []
                doy = []
                for i in range(63):
                    mask = (self.pr_period==j)*(self.elev_period==self.uelev_period[k])*(self.step_period==i)*(self.tof_period>sq_list[i][1])*(self.tof_period<sq_list[i][2])*(self.essd_period>sq_list[i][3])*(self.essd_period<sq_list[i][4])
                    vels.append(self.vels_He2_period[mask])
                    doy.append(self.doy_period[mask])
                vels = np.hstack(vels)
                doy = np.hstack(doy)
                bins_doy=ap(self.udoy_period, self.udoy_period[-1]+(self.udoy_period[-1]-self.udoy_period[-2]))
                bins_vels=ap(self.uvels_He2_period, self.uvels_He2_period[-1]+(self.uvels_He2_period[-1]-self.uvels_He2_period[-2]))
                C_h,tb,vb=np.histogram2d(doy, vels,bins=[bins_doy,bins_vels])
                pr_rates_list = []
                for  i in range(len(self.rates_period)):
                    pr_rates_list.append(self.rates_period[i][:,j,:,k])
                pr_rates = np.concatenate(pr_rates_list,axis=0)
                
                Cv = C_h*pr_rates
                matrices.append(Cv)
            
            C_pr = np.sum(matrices,axis=0)
            C.append(C_pr)  
            
        C = np.sum(C,axis=0)

        plt.figure()
        plt.xlabel('time [DOY]')
        plt.title('HIS timeseries')
        plt.ylabel('velocity [km/s]')
        plt.pcolormesh(tb[:-1],vb[:-1],C.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(C)))) 
        plt.colorbar()
        save_path = f"C:/Users/lukas.bertoli/Desktop/HIS-Data-analysis/Code/PHA_plots/{round(self.udoy_period[0])}-{round(self.udoy_period[-1])}.png"
        plt.savefig(save_path)


        mean_speed = np.sum(C*vb[:-1], axis=1)/np.sum(C, axis=1)
        sum_counts = np.sum(C, axis=1)
        
        thermal_speed = [] 
        for j in range(len(tb[:-1])):
            th_speed = (np.sum(C[j]*(mean_speed[j]-vb[:-1])**2) / np.sum(C[j]))**0.5
            thermal_speed.append(th_speed)

        try:
            if save_to_file==True:
                filename = f"his_timeseries_{round(self.udoy_period[0])}-{round(self.udoy_period[-1])}"
                np.savetxt('his_timeseries_calc/'+filename+'.txt', np.column_stack((tb[:-1], sum_counts, mean_speed, thermal_speed)), delimiter=' ')  
        except Exception as e:
            print(e)
            pass
    def clear_attributes(self):
        for key in list(self.__dict__.keys()):
            self.__dict__[key] = None
    
class HIS_Rates(object):            
    """
    Auxilliary class to load the HIS priority rates to correct the PHA data.
    Object of this class is called in the HIS_PHA class.
    """
    def __init__(self,date=(2022,9,16),version="V02",inpath="Y:/HIS/2022/06/"): 
        
        #load file (includes 1 day of HIS PHA data (=raw data)
        dyear=date[0]
        dmon=date[1]
        dday=date[2]
        syear=str(dyear)
        if dmon<10:
            smon="0"+str(dmon)
        else:
            smon=str(dmon)
        if dday<10:
            sday="0"+str(dday)
        else:
            sday=str(dday)
        sdate=syear+smon+sday   
        filename_core="solo_L2_swa-his-rates"
        filename=filename_core+"_"+sdate+"_"+version+".cdf"
        
        print(inpath+filename)
        self.cdf_file=cdflib.CDF(inpath+filename) 
        print("loaded %s"%(filename))
        
        #load data products
        self.epoch = self.cdf_file.varget("EPOCH")
        self.prange = self.cdf_file.varget("PRIORITY")        
        self.rates = self.cdf_file.varget("PRIORITY_RATE")
        self.rates_alpha = self.rates[:,5,:,:]