import cdflib
import numpy as np
from numpy import append as ap
import scipy as sp
import datetime as dt

import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import cm, cbook
import matplotlib.colors as colors
from matplotlib.colors import Normalize, LogNorm
import matplotlib.colors as mcolors
from matplotlib.path import Path
from physt import cylindrical, polar, spherical
from scipy.optimize import curve_fit

from auxfuncs import epoch_to_unixseconds,unixseconds_to_date

import imageio
import os



#General plot definitions

my_cmap = cm.get_cmap("jet",1024*16)
my_cmap.set_under('w')

class HIS_PHA(object):
    """
    Class to analyze SolO/SWA/HIS data.
    Class methods are explained shortly in a docstring for each method. More documentation will follow...
    """
    def __init__(self,date=(2022,7,18),version="V02",inpath="Data/", load_prates=False):
        
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
        self.date_str = sdate
        filename_core="solo_L2_swa-his-pha"
        filename=filename_core+"_"+sdate+"_"+version+".cdf"
        self.cdf_file=cdflib.CDF(inpath+filename) 
        print("loaded %s"%(filename))
        
        #load data products
        self.epoch = self.cdf_file.varget("EPOCH")        
        self.step = self.cdf_file.varget("PHA_EOQ_STEP")
        self.eoq = self.cdf_file.varget("PHA_EOQ")#in kV
        self.tof = self.cdf_file.varget("PHA_TOF")
        self.essd = self.cdf_file.varget("PHA_SSD_Energy")
        self.azim = self.cdf_file.varget("PHA_AZIMUTH")
        self.azim_rad = np.pi/180*self.azim

        self.elev = self.cdf_file.varget("PHA_ELEVATION")
        self.elev_rad = np.pi/180*self.elev
        self.pr = self.cdf_file.varget("PHA_PRIORITIZATION_RANGE") 
        self.eoq_delta = self.cdf_file.varget("EOQ_Delta")
        self.qual = self.cdf_file.varget("QUALITY_FLAG")

        #get unique data product arrays (to be used below in histograms, etc.)
        self.ustep=np.unique(self.step)
        self.ueoq=np.unique(self.eoq)
        self.uelev=np.unique(self.elev)
        self.uazim=np.unique(self.azim)
        self.uazim_rad=np.unique(self.azim_rad)
        self.uepoch,self.inverse=np.unique(self.epoch,return_inverse=True)
        self.uepoch_seconds = epoch_to_unixseconds(self.uepoch)
        self.utime = np.array(unixseconds_to_date(self.uepoch_seconds))
        
        self.udoy = [(t - dt.datetime(t.year, 1, 1)).days + 1 + t.hour / 24 + t.minute / 1440 + t.second / 86400 + t.microsecond / 86400000000 for t in self.utime]
        self.udoy = np.array(self.udoy)
        self.doy=self.udoy[self.inverse]
        
        #derive data quantities for He2+ (=alphas)
        self.pr_helium=5#priority range for helium (includes He2+ (=main species) and He1+)
        self.steprange=np.arange(0,64,1)
        self.uvels_He2=self.calc_vel(self.steprange,m=4,q=2)
        self.vels_He2=self.calc_vel(self.step,m=4,q=2)
        
        self.umeanspeeds=None
        self.umeanspeeds_uncor=None    
        
        if load_prates==True:
            self.rdata=HIS_Rates(date=date, inpath=inpath)
            self.repoch=self.rdata.epoch #should alwys be equal to unique(self.epoch)!
            self.prange=self.rdata.prange
            self.rates=self.rdata.rates
            self.rates_alpha=self.rdata.rates_alpha
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
        Method to get He2+ with hardcoded Box selection in the ET-Matrix to choose He2+. 
        """
    
        C = []
        matrices = []
        dateipfad = 'he2+_boxes.txt'
        for j in range(8):
            for k in range(16):
                print(k)
                vels = []
                doy = []
                for i in range(63):
                    with open(dateipfad, 'r') as datei:
                        for index, zeile in enumerate(datei):
                            if index == i:
                                sq = zeile.strip().split(',')
                                sq = list(map(int,sq))    
                    x = [sq[1], sq[2], sq[2], sq[1],sq[1]]
                    y = [sq[3], sq[3], sq[4], sq[4],sq[3]]
                    mask = (self.pr==j)*(self.elev==self.uelev[k])*(self.step==i)*(self.tof>sq[1])*(self.tof<sq[2])*(self.essd>sq[3])*(self.essd<sq[4])
                    vels.append(self.vels_He2[mask])
                    doy.append(self.doy[mask])
                vels = np.hstack(vels)
                doy = np.hstack(doy)
                bins_doy=ap(self.udoy, self.udoy[-1]+(self.udoy[-1]-self.udoy[-2]))
                bins_vels=ap(self.uvels_He2, self.uvels_He2[-1]+(self.uvels_He2[-1]-self.uvels_He2[-2])) 
                pr_rates = self.rates[:,j,:,k]   
                Cv,tb,vb=np.histogram2d(doy, vels,bins=[bins_doy,bins_vels])
                
                Cv = Cv*pr_rates
                matrices.append(Cv)
            
            C_pr = np.sum(matrices,axis=0)
            C.append(C_pr)  
         
        C = np.sum(C,axis=0)
        
        
        mean_speed = np.sum(C*vb[:-1], axis=1)/np.sum(C, axis=1)
        sum_counts = np.sum(C, axis=1)
        thermal_speed = []

        for j in range(len(self.uepoch)):
            th_speed = (np.sum(C[j]*(mean_speed[j]-vb[:-1])**2) / np.sum(C[j]))**0.5
            thermal_speed.append(th_speed)
        
        if save_to_file==True:
            filename = self.date_str
            np.savetxt('his_timeseries/'+filename+'.txt', np.column_stack((tb[:-1], sum_counts, mean_speed, thermal_speed)), delimiter=' ')  
        return C,tb,vb
    
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