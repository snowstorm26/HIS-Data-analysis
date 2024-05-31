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
    def get_EThist(self,uepoch_ind="all",step_select=30,pr_select=None,wgts=None,tofrange=np.arange(0,150,2),Erange=np.arange(1,300,2),plot_ET=True,CBlog=True): 
        """
        Method to get ET-matrix for a given Epq-steps and priority range. 
        Input: If uepoch_ind="all" is selected, the ET-matrix histogram is accumulated over the whole day (=file content). Otherwise the histogram is accumulated over time stamp of the given index uepoch_ind=i. 
        """
        
        if pr_select!=None:
            if uepoch_ind=="all":
                m=(self.step==step_select)*(self.pr==pr_select)
                time_min, time_max = str(self.utime[0])[:-7], str(self.utime[-1])[:-7]
            else:
                m=(self.epoch==self.uepoch[uepoch_ind])*(self.step==step_select)*(self.pr==pr_select)  
                time_min, time_max = str(self.utime[uepoch_ind])[:-7], str(self.utime[uepoch_ind])[:-7]
        else:
            if uepoch_ind=="all":
                m=(self.step==step_select)
                #doy_min, doy_max = self.udoy[0], self.udoy[-1]
                time_min, time_max = str(self.utime[0])[:-7], str(self.utime[-1])[:-7]
            else:
                m=(self.epoch==self.uepoch[uepoch_ind])*(self.step==step_select)
                time_min, time_max = str(self.utime[uepoch_ind])[:-7], str(self.utime[uepoch_ind])[:-7]
        C,tb,eb=np.histogram2d(self.tof[m],self.essd[m],[tofrange,Erange])
        #Todo: Add base rate correction by adding weights to the histogram:
        #hcor=np.histogram2d(self.tof[m],self.essd[m],[tofrange,Erange],weights=self.wgts[m])
        
        #plot PHA ET-matrix (uncorrected for base rates)
        if plot_ET==True:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
            my_cmap.set_under('w')
            if CBlog==True:
                #return h[0],h[1],h[2],
                Cont1=ax.pcolormesh(tb,eb,C.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(C))))
            else:
                Cont1=ax.pcolormesh(hcor[1],hcor[2],hcor[0].T,cmap=my_cmap, vmin=1,vmax=max(np.ravel(h[0])))
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("counts per bin")	
            if pr_select==None:
                ax.set_title("ET-matrix, uncorr. PHA, E/q-step: %i\n time range: [%s, %s]"%(step_select, time_min, time_max))
                ax.set_title("E/q-step: %i"%(step_select)) 
            else:
                # ax.set_title("ET-matrix, uncorr. PHA, E/q-step: %i, PR: %i\n time range: [%s, %s]"%(step_select, pr_select, time_min, time_max)) 
                ax.set_title("E/q-step: %i, Priority range: %i"%(step_select, pr_select))                 
            ax.set_xlabel("TOF [ch]")
            ax.set_ylabel("ESSD [ch]")
            fig.savefig("pha_ET_matrix_alpha.png", dpi=400)
        return tb,eb,C
    def calc_alpha_counthist_elevdep(self, elev_ind=0, plot_hist=False, yax_prod="speed", pr_cor=False):
        """
        Method to calculate a time series of speed distributons in a fast way by using 2D-histograms. The x-axis of the histogram is the timestamp, the y-axis can be chosen to be the Epq-step or speed (see below).
        The distribution counts are included in the resulting Cv or Cs histograms, respectively. The histogram can be plotted for better understanding.   
        
        Input: yax_prod can be either "Epq-step" or "speed"
        
        """
        mask=(self.elev==self.uelev[elev_ind])*(self.pr==self.pr_helium)
        #prmask=(self.pr==self.pr_helium)#Todo: cut He1+ from the data! 
        step_He2=self.step[mask]
        vels_He2=self.vels_He2[mask]
        doy_He2=self.doy[mask]
        bins_doy=ap(self.udoy, self.udoy[-1]+(self.udoy[-1]-self.udoy[-2]))
        bins_vel=ap(self.uvels_He2, self.uvels_He2[-1]+(self.uvels_He2[-1]-self.uvels_He2[-2]))
                
        bins_step=ap(self.steprange, self.steprange[-1]+(self.steprange[-1]-self.steprange[-2]))
        Cs,tb,sb=np.histogram2d(doy_He2,step_He2, [bins_doy, bins_step])
        Cv,tb,vb=np.histogram2d(doy_He2,vels_He2, [bins_doy, bins_vel])
        
        #spalten_summen = np.sum(Cv, axis=0)
        #print(spalten_summen.shape)
        #fig, ax1 = plt.subplots(1,1)
        
        #ax1.plot(self.udoy,spalten_summen)
        if pr_cor==True:
            pr_weights=self.rates_alpha[:,:,elev_ind]
            Cs=Cs*pr_weights         
            Cv=Cv*pr_weights 
        
        
        if plot_hist==True:
            fig, ax = plt.subplots(1,1)
            my_cmap.set_under('w')
            if yax_prod=="Epq-step":
                Cont1=ax.pcolormesh(tb,sb,Cs.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1.,vmax=max(np.ravel(Cs))))
                ax.set_ylabel("Epq-step")
            elif yax_prod=="speed":
                Cont1=ax.pcolormesh(tb,vb,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1.,vmax=max(np.ravel(Cv))))
                ax.set_ylabel("speed [km/s]")
            else:
                print("invalid yax_prod")
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("PHA counts per bin")	
            ax.set_title("uncorr. %s-dist for He2+, elev_ind=%i"%(yax_prod,elev_ind))
            ax.set_xlabel("time [DOY 2022]")
            
        return Cv,tb,vb
    def calc_alpha_counthist(self, pr_cor=True, plot_hist=False):
        # can not use this for speed calculation because angles are not included
        Cv=np.zeros((len(self.uelev), len(self.uepoch), len(self.uvels_He2)))
        i=0
        while i<len(self.uelev):
          #print(i, self.uelev[i])
          Cv_elev,tb,vb = self.calc_alpha_counthist_elevdep(elev_ind=i, plot_hist=False, yax_prod="speed", pr_cor=pr_cor)  
          Cv[i]=Cv_elev
          i=i+1
        Cv=np.sum(Cv,axis=0)

        if plot_hist==True:
            fig, ax = plt.subplots(1,1)
            my_cmap.set_under('w')
            Cont1=ax.pcolormesh(tb,vb,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cv))))
            #ax.set_xlim(205.76,205.79)
            ax.set_ylabel("speed [km/s]")
            ax.set_xlabel("time [DOY 2022]")
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("PHA counts per bin")	
            if pr_cor==True:
                ax.set_title("pr-corr. speed-dist. for He2+")
            else:
                ax.set_title("uncorr. speed-dist. for He2+")
        return Cv,tb,vb
    def gaussian(self, x, std_dev, mean, ampl):
        """
        Gaussian function for fitting the speed distribution above the edge.
        """
        return ampl*np.exp(-(x-mean)**2/(2*std_dev**2))
    def calc_mean_speed(self, step_ind=29, pr_cor=True, save_to_file=False):
        """  
        This function calculates a gaussian fit for every measurement. Additionally it checks how good the fit is and marks the data with a quaity flag to determine wheter the fit is useful. It also checks wheter there is certain amount of counts and it calculates the mean speed and thermal speed before and after the fit. In the end everything is saved to a file for further processing. 
        """   
        Cv,tb,vb = self.calc_alpha_counthist(pr_cor=pr_cor, plot_hist=False)
        mean_speed_edge_uncorr = np.sum(Cv * vb[:-1], axis=1) / np.sum(Cv, axis=1)
        mean_speed = np.full_like(mean_speed_edge_uncorr, np.nan)

        
        quality_flag = np.zeros(len(self.uepoch)) #0: everything ok (no fit or fit successfull),  1: fit not useful (650 - 700 km/s) or reduced chi^2 to big, 2: less than 10000 counts, 3: relative uncertainty of fit bigger then 1%:
        chi_list = np.full_like(quality_flag, np.nan)
        rel_sigmas =  np.full_like(quality_flag, np.nan)

        counts = np.full_like(quality_flag, np.nan)
        thermal_speed = np.full_like(quality_flag, np.nan)
        sum_counts = np.full_like(quality_flag, np.nan)
        
        unc_sigma_list = np.full_like(quality_flag, np.nan)
        unc_mu_list = np.full_like(quality_flag, np.nan)
        unc_ampl_list = np.full_like(quality_flag, np.nan)
        

        slope_fit_list = np.full_like(quality_flag, np.nan)
        th_speed_list = []
        #max for uepoch_ind is the value of the length of sel.uepoch maximum 2880 , it stands for 1 measurement period of 30 seconds
        for ind in range(len(self.uepoch)):
            th_speed = np.sum(Cv[ind]*(mean_speed_edge_uncorr[ind]-vb[:-1])**2) / np.sum(Cv[ind])
            th_speed = th_speed**0.5
            th_speed_list.append(th_speed)
            
            
            v_dist = 1.0*Cv[ind, :]
            v_dist_fit = 1.0*Cv[ind, :]
            
            summe = np.sum(v_dist) 
            sum_counts[ind] = summe
            
            
            if np.sum(v_dist) < 10000:
                quality_flag[ind] = 1
                
            else:
                max_ind = np.argmax(v_dist[step_ind:]) + step_ind
                mean = self.uvels_He2[max_ind]
                ampl = v_dist[max_ind]
                slope_fit = v_dist[step_ind+1]-v_dist[step_ind]
                if slope_fit >0: 
                    try:
                        params, covariance = curve_fit(self.gaussian, self.uvels_He2[step_ind:max_ind+12], v_dist[step_ind:max_ind+12], p0=[100, mean, ampl])
                        counts[ind] = params[2]
                        thermal_speed[ind] = params[0]
                        mean_speed[ind] = params[1]
                        
                        #Handling the uncertanties of the covariance matrix with square function
                        unc_sigma, unc_mu, unc_ampl = np.sqrt(np.diag(covariance))
                        
                        
                        unc_sigma_list[ind] = unc_sigma
                        unc_mu_list[ind] = unc_mu
                        unc_ampl_list[ind] = unc_ampl
                        
                        #Calculating the reduced chi squared of a successfull fit
                        v_dist_fit= self.gaussian(self.uvels_He2, params[0], params[1], params[2])
                        #Setting all 0 to 1 so that no division by 0 occurs
                        v_dist[v_dist==0] = 1
                        v_dist_fit[v_dist_fit == 0] = 1
                        o = v_dist
                        e = v_dist_fit
                        
                        chi_squared = sum((o - e) ** 2/e)
                        nu = 3
                        N = max_ind+12-step_ind
                        reduced_chi = chi_squared / (N - nu)
                        chi_list[ind] = reduced_chi
                    except RuntimeError:
                        params = [-1, -1, -1]
                        print("RuntimeError from gaussian fit...")
                        quality_flag[ind] = 2
                    '''
                    Cv[ind, :step_ind] = self.gaussian(self.uvels_He2[:step_ind], params[0], params[1], params[2])
                    mean_speed[ind] = np.sum(Cv[ind,:] * vb[:-1])/np.sum(Cv[ind,:])
                    '''
                else:
                    slope_fit_list[ind] = slope_fit
                    quality_flag[ind] = 2


        if save_to_file==True:
            filename = self.date_str
            np.savetxt('his_timeseries/'+filename+'.txt', np.column_stack((self.udoy, sum_counts, mean_speed_edge_uncorr, th_speed_list,counts, unc_sigma_list, mean_speed, unc_mu_list, thermal_speed, unc_ampl_list, chi_list, quality_flag)), delimiter=' ')

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
