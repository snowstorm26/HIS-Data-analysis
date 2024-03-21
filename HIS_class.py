import cdflib

import numpy as np
from numpy import append as ap
import scipy as sp
import datetime as dt

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
    def __init__(self,date=(2022,9,16),version="V02",inpath="Data/", load_prates=False):
        
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
            self.rdata=HIS_Rates(date=date)
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

    
    def calc_alpha_speeddist(self, uepoch_ind=0, steps_select=np.arange(0,64,1), plot_speeddist=False):
        """
        method to plot a single alpha (=He2+) speed (=abs. value of velocity) distribution for a given time stamp. 
        
        Remark: At the moment the distributions are not corrected for base rates, phase space coverage of the sensor etc, and also contain He+ counts (probably negligible within the thermal core of the distribution).   
        """
        steps=steps_select
        vels=self.calc_vel(steps,m=4.,q=2.)
        pr_helium=5#TODO: cut out He1+ from priority range
        
        #get He2+ counts 
        counts=np.zeros((len(steps_select)))
        for i,step in enumerate(steps):
            tb,eb,C=self.get_EThist(uepoch_ind=uepoch_ind, step_select=step,pr_select=pr_helium,wgts=None,tofrange=np.arange(0,150,2),Erange=np.arange(1,300,2),plot_ET=False,CBlog=True)
            counts[i]=np.sum(C)   
        
        #plot speed distribution
        if plot_speeddist==True:
            plt.figure()
            plt.plot(vels,counts, label="He2+")
            plt.xlabel("speed [km/s]")
            plt.ylabel("counts")
            plt.legend()
        return steps, vels, counts
   


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
          print(i, self.uelev[i])
          Cv_elev,tb,vb = self.calc_alpha_counthist_elevdep(elev_ind=i, plot_hist=False, yax_prod="speed", pr_cor=pr_cor)  
          Cv[i]=Cv_elev
          i=i+1
        Cv=np.sum(Cv,axis=0)

        if plot_hist==True:
            fig, ax = plt.subplots(1,1)
            my_cmap.set_under('w')
            Cont1=ax.pcolormesh(tb,vb,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cv))))
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
    
    def calc_mean_speed(self, step_ind=29, pr_cor=True, save_to_file=False, plot_hist=False):
        Cv,tb,vb = self.calc_alpha_counthist(pr_cor=pr_cor, plot_hist=False)
        mean_speed_edge_uncorr = np.sum(Cv * vb[:-1], axis=1) / np.sum(Cv, axis=1)
        mean_speed = np.zeros(len(self.uepoch))
        quality_flag = np.zeros(len(self.uepoch)) # 0: v < 650 km/s no fit, 1: v > 700 km/s gauss-fit, 2: 650 < v < 700 untrustworthy
        
        # TODO: optimize the range where the fit method is used
        for ind in range(len(self.uepoch)):
            if mean_speed_edge_uncorr[ind] < 650:
                mean_speed[ind] = mean_speed_edge_uncorr[ind]
                quality_flag[ind] = 0
            
            elif mean_speed_edge_uncorr[ind] > 700:
                v_dist = 1.0*Cv[ind, :]
                max_ind = np.argmax(v_dist[step_ind:]) + step_ind
                mean = self.uvels_He2[max_ind]
                ampl = v_dist[max_ind]
                try:
                    # TODO: look covariance matrix for accuracy of fit parameters, especially width of distribution (sigma)
                    params, covariance = curve_fit(self.gaussian, self.uvels_He2[step_ind:max_ind+12], v_dist[step_ind:max_ind+12], p0=[100, mean, ampl])
                except RuntimeError:
                    params = [-1, -1, -1]
                    print("RuntimeError from gaussian fit...")
                Cv[ind, :step_ind] = self.gaussian(self.uvels_He2[:step_ind], params[0], params[1], params[2])
                mean_speed[ind] = np.sum(Cv[ind,:] * vb[:-1])/np.sum(Cv[ind,:])
                # print("gauss mean speed :", mean_speed[ind])
                quality_flag[ind] = 1
            
            else:
                mean_speed[ind] = mean_speed_edge_uncorr[ind]
                quality_flag[ind] = 2
        
        if save_to_file==True:
            filename = self.date_str
            np.savetxt('speed data 2022/'+filename+'.txt', np.column_stack((self.udoy, mean_speed, quality_flag)), delimiter=' ')
        
        if plot_hist==True:
            fig, ax = plt.subplots(1,1, figsize=(6,4))
            my_cmap.set_under('w')
            Cont1=ax.pcolormesh(tb,vb,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cv)))) # 5299579.02696228))
            ax.set_ylabel("speed [km/s]")
            ax.set_xlabel("time [DOY 2022]")
            cb1 = fig.colorbar(Cont1)
            cb1.set_label(r"He$^{2+}$ counts per bin")
            
        return Cv,tb,vb
        
    def calc_alpha_counthist_step(self, step_ind=29, rescale_simple=False, rescale_fit_mirror=True, pr_cor=True, uepoch_ind=0, plot_1D_cut=False, plot_diff=False, plot_hist=False, return_weights=False, save_speed=False):
        Cv,tb,vb = self.calc_alpha_counthist(pr_cor=pr_cor, plot_hist=False)
        
        mean_speed_edge_uncorr = np.sum(Cv * vb[:-1], axis=1) / np.sum(Cv, axis=1)
        thermal_velocity = np.zeros_like(mean_speed_edge_uncorr)
        
            

        
        if rescale_fit_mirror==True:
            for ind in range(len(self.uepoch)):
                v_dist = 1.0*Cv[ind, :]
                max_ind = np.argmax(v_dist[step_ind:]) + step_ind
                mean = self.uvels_He2[max_ind]
                ampl = v_dist[max_ind]
                
                try:
                    params, covariance = curve_fit(self.gaussian, self.uvels_He2[step_ind:max_ind+12], v_dist[step_ind:max_ind+12], p0=[100, mean, ampl])
                except RuntimeError:
                    params = [-1, -1, -1]
                    
                # Cv[ind, :step_ind] = self.gaussian(self.uvels_He2[:step_ind], params[0], params[1], params[2])
                thermal_velocity[ind] = params[0]
                
                if ind==uepoch_ind:
                    if plot_1D_cut==True:
                        # good 1130 (120), working 2300 (1850), bad 2136 (2000)
                        vel = np.linspace(self.uvels_He2[0], self.uvels_He2[-1], num=1000)
                        fig, ax = plt.subplots(1,1, figsize=(3.5,5))
                        
                        
                        
                        ax.plot(self.uvels_He2[step_ind:], v_dist[step_ind:], 'o', color='tab:red', label='above edge', zorder=4)
                        # ax.scatter(self.uvels_He2[:step_ind], v_dist[:step_ind], marker='o', facecolors='none', edgecolors='tab:red', label='below edge', zorder=3)
                        # ax.plot(vel, self.gaussian(vel, params[0], params[1], params[2]), color='black', label='gaussian fit:\n'+r'$\mu$=%i'%(params[1])+r' km/s'+'\n'+ '$\sigma$=%i'%(params[0])+' km/s', zorder=1)
                        # ax.plot(self.uvels_He2[:step_ind], Cv[uepoch_ind,:step_ind], 'o', color='grey', label='values from fit', zorder=2)
                        

                        handles, labels = ax.get_legend_handles_labels()
                        # order = [0, 2, 1]#, 2] # Adjust this list to match your desired order
                        # order = [0, 1]
                        order = [0, 3,1,2]
                        # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right", ncol=1,fontsize='7')#), bbox_to_anchor)


                        # ax.plot(self.uvels_He2[step_ind-1], v_dist[step_ind-1], marker='o', linestyle='None', color='red', label='value just below edge')
                        # ax.set_xlim(400, 1300)
                        # ax.set_xticks([400, 600, 800, 1000, 1200])
                        # ax.set_ylim(-1e5, 3e6)
                        # ax.set_yticks([0, 1e6, 2e6, 3e6])

                        ax.set_xlabel("speed [km/s]")
                        ax.set_ylabel(r"He$^{2+}$ counts")
                        
                        # ax2 = ax.twiny()
                        # ax2.set_xlim(20, 43)
                        # ax2.set_xticks([20, 25, 30, 35, 40])
                        # ax2.set_xlabel('E/q-step')
                        # ax.yaxis.get_offset_text().set(x=-0.15, ha='left')
                        # ax.plot(self.uvels_He2, v_dist, 'o', color='tab:red')
                        
                        # ax.legend(loc="upper right",fontsize='7')
                        # ax.grid(True)
                        # ax.set_title(f"pr-corr. speed-dist. for He2+, DOY={self.udoy[uepoch_ind]:.5f}")
                        fig.savefig('fit_plot.png', dpi=400)
                
        # if plot_1D_cut==True:
        #     fig, ax = plt.subplots(1,1)
        #     ax.plot(self.uvels_He2, Cv[uepoch_ind,:], 'o')
        #     ax.plot(self.uvels_He2[step_ind-1], Cv[uepoch_ind,step_ind-1], marker='o', color='red')
        #     ax.set_xlabel("speed [km/s]")
        #     ax.set_ylabel("PHA counts")
        #     ax.grid(True)
        #     if pr_cor==True:
        #         ax.set_title(f"pr-corr. speed-dist. for He2+, DOY={self.udoy[uepoch_ind]:.5f}")
        #     else:
        #         ax.set_title(f"uncorr. speed-dist. for He2+, DOY={self.udoy[uepoch_ind]:.5f}")     
                
       
        mean_speed = np.sum(Cv * vb[:-1], axis=1) / np.sum(Cv, axis=1)

        print("Where fit doesn't work:", np.where(thermal_velocity==-1))
        print(max(np.ravel(Cv)))
        if plot_hist==True:
            fig, ax = plt.subplots(1,1, figsize=(6,4))
            my_cmap.set_under('w')
            Cont1=ax.pcolormesh(tb,vb,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cv)))) # 5299579.02696228))#
            # ax.plot(self.udoy, mean_speed_edge_uncorr, color='black', label='mean speed edge uncorr.', alpha=0.8)
            # ax.plot(self.udoy, mean_speed_edge_uncorr, color='black', label='mean speed')
            # ax.plot(self.udoy, mean_speed, color='darkviolet', label=r'mean speed edge corr.', alpha=0.8) #' $\pm$ thermal velocity', alpha=0.8)
            # ax.plot(self.udoy, mean_speed+thermal_velocity, ':', color='m', lw=0.8, alpha=0.8)
            # ax.plot(self.udoy, mean_speed-thermal_velocity, ':', color='m', lw=0.8, alpha=0.8)
            ax.set_ylabel("speed [km/s]")
            ax.set_xlabel("time [DOY 2022]")
            cb1 = fig.colorbar(Cont1)
            cb1.set_label(r"He$^{2+}$ counts per bin")
            # ax.legend(loc="upper right",fontsize='7')
            if pr_cor==True:
                if rescale_fit_mirror==True:
                    # ax.set_title("pr-corr. speed-dist. for He2+, edge corr.")
                    fig.savefig('speed_dist_edge_corrected.png', dpi=400)
                else:
                    # ax.set_title("pr-corr. speed-dist. for He2+, edge uncorr.")
                    fig.savefig('speed_dist_edge_uncorrected.png', dpi=400)
            else:
                ax.set_title("uncorr. speed-dist. for He2+")
        print("mean speed larger:", np.where(mean_speed>mean_speed_edge_uncorr))
        
        if return_weights==True:
            return rescaling_factor
        
        if save_speed==True:
            filename = self.date_str
            np.savetxt('speed data 2022/'+filename+'.txt', np.column_stack((self.udoy, mean_speed, thermal_velocity)), delimiter=' ') #, header='time,<|v|>, v_th\ndoy km/s km/s\n', comments='')
            
        
        
        return Cv,tb,vb
            

    def calc_alpha_speed_fast(self, pr_cor=True):
        """
        Method to calculate a time series of alpha mean speed by using the (fast) "calc_alpha_counthist" method for all time stamps together. The methods calculates the mean speed and the speed of maximum counts (as sanity check).
        """
        meanvels=np.zeros((len(self.uepoch)))-1
        maxvels=np.zeros((len(self.uepoch)))-1
        counts, tb, vb = self.calc_alpha_counthist(pr_cor=pr_cor, plot_hist=False)
        i=0
        while i < len(self.uepoch):        
            meanvels[i]=np.average(self.uvels_He2,weights=counts[i],axis=0)
            maxvels[i]=self.uvels_He2[counts[i]==max(counts[i])][0]
            print(i, self.utime[i])
            i=i+1
        if pr_cor==True:
            self.umeanspeeds=meanvels
            self.umaxspeeds=maxvels
        elif pr_cor==False:
            self.umeanspeeds_uncor=meanvels
            self.umaxspeeds_uncor=maxvels
        

    def plot_alpha_meanspeed(self, plot_meanspeeds_uncor=False):
        fig, ax = plt.subplots(1,1)
        if plot_meanspeeds_uncor==True:
            ax.plot(self.udoy, self.umeanspeeds_uncor,color="r",linewidth=2, label="|v_uncor|")
            ax.plot(self.udoy, self.umeanspeeds,color="k",linewidth=2,label="|v_prcor|")                    
        else:
            ax.plot(self.udoy, self.umeanspeeds,color="k",linewidth=2,label="|v|")
        ax.set_ylabel("speed [km/s]")
        ax.set_xlabel("time [DOY 2022]")
        ax.legend()
        ax.set_title("mean speed for He2+")
        

    
    
    def plot_alpha_vdf(self, uepoch_ind=0, uelev_ind=9, vsw_min=100, pr_cor=False, filename='plot_alpha_vdf', save_fig=False, return_azim_speed_hist=False, return_hist=False, same_x_y_bins=False, plot_B_field=False):
        """
        Method to plot slices (in radial velocity, azimuth angle) of the 3D-VDF for alphas (He2+) for a given time stamp. The elevation angle can be selected via elev_select. histcoord representation should be "polar" for the standard plots. The phi-binning should be chosen fine enough to resolve the distribution well.
        
        Remark: vsw_min is used to filter "unphysical counts" below a certain threshold. 
        """
        prc_label="pr_uncor"
        m=(self.epoch==self.uepoch[uepoch_ind])*(self.elev==self.uelev[uelev_ind])*(self.pr==self.pr_helium)
        epoch_he2=self.epoch[m]
        step_he2=self.step[m]
        elev_he2=self.elev[m]        
        vx_He2=np.cos(self.azim_rad[m])*self.vels_He2[m]
        vy_He2=np.sin(self.azim_rad[m])*self.vels_He2[m]
        bins_vels_He2=ap(self.uvels_He2, self.uvels_He2[-1]+(self.uvels_He2[-1]-self.uvels_He2[-2]))
        print(bins_vels_He2)
        bins_azim=ap(self.uazim, self.uazim[-1]+(self.uazim[-1]-self.uazim[-2]))
        Cs,sb,ab=np.histogram2d(self.vels_He2[m],self.azim[m],[bins_vels_He2,bins_azim])        
        
        # print(step_he2[:10], vx_He2[:10])
        if pr_cor==True:
            prc_label="pr_cor"
            pr_weights=self.rates_alpha[uepoch_ind,:,uelev_ind]
            pr_weights_vdfslice=np.array([pr_weights]*len(self.uazim))
            Cs=Cs*pr_weights_vdfslice.T
            weights_v=np.zeros((len(vx_He2)))
            for i,v in enumerate(vx_He2):
                ind_epoch=np.where(self.uepoch==epoch_he2[i])[0][0]
                ind_step=np.where(self.ustep==step_he2[i])[0][0]
                ind_elev=np.where(self.uelev==elev_he2[i])[0][0]
                weights_v[i]=self.rates_alpha[ind_epoch, ind_step, ind_elev]
                # print(i, ind_epoch, ind_step, ind_elev, weights_v[i])
            if same_x_y_bins==True:
                x_bins = np.arange(200, 1100, 30)
                y_bins = np.arange(-700, 800, 30)
            else:
                x_bins = np.arange(min(vx_He2),max(vx_He2),max(vx_He2)/50.)
                y_bins = np.arange(min(vy_He2),max(vy_He2),max(vy_He2/50.))
            Cv,vxb,vyb=np.histogram2d(vx_He2,vy_He2,[x_bins,y_bins], weights=weights_v)
            # print(Cv, vxb, vyb, x_bins, y_bins)
        else:
            Cv,vxb,vyb=np.histogram2d(vx_He2,vy_He2,[np.arange(min(vx_He2),max(vx_He2),max(vx_He2)/50.),np.arange(min(vy_He2),max(vy_He2),max(vy_He2/50.))])                
        
        vxb_mean = vxb[:-1]+(vxb[1:]-vxb[:-1])
        vyb_mean = vyb[:-1]+(vyb[1:]-vyb[:-1])
        vyb_grid, vxb_grid=np.meshgrid(vyb_mean,vxb_mean) 
        vxbg,vybg = np.ravel(vxb_grid), np.ravel(vyb_grid)
        Cvg=np.ravel(Cv)
        outl_mask=(vxbg>vsw_min)        
        vx_mean=np.average(vxbg[outl_mask],weights=Cvg[outl_mask])
        vx_var = np.average((vxbg[outl_mask]-vx_mean)**2, weights=Cvg[outl_mask])
        vx_th=np.sqrt(vx_var)
        vy_mean=np.average(vybg[outl_mask],weights=Cvg[outl_mask])
        vy_var = np.average((vybg[outl_mask]-vy_mean)**2, weights=Cvg[outl_mask])
        vy_th=np.sqrt(vy_var)
        # print("mean velocity of elev_slice:", (vx_mean,vy_mean))        
        # print("thermal velocity of elev_slice:", (vx_th,vy_th)) 
    
        if return_hist==True:
            return Cv, vxb, vyb, vx_mean, vy_mean
        
        if return_azim_speed_hist==True:
            return Cs, sb, ab
        #v in instrument frame, in (|v|, azim) coords
        fig, ax = plt.subplots(1,1)
        Cont1=ax.pcolormesh(sb,ab,Cs.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cs))))
        cb1 = fig.colorbar(Cont1)
        cb1.set_label("PHA counts per bin")	
        ax.set_xlim(0, 1500)
        
        ax.set_title("He2+ VDF elevation-slice in instr. frame, (|v|, azim)-coords\n DOY: %.5f , elev: %.2f deg, %s"%(self.udoy[uepoch_ind], self.uelev[uelev_ind],prc_label))
        ax.set_xlabel("|v| [km/s]")
        ax.set_ylabel("azim. angle [deg]")
        fig.savefig("velocity_azimuth_angle_fast.png", dpi=400)
        
        
        #v in instrument frame
        fig, ax = plt.subplots(1,1)
        Cont1=ax.pcolormesh(vxb,vyb,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cv))))
        cb1 = fig.colorbar(Cont1)
        cb1.set_label("PHA counts per bin")	
        ax.set_title("He2+ VDF elevation-slice in instr. frame\n DOY: %.5f , elev: %.2f deg, %s"%(self.udoy[uepoch_ind], self.uelev[uelev_ind], prc_label))
        ax.set_xlabel("v_r [km/s]")
        ax.set_ylabel("v_t [km/s]")
        ax.set_xlim(100, 1100)
        ax.set_xticks([200, 400, 600, 800, 1000])
        ax.set_ylim(-700, 800)
        ax.set_yticks([-600, -400, -200, 0, 200, 400, 600])
        plt.plot([vx_mean],[vy_mean],marker="o", linestyle='none', markersize=10,color="m", label="mean velocity in elev. slice: (%i km/s, %i km/s)"%(vx_mean,vy_mean))
        plt.legend(loc="upper right",fontsize='small')
        if save_fig==True:
            plt.savefig(filename+'.png', dpi=400)
            plt.close()           
            
        if plot_B_field==True:
            fig, ax = plt.subplots(1,1)
            Cont1=ax.pcolormesh(vxb-vx_mean,vyb-vy_mean,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cv))))
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("PHA counts per bin")	
            ax.set_title("He2+ VDF elevation-slice in instr. frame\n DOY: %.5f , elev: %.2f deg, %s"%(self.udoy[uepoch_ind], self.uelev[uelev_ind], prc_label))
            ax.set_xlabel("v_r [km/s]")
            ax.set_ylabel("v_t [km/s]")
            # ax.set_xlim(100, 1100)
            # ax.set_xticks([200, 400, 600, 800, 1000])
            # ax.set_ylim(-700, 800)
            # ax.set_yticks([-600, -400, -200, 0, 200, 400, 600])
            ax.quiver(0, 0, self.Br_ts[0], self.Bt_ts[0], angles='xy', scale_units='xy', scale=1, color='black', label='B-field')
            plt.axis('equal')
            plt.plot([vx_mean],[vy_mean],marker="o", linestyle='none', markersize=10,color="m", label="mean velocity in elev. slice: (%i km/s, %i km/s)"%(vx_mean,vy_mean))
            plt.legend(loc="upper right",fontsize='small')
            if save_fig==True:
                plt.savefig(filename+'.png', dpi=400)
                plt.close()   
    
    def plot_alpha_vdf_summed(self, uepoch_ind_array=np.arange(0,10), uelev_ind=9, vsw_min=100, pr_cor=True, filename='plot_alpha_vdf', save_fig=False, plot_B_field=False):
        """
        Method to plot slices (in radial velocity, azimuth angle) of the 3D-VDF for alphas (He2+) summed over array of times. The elevation angle can be selected via elev_select. histcoord representation should be "polar" for the standard plots. The phi-binning should be chosen fine enough to resolve the distribution well.
        
        Remark: vsw_min is used to filter "unphysical counts" below a certain threshold. 
        """
        Cv0, vxb, vyb, x_bins, y_bins = self.plot_alpha_vdf(uepoch_ind=0, uelev_ind=7, vsw_min=100, pr_cor=True, filename='plot_alpha_vdf', save_fig=False, return_hist=True, same_x_y_bins=True)
        Cs_add = np.zeros_like(Cv0)
        Cv_sum = np.zeros_like(Cv0)
        prc_label="pr_uncor"
        for uepoch_ind in uepoch_ind_array:
            prc_label="pr_uncor"
            m=(self.epoch==self.uepoch[uepoch_ind])*(self.elev==self.uelev[uelev_ind])*(self.pr==self.pr_helium)
            epoch_he2=self.epoch[m]
            step_he2=self.step[m]
            elev_he2=self.elev[m]        
            vx_He2=np.cos(self.azim_rad[m])*self.vels_He2[m]
            vy_He2=np.sin(self.azim_rad[m])*self.vels_He2[m]  
            bins_vels_He2=ap(self.uvels_He2, self.uvels_He2[-1]+(self.uvels_He2[-1]-self.uvels_He2[-2]))
            bins_azim=ap(self.uazim, self.uazim[-1]+(self.uazim[-1]-self.uazim[-2]))
            
            Cs,sb,ab=np.histogram2d(self.vels_He2[m],self.azim[m],[bins_vels_He2,bins_azim])
            if pr_cor==True:
                prc_label="pr_cor"
                pr_weights=self.rates_alpha[uepoch_ind,:,uelev_ind]
                pr_weights_vdfslice=np.array([pr_weights]*len(self.uazim))
                # Cs_add=Cs*pr_weights_vdfslice.T
                if uepoch_ind==0:
                    Cs_add = Cs*pr_weights_vdfslice.T
                else:
                    Cs_add += Cs*pr_weights_vdfslice.T
                    
                    # print(Cs_add)
                weights_v=np.zeros((len(vx_He2)))
                for i,v in enumerate(vx_He2):
                    ind_epoch=np.where(self.uepoch==epoch_he2[i])[0][0]
                    ind_step=np.where(self.ustep==step_he2[i])[0][0]
                    ind_elev=np.where(self.uelev==elev_he2[i])[0][0]
                    weights_v[i]=self.rates_alpha[ind_epoch, ind_step, ind_elev]
                x_bins = np.arange(200, 1100, 30)
                y_bins = np.arange(-700, 800, 30)
                Cv,vxb,vyb=np.histogram2d(vx_He2,vy_He2,[x_bins,y_bins], weights=weights_v)
            else:
                Cv,vxb,vyb=np.histogram2d(vx_He2,vy_He2,[np.arange(min(vx_He2),max(vx_He2),max(vx_He2)/50.),np.arange(min(vy_He2),max(vy_He2),max(vy_He2/50.))])                
            Cv_sum = Cv_sum + Cv
        Cv = Cv_sum    
        vxb_mean = vxb[:-1]+(vxb[1:]-vxb[:-1])
        vyb_mean = vyb[:-1]+(vyb[1:]-vyb[:-1])
        vyb_grid, vxb_grid=np.meshgrid(vyb_mean,vxb_mean) 
        vxbg,vybg = np.ravel(vxb_grid), np.ravel(vyb_grid)
        Cvg=np.ravel(Cv)
        outl_mask=(vxbg>vsw_min)        
        vx_mean=np.average(vxbg[outl_mask],weights=Cvg[outl_mask])
        vy_mean=np.average(vybg[outl_mask],weights=Cvg[outl_mask])
        
        Cs = Cs_add
        #v in instrument frame, in (|v|, azim) coords
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        Cont1=ax.pcolormesh(sb,ab,Cs.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cs))))
        cb1 = fig.colorbar(Cont1)
        cb1.set_label("PHA counts per bin")	
        ax.set_title("He2+ VDF elevation-slice in instr. frame, (|v|, azim)-coords\n DOY: %.5f , elev: %.2f deg, %s"%(self.udoy[uepoch_ind], self.uelev[uelev_ind],prc_label))
        ax.set_xlabel("|v| [km/s]")
        ax.set_ylabel("azim. angle [deg]")
        ax.set_xlim(0, 1500)
        fig.savefig('HIS_azimuth_speed_hist_50min.png', dpi=400)
        # plt.close()
            
            
            
        #v in instrument frame
        fig, ax = plt.subplots(1,1)
        Cont1=ax.pcolormesh(vxb,vyb,Cv.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cv))))
        cb1 = fig.colorbar(Cont1)
        cb1.set_label("PHA counts per bin")	
        ax.set_title("He2+ VDF elevation-slice in instr. frame\n DOY: %.5f , elev: %.2f deg, %s"%(self.udoy[uepoch_ind_array[0]], self.uelev[uelev_ind], prc_label))
        ax.set_xlabel("v_r [km/s]")
        ax.set_ylabel("v_t [km/s]")
        ax.set_xlim(100, 1100)
        ax.set_xticks([200, 400, 600, 800, 1000])
        ax.set_ylim(-700, 800)
        ax.set_yticks([-600, -400, -200, 0, 200, 400, 600])
        plt.plot([vx_mean],[vy_mean],marker="o", linestyle='none', markersize=10,color="m", label="mean velocity in elev. slice: (%i km/s, %i km/s)"%(vx_mean,vy_mean))
        plt.legend(loc="upper right",fontsize='small')
        # plt.savefig('HIS_vr_vt_hist_5_min_avg_20220916.png', dpi=400)
        if save_fig==True:
            plt.savefig(filename+'.png', dpi=400)
            plt.close()
            

            
    
    def plot_alpha_vdf_animation_elevations(self, uepoch_ind=0):
        folder_name = "elevation_animation_uepoch_ind_"+str(uepoch_ind)
        os.makedirs(folder_name)
        for i in range(len(self.uelev)):
            if i<10:
                end = '0' + str(i)
            else:
                end = str(i)
            self.plot_alpha_vdf(uepoch_ind=uepoch_ind, uelev_ind=i, pr_cor=True, filename=folder_name+"/elev_"+end, save_fig=True)
    
    
    
    def animation_zero_elevation(self, uepoch_ind_start=0, uepoch_ind_end=1000):
        for i in range(uepoch_ind_start, uepoch_ind_end):
            if i<10:
                end = '00'+str(i)
            elif i<100:
                end = '0'+str(i)
            else:
                end = str(i)
            self.plot_alpha_vdf(uepoch_ind=i, uelev_ind=9, pr_cor=True, filename="zero_elevation_animation_whole_day/uepoch_ind_"+end, save_fig=True)
          
            
          
    def animation_zero_elevation_summed(self, num=10):
        for i in range(33, int(np.floor(len(self.uepoch)/num))):
            if i<10:
                end = '00'+str(i)
            elif i<100:
                end = '0'+str(i)
            else:
                end = str(i)
            self.plot_alpha_vdf_summed(uepoch_ind_array=np.arange(num*i, num*i + num), uelev_ind=9, pr_cor=True, filename="zero_elevation_animation_whole_day_summed/uepoch_sum_"+end, save_fig=True)
                      
            
            
    def create_GIF(self, total_duration=15, folder_path='elevation_animation_uepoch_ind_0', gif_filename='elev_anim_uepoch_ind_0.gif', start=0, stop=15):
        images = []
        files = os.listdir(folder_path)
        num_files = len([entry for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))])
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        i = 0
        num_images = 0
        for filename in sorted_files:
            if i>start:
                print(i)
                file_path = os.path.join(folder_path, filename)
                images.append(imageio.imread(file_path))
                num_images += 1
            elif i>stop or i>=num_files:
                break
            i += 1
            
        duration_per_frame = total_duration/num_images
        imageio.mimsave(gif_filename, images, duration=duration_per_frame)
    
    def calc_velocities_temperatures_slice(self, uelev_ind=9, vsw_min=100, save_file=False):
        '''
        Method to calculate the velocity distribution (vr, vt) for one specific elevation slice (uelev_ind)
        Remark: vsw_min is used to filter "unphysical counts" below a certain threshold. 
        '''
        # calculate for whole day for one elevation
        doy = np.array([])
        v_0 = np.array([])
        vr_0 = np.array([])
        vt_0 = np.array([])
        Tr_0 = np.array([])
        Tt_0 = np.array([])
        
        for uepoch_ind in range(0, len(self.uepoch)):
            print('ind: ', uepoch_ind)
            m=(self.epoch==self.uepoch[uepoch_ind])*(self.elev==self.uelev[uelev_ind])*(self.pr==self.pr_helium)
            epoch_he2=self.epoch[m]
            step_he2=self.step[m]
            elev_he2=self.elev[m]
            speed_he2=self.vels_He2[m]
            vx_He2=np.cos(self.azim_rad[m])*self.vels_He2[m]
            vy_He2=np.sin(self.azim_rad[m])*self.vels_He2[m]
                
            
            
            ind_epoch = np.searchsorted(self.uepoch, epoch_he2)
            ind_step = np.searchsorted(self.ustep, step_he2)
            ind_elev = np.searchsorted(self.uelev, elev_he2)
            weights_v = self.rates_alpha[ind_epoch, ind_step, ind_elev]
            
            min_mask = (vx_He2>vsw_min)
            
            speed_mean = np.average(speed_he2[min_mask], weights=weights_v[min_mask])
            vx_mean = np.average(vx_He2[min_mask], weights=weights_v[min_mask])
            vy_mean = np.average(vy_He2[min_mask], weights=weights_v[min_mask])
            vx_var = np.average((vx_He2[min_mask]-vx_mean)**2, weights=weights_v[min_mask])
            vy_var = np.average((vy_He2[min_mask]-vy_mean)**2, weights=weights_v[min_mask])
            vx_th = np.sqrt(vx_var)
            vy_th = np.sqrt(vy_var)
                        
            doy = ap(doy, self.udoy[uepoch_ind])
            vr_0 = ap(vr_0, vx_mean)
            vt_0 = ap(vt_0, vy_mean)
            v_0 = ap(v_0, speed_mean)
            Tr_0 = ap(Tr_0, vx_th)
            Tt_0 = ap(Tt_0, vy_th)
            
            
        m_He2 = 4*1.66e-27#in kg
        to_eV = 6.242e18#eV/J
        Tr_0 = 1/2*m_He2*(Tr_0*1000)**2*to_eV
        Tt_0 = 1/2*m_He2*(Tt_0*1000)**2*to_eV
        
        if save_file==True:
            filename = '2022-09-17_velocities_temperatures_0'
            np.savetxt(filename+'.txt', np.column_stack((doy, v_0, vr_0, vt_0, Tr_0, Tt_0)), delimiter=' ', header='only looking at elevation slice = '+str(self.uelev[uelev_ind])+' deg\ndoy,v_0,vr_0,vt_0,Tr_0,Tt_0\ndoy km/s km/s km/s eV eV\n', comments='')
        
      
    def calc_velocities_temperatures(self, vsw_min=100, save_file=False):
        '''
        Method to calculate the 3D-velocity distribution. This is done by calculating the mean of the components (vr, vt, vn). And then the mean solar wind vector is given by sqrt(vr**2 + vt**2 + vn**2)
        '''
        # calculate for whole day for all elevations
        doy = np.array([])
        v = np.array([])
        vr = np.array([])
        vt = np.array([])
        vn = np.array([])
        Tr = np.array([])
        Tt = np.array([])
        

            
        for uepoch_ind in range(0, len(self.uepoch)):
            print(uepoch_ind)
            
            m=(self.epoch==self.uepoch[uepoch_ind])*(self.pr==self.pr_helium)
            epoch_he2=self.epoch[m]
            step_he2=self.step[m]
            elev_he2=self.elev[m]
            

            vx_He2=np.cos(self.azim_rad[m])*np.cos(self.elev_rad[m])*self.vels_He2[m]
            vy_He2=np.sin(self.azim_rad[m])*np.cos(self.elev_rad[m])*self.vels_He2[m]
            vz_He2=np.sin(self.elev_rad[m])*self.vels_He2[m]
            
            ind_epoch = np.searchsorted(self.uepoch, epoch_he2)
            ind_step = np.searchsorted(self.ustep, step_he2)
            ind_elev = np.searchsorted(self.uelev, elev_he2)
            weights_v = self.rates_alpha[ind_epoch, ind_step, ind_elev]
            
            min_mask = (vx_He2>vsw_min)
            # print("speed :", np.average(self.vels_He2[m], weights=weights_v))

            vx_mean = np.average(vx_He2[min_mask], weights=weights_v[min_mask])
            vy_mean = np.average(vy_He2[min_mask], weights=weights_v[min_mask])
            vz_mean = np.average(vz_He2[min_mask], weights=weights_v[min_mask])
            vx_var = np.average((vx_He2[min_mask]-vx_mean)**2, weights=weights_v[min_mask])
            vy_var = np.average((vy_He2[min_mask]-vy_mean)**2, weights=weights_v[min_mask])
            vz_var = np.average((vz_He2[min_mask]-vz_mean)**2, weights=weights_v[min_mask])
            vx_th = np.sqrt(vx_var)
            vy_th = np.sqrt(vy_var)
            vz_th = np.sqrt(vz_var)    
            
            doy = ap(doy, self.udoy[uepoch_ind])
            v = ap(v, np.sqrt(vx_mean**2+vy_mean**2+vz_mean**2))
            vr = ap(vr, vx_mean)
            vt = ap(vt, vy_mean)
            vn = ap(vn, vz_mean)

            Tr = ap(Tr, vx_th)
            Tt = ap(Tt, vy_th)
            
        print(v)
        m_He2 = 4*1.66e-27#in kg
        to_eV = 6.242e18#eV/J
        Tr = 1/2*m_He2*(Tr*1000)**2*to_eV
        Tt = 1/2*m_He2*(Tt*1000)**2*to_eV
        
        if save_file==True:
            filename = '2022-09-17_velocities_temperatures'
            formats = ['%.8f', '%.6f', '%.6f', '%.6f', '%.6f', '%.2f', '%.2f']
            np.savetxt(filename+'.txt', np.column_stack((doy, v, vr, vt, vn, Tr, Tt)), delimiter=' ', header='time,<|v|>,<v_r>,<v_t>,<v_n>,<T_r>,<T_t>\ndoy km/s km/s km/s km/s eV eV\n', comments='', fmt=formats)
        # return v, vr, vt, vn
    
    def calc_velocities_temperatures_new(self, vsw_min=100, pr_cor=True, step_cor=False, save_file=False):
        '''
        Not working!
        Method which calculates the mean velocity vector considering the missing counts below the edge. 
        '''
        doy = np.array([])
        v = np.array([])
        vr = np.array([])
        vt = np.array([])
        vn = np.array([])
        
        rescaling_factor = self.calc_alpha_counthist_step(rescale=True, return_weights=True)
        
        for uepoch_ind in range(0,2944):
            print(uepoch_ind)
            m=(self.epoch==self.uepoch[uepoch_ind])*(self.pr==self.pr_helium)*(self.vels_He2>vsw_min)
            data = np.array([self.vels_He2[m], self.elev[m], self.azim_rad[m]])
            bins_vels_He2 = ap(self.uvels_He2, self.uvels_He2[-1]+(self.uvels_He2[-1]-self.uvels_He2[-2]))
            bins_azim = ap(self.uazim_rad, self.uazim_rad[-1]+(self.uazim_rad[-1]-self.uazim_rad[-2]))
            bins_elev = ap(self.uelev, self.uelev[-1]+(self.uelev[-1]-self.uelev[-2]))
            bins = [bins_vels_He2, bins_elev, bins_azim]
            self.hist, self.edges = np.histogramdd(data.T, bins) 
            
            if pr_cor==True:
                pr_weights = self.rates_alpha[uepoch_ind,:,:]
                pr_weights_extended = np.tile(pr_weights[:,:,np.newaxis], (1,1,64))
                self.hist = self.hist*pr_weights_extended
                
            if step_cor==True:
                print(rescaling_factor[uepoch_ind])
                self.hist[:28,:,:] = rescaling_factor[uepoch_ind]*self.hist[:28,:,:]
                
            vels_He2_mesh, elev_mesh, azim_mesh = np.meshgrid(self.edges[0][:-1],self.edges[1][:-1],self.edges[2][:-1],indexing='ij')
            
            speed_He2 = vels_He2_mesh.ravel()
            elevation = elev_mesh.ravel()
            azimuth = azim_mesh.ravel()

            weights = np.ravel(self.hist)
            
            # wrong_mean_speed = np.average(speed_He2, weights=weights)
            # print("speed :", wrong_mean_speed)
            
            vx = np.cos(azimuth)*np.cos(np.pi/180*elevation)*speed_He2
            vy = np.sin(azimuth)*np.cos(np.pi/180*elevation)*speed_He2
            vz = np.sin(np.pi/180*elevation)*speed_He2
            
            min_mask = (vx>vsw_min)
            vx_mean = np.average(vx[min_mask], weights=weights[min_mask])
            vy_mean = np.average(vy[min_mask], weights=weights[min_mask])
            vz_mean = np.average(vz[min_mask], weights=weights[min_mask])
            
            doy = ap(doy, self.udoy[uepoch_ind])
            vr = ap(vr, vx_mean)
            vt = ap(vt, vy_mean)
            vn = ap(vn, vz_mean)
            v = ap(v, np.sqrt(vx_mean**2 + vy_mean**2 + vz_mean**2))
            
        print(v)
        if save_file==True:
            filename = '2022-09-16_velocities_new_hist_calc'
            formats = ['%.8f', '%.6f', '%.6f', '%.6f', '%.6f']
            np.savetxt(filename+'.txt', np.column_stack((doy, v, vr, vt, vn)), delimiter=' ', header='time,<|v|>,<v_r>,<v_t>,<v_n>\ndoy km/s km/s km/s km/s\n', comments='', fmt=formats)
        # return v, vr, vt, vn
    
    
    def calc_velocities_elevations(self, uepoch_ind=0, vsw_min=100):
        '''
        Method to calculate the mean velocity vector for different elevations for one specific measurement time.
        '''
        doy = np.array([])
        v = np.array([])
        vr = np.array([])
        vt = np.array([])
        vn = np.array([])

        for uelev_ind in range(0, len(self.uelev)):
            print('uelev_ind: ', uelev_ind)
            m=(self.epoch==self.uepoch[uepoch_ind])*(self.elev==self.uelev[uelev_ind])*(self.pr==self.pr_helium)
            epoch_he2=self.epoch[m]
            step_he2=self.step[m]
            elev_he2=self.elev[m]

            vx_He2=np.cos(self.azim_rad[m])*np.cos(self.elev_rad[m])*self.vels_He2[m]
            vy_He2=np.sin(self.azim_rad[m])*np.cos(self.elev_rad[m])*self.vels_He2[m]
            vz_He2=np.sin(self.elev_rad[m])*self.vels_He2[m]
            
            ind_epoch = np.searchsorted(self.uepoch, epoch_he2)
            ind_step = np.searchsorted(self.ustep, step_he2)
            ind_elev = np.searchsorted(self.uelev, elev_he2)
            
            weights_v = self.rates_alpha[ind_epoch, ind_step, ind_elev]
            
            min_mask = (vx_He2>vsw_min)
            
            
            vx_mean = np.average(vx_He2[min_mask], weights=weights_v[min_mask])
            vy_mean = np.average(vy_He2[min_mask], weights=weights_v[min_mask])
            vz_mean = np.average(vz_He2[min_mask], weights=weights_v[min_mask])
            # vx_var = np.average((vx_He2[min_mask]-vx_mean)**2, weights=weights_v[min_mask])
            # vy_var = np.average((vy_He2[min_mask]-vy_mean)**2, weights=weights_v[min_mask])
            # vz_var = np.average((vz_He2[min_mask]-vz_mean)**2, weights=weights_v[min_mask])
            # vx_th = np.sqrt(vx_var)
            # vy_th = np.sqrt(vy_var)
            # vz_th = np.sqrt(vz_var)
                        
            doy = ap(doy, self.udoy[uepoch_ind])
            vr = ap(vr, vx_mean)
            vt = ap(vt, vy_mean)
            vn = ap(vn, vz_mean)
            v = ap(v, np.sqrt(vx_mean**2 + vy_mean**2 + vz_mean**2))

        
        v_all_elev = 525.5033
        plt.figure()
        plt.plot(self.uelev, v, 'o', label='v')
        plt.plot(self.uelev, np.sqrt(vr**2 + vt**2 + vn**2), 'x', label='sqrt(vr^2 + vt^2 + vn^2)')
        plt.plot(self.uelev, vr, 'o', label='v_r')
        plt.plot(self.uelev, vt, 'o', label='v_t')
        plt.plot(self.uelev, vn, 'o', label='v_n')
        plt.plot(self.uelev, np.ones_like(self.uelev)*v_all_elev, color='black', label=r'$\langle|v|\rangle$')
        plt.xlabel("elevation [°]")
        plt.ylabel("v [km/s]")
        plt.legend(loc='lower right')
        plt.savefig('Compare_v_vr_vt_vn_for_elevations.png', dpi=400)
        
       
    def plot_hist_speed_azim(self, uepoch_ind=np.arange(100), uelev_ind=9):
        '''
        Method to plot the azimuth angle against speed to see the unphysical azimuth angle measurement.
        Input uepoch_ind: array of times where we sum the counts.
        '''
        prc_label="pr_cor"
        m=(self.epoch==self.uepoch[uepoch_ind[0]])*(self.elev==self.uelev[uelev_ind])*(self.pr==self.pr_helium)
        bins_vels_He2=ap(self.uvels_He2, self.uvels_He2[-1]+(self.uvels_He2[-1]-self.uvels_He2[-2]))
        bins_azim=ap(self.uazim, self.uazim[-1]+(self.uazim[-1]-self.uazim[-2]))
        Cs0,sb,ab=np.histogram2d(self.vels_He2[m],self.azim[m],[bins_vels_He2,bins_azim])      
        
        Cs_sum = np.zeros_like(Cs0)
        
        for ind in uepoch_ind:
            m=(self.epoch==self.uepoch[ind])*(self.elev==self.uelev[uelev_ind])*(self.pr==self.pr_helium)
            
            Cs,sb,ab=np.histogram2d(self.vels_He2[m],self.azim[m],[bins_vels_He2,bins_azim])
            
            pr_weights=self.rates_alpha[ind,:,uelev_ind]
            pr_weights_vdfslice=np.array([pr_weights]*len(self.uazim))
            Cs=Cs*pr_weights_vdfslice.T
            Cs_sum += Cs
            
        fig, ax = plt.subplots(1,1)
        Cont1=ax.pcolormesh(sb,ab,Cs_sum.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(np.ravel(Cs_sum))))
        cb1 = fig.colorbar(Cont1)
        cb1.set_label("PHA counts per bin")	
        ax.set_title("He2+ VDF elevation-slice in instr. frame, (|v|, azim)-coords\n DOY: %.5f , elev: %.2f deg, %s"%(self.udoy[0], self.uelev[uelev_ind],prc_label))
        ax.set_xlabel("|v| [km/s]")
        ax.set_ylabel("azim. angle [deg]")
        plt.savefig('HIS_azimuth_speed_hist_50_min_avg_20230120.png', dpi=400)
        # plt.close()
        return
    

        
        
        
class HIS_Rates(object):            
    """
    Auxilliary class to load the HIS priority rates to correct the PHA data.
    Object of this class is called in the HIS_PHA class.
    """
    def __init__(self,date=(2022,9,16),version="V02",inpath="Data/2022 his rates/"):
        
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
        self.cdf_file=cdflib.CDF(inpath+filename) 
        print("loaded %s"%(filename))
        
        #load data products
        self.epoch = self.cdf_file.varget("EPOCH")
        self.prange = self.cdf_file.varget("PRIORITY")        
        self.rates = self.cdf_file.varget("PRIORITY_RATE")
        self.rates_alpha = self.rates[:,5,:,:]
        
def HIS_timeseries(num_files=180, inpath="Data/2022/"):
    filenames = [file for file in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, file))]
    num_files = len(filenames)
    
    stop_index = 180
    index = 0
    for file in filenames:
        if index >= stop_index:
            break
        date_tuple = (int(file[20:24]), int(file[24:26]), int(file[26:28]))
        # print(date_tuple)
        try:
            h = HIS_PHA(date=date_tuple, load_prates=True, inpath=inpath)
        except:
            continue
        try:
            h.calc_mean_speed(save_to_file=True)
        except:
            continue
        # h.calc_alpha_counthist_step(save_speed=1)
        index += 1
    
