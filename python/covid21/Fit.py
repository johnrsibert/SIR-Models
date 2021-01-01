from covid21 import config as cv
from covid21 import Geography as GG
from covid21 import GraphicsUtilities as GU

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import pyreadr



class Fit(GG.Geography):

    def __init__(self,fit_file,**kwargs):
        super().__init__(**kwargs)
        pn = fit_file #cv.fit_path+fit_file
        filename, extension = os.path.splitext(pn)
        if (extension == '.RData'):
            self.fit_type = 'TMB'
            tfit=pyreadr.read_r(pn)
        else:
            sys.exit('class Fit  not yet implemented on '+extension+' files.')

        head, tail = os.path.split(pn)
        self.moniker = os.path.splitext(tail)[0]
        self.diag = tfit['diag']
        self.md = tfit['meta']
        self.ests = tfit['ests']
        if (cv.pyreadr_kludge):
        #   print('keys',tfit.keys())
        #   print(tfit['diag'].columns)
        #   print(len(tfit['diag'].columns))
            insert = len(tfit['diag'].columns)
        #   print('head:',head,'tail:',tail,tail[0])
        #   print('filename',filename,'extension',extension)
            csv_file = filename + '_stderror.csv'
        #   print('reading',csv_file)
            stderror = pd.read_csv(csv_file,header=0)
        #   print(stderror['logbeta'],stderror['logmu'])
        #   SElogbeta = stderror['logbeta']
        #   SElogmu = stderror['logmu']
            self.diag.insert(insert,'SElogbeta',stderror['logbeta'])
            self.diag.insert(insert+1,'SElogmu',stderror['logmu'])
        #   print(self.diag)

        self.date0 = self.get_metadata_item('Date0')
        self.ntime = int(self.get_metadata_item('ntime'))
    #   Date0 = self.get_metadata_item('Date0')
    #   print('Date0:',Date0)

    def print_metadata(self):
        super().print_metadata()
        print('type:',self.fit_type)

    def get_metadata_item(self,mdname):
        r = self.md['names'].isin([mdname])
        return(self.md.data[r].values[0])

    def get_estimate_item(self, ename):
        r = self.ests['names'].isin([ename])
        if (r.any() == True):
            return(float(self.ests.est[r]))
        else:
            return(None)
    
    def get_est_or_init(self,name):
        v = self.get_estimate_item(name) 
    #   if (isNaN(v)):
    #       v = self.get_initpar_item(name)
    #       return(v)
    #   else:
    #       return(v)
        return(v)
    
    def get_initpar_item(self,pname):
        r = self.ests['names'].isin([pname])
        if (r.any() == True):
            return(float(ests.init[r]))
        else:
            return(None)
       
    def plot(self,logscale=True, per_capita=False, delta_ts=False,
             npl = 4, save = True, show_median = False):
        """ 
        """
    
        firstDate = mdates.date2num(cv.FirstNYTDate)
        orderDate = mdates.date2num(cv.CAOrderDate)
        lastDate  = mdates.date2num(cv.EndOfTime)
    
        date_list = pd.date_range(start=firstDate,end=lastDate)
        date_lim = [date_list[0],date_list[len(date_list)-1]]
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams["scatter.marker"] = '+'
        plt.rcParams["lines.markersize"] = 6
        prefix = ''
        if (logscale):
            prefix = 'ln '
             
        fig, ax = plt.subplots(npl,1,figsize=(9.0,(npl)*3.0))
        if (per_capita):
            ax[0].set_ylabel(prefix+'Cases'+' per '+str(mult))
            ax[1].set_ylabel(prefix+'Deaths'+' per '+str(mult))
        else:
            ax[0].set_ylabel(prefix+'Cases')
            ax[1].set_ylabel(prefix+'Deaths')
        if (npl > 2):
            ax[2].set_ylabel(r'$\ln\ \beta\ (da^{-1})$')
        if (npl > 3):
            ax[3].set_ylabel(r'$\ln\ \mu\ (da^{-1})$')
    
    
        for a in range(0,len(ax)):
            GU.make_date_axis(ax[a],firstDate)
    
        Date0 = self.get_metadata_item('Date0')
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')
        pdate = []
        for t in range(0,len(self.diag.index)):
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))
    
        obsI = self.diag['log_obs_cases']
        preI = self.diag['log_pred_cases']
        obsD = self.diag['log_obs_deaths']
        preD = self.diag['log_pred_deaths']
        SElogbeta = self.diag['SElogbeta']
        SElogmu   = self.diag['SElogmu']
        sigma_logC = np.exp(self.get_est_or_init('logsigma_logC'))
        sigma_logD = np.exp(self.get_est_or_init('logsigma_logD'))
        sigma_logbeta = self.get_est_or_init('logsigma_logbeta')
        sigma_logmu = np.exp(self.get_est_or_init('logsigma_logmu'))
        if (logscale):
            ax[0].set_ylim(0.0,1.2*max(obsI))
            ax[0].scatter(pdate,obsI,marker = '.',s=16,c='r')
            ax[0].plot(pdate,preI)

        else:
            ax[0].set_ylim(0.0,1.2*max(np.exp(obsI)))
            ax[0].scatter(pdate,np.exp(obsI),marker = '.',s=16,c='r')
            ax[0].plot(pdate,np.exp(preI))
        GU.plot_error(ax[0],pdate,obsI,sigma_logC,logscale)
        tx = GU.prop_scale(ax[0].get_xlim(), 0.05)
        ty = GU.prop_scale(ax[0].get_ylim(), 0.90)
        sigstr = '%s = %.3g'%('$\sigma_I$',sigma_logC)
        ax[0].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)
    
        if (logscale):
            ax[1].set_ylim(0.0,1.2*max(obsD))
            ax[1].scatter(pdate,obsD,marker = '.',s=16,c='r')
            ax[1].plot(pdate,preD)
        else:
            ax[1].set_ylim(0.0,1.2*max(np.exp(obsD)))
            ax[1].scatter(pdate,np.exp(obsD),marker = '.',s=16, c='r')
            ax[1].plot(pdate,np.exp(preD))

        GU.plot_error(ax[1],pdate,obsD,sigma_logD,logscale)
        tx = GU.prop_scale(ax[1].get_xlim(), 0.05)
        ty = GU.prop_scale(ax[1].get_ylim(), 0.90)
        sigstr = '%s = %.3g'%('$\sigma_D$',sigma_logD)
        ax[1].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)
    
        if (npl > 2):
            log_beta = self.diag['logbeta']
        #   log_beta_ticks = [ 1.01978144,0.32663426,-0.36651292,
        #                     -1.0596601,-1.75280728,-2.44595446,
        #                     -3.13910164,-3.83224882,-4.525396  ]
        #   print(min(log_beta_ticks),max(log_beta_ticks))
        #   print(0.8*min(log_beta_ticks),1.2*max(log_beta_ticks))
        #   ax[2].set_ylim((1.2*min(log_beta)),1.2*max(log_beta))
        #   ax[2].set_yticks(log_beta_ticks)
            sigstr = '%s = %.3g'%('$\sigma_\\beta$',sigma_logbeta)
            tx = GU.prop_scale(ax[2].get_xlim(), 0.05)
            ty = GU.prop_scale(ax[2].get_ylim(), 0.90)
            ax[2].plot(pdate,log_beta)
        #   plot_error(ax[2],pdate,log_beta,sigma_logbeta,logscale=True)
            GU.plot_error(ax[2],pdate,log_beta,SElogbeta,logscale=True)
            ax[2].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)

            if (show_median):
                med = median(np.exp(log_beta))
                logmed = np.log(med)
                medstr = '%s = %.3g'%('$\\tilde{\\beta}$',med)
                ax[2].text(ax[2].get_xlim()[0],logmed,medstr,ha='left',va='bottom',fontsize=10)
                ax[2].plot(ax[2].get_xlim(),[logmed,logmed])

        #   increase frequcncy of tick marks
            start, end = ax[2].get_ylim()
            dtick = (end - start)/5
            ax[2].set_yticks(np.arange(start, end, dtick))

        #   finagle doubling time axis at same scale as beta
            add_doubling_time = False
            if (add_doubling_time):
                dtax = ax[2].twinx()
                dtax.set_ylim(ax[2].get_ylim())
                dtax.grid(False,axis='y') # omit grid lines
                dtax.set_ylabel('Doubling Time (da)')
            #   render now to get the tick positions and labels
            #   fig.show()
            #   fig.canvas.draw()
                y2_ticks = dtax.get_yticks()
                labels = dtax.get_yticklabels()
                labels[0] = ''
                for i in range(1,len(y2_ticks)):
                    y2_ticks[i] = np.log(2)/np.exp(y2_ticks[i])
                    labels[i] = round(float(y2_ticks[i]),2)
                      
                dtax.tick_params(length=0)
                dtax.set_yticklabels(labels)
            #   fig.show()
            #   fig.canvas.draw()

        if (npl > 3):
            logmu = self.diag['logmu']
        #   ymin = min(self.diag['logmu'])-2.5*sigma_logmu
        #   ax[3].set_ylim(ymin,1.2*max(self.diag['logmu']))
        #   print(min(logmu),max(logmu))
        #   print(1.2*min(logmu),0.8*max(logmu))
            ax[3].set_ylim((1.2*min(logmu),0.8*max(logmu)))

            sigstr = '%s = %.3g'%('$\sigma_\\mu$',sigma_logmu)
            tx = GU.prop_scale(ax[3].get_xlim(), 0.05)
            ty = GU.prop_scale(ax[3].get_ylim(), 0.90)
            ax[3].plot(ax[2].get_xlim(),[0.0,0.0],color='0.2',linestyle='--')
            ax[3].plot(pdate,self.diag['logmu'])
        #   plot_error(ax[3],pdate,logmu,sigma_logmu,logscale=True)
            GU.plot_error(ax[3],pdate,logmu,SElogmu,logscale=True)
            ax[3].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)

            if (show_median):
                med = median(np.exp(self.diag['logmu']))
                logmed = np.log(med)
                medstr = '%s = %.3g'%('$\\tilde{\\mu}$',med)
                ax[3].text(ax[3].get_xlim()[0],logmed,medstr,ha='left',va='bottom',fontsize=10)
                ax[3].plot(ax[3].get_xlim(),[logmed,logmed])
    
        title = self.moniker #self.name+' County, '+self.enclosed_by
        fig.text(0.5,0.95,title ,ha='center',va='bottom')

        if save:
            gfile = cv.graphics_path+self.moniker+'_'+self.fit_type+'_estimates'
            if (not logscale):
                gfile = cv.graphics_path+self.moniker+'_'+self.fit_type+'_estimates_a'
            fig.savefig(gfile+'.png',dpi=300)
            plt.show(block=False)
            print('plot saved as',gfile)
            plt.pause(3)
            plt.close()
        else:
            plt.show(True)

    def plot_CMR(self, save = True):
        npl = 3
        fig, ax = plt.subplots(npl,1,figsize=(9.0,(npl)*3.0))
        ax[0].set_ylabel('Empirical CMR')
        ax[1].set_ylabel('Estimated CMR')
        ax[2].set_ylabel(r'$\mu\ (da^{-1})$')
 
        for a in range(0,len(ax)):
            self.make_date_axis(ax[a])
    
        Date0 = self.get_metadata_item('Date0')
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')
        pdate = []
        for t in range(0,len(self.diag.index)):
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))
        
    #   self.get_pdate()
        obsI = self.diag['obs_cases']
        obsD = self.diag['obs_deaths']
        obsCMR = obsD/obsI
        estI = np.exp(self.diag['log_pred_cases'])
        estD = np.exp(self.diag['log_pred_deaths'])
        estCMR = estD/estI 
        mu   = np.exp(self.diag['logmu'])

        ax[0].plot(pdate,obsCMR)
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].plot(pdate,estCMR)
        ax[2].plot(pdate,mu)

        fig.text(0.5,0.95,'Case Mortality Ratio' ,ha='center',va='bottom')
        if (save):
            gfile = cv.graphics_path+self.moniker+'_CMR.png'
            fig.savefig(gfile)
            plt.show(False)
            print('plot saved as',gfile)
            plt.pause(3)
        else:
            plt.show()

