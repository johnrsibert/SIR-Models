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
import glob
#import re
import statistics as stats
from tabulate import tabulate



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
        self.md.set_index('names',inplace=True)
        self.ests.set_index('names',inplace=True)
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
        return(self.md.loc[mdname].data)

    def get_estimate_item(self, ename):
        try:
            r = self.ests.loc[ename].est
        except (KeyError):
            r = None
        return(r)


#   def get_active(self, name):
#       print('get_active',name)
#       try:
#           a = self.ests.loc[name].map
#       except (KeyError):
#           a = None

#       print('aaaaaaaaaaaaaaaaaa')
#       print('type =',type(a))
#       print('a:',a)

#       print('aaaaaaaaaaaaaaaaaa')
#       
#       if (a is None):
#           return('Fixed')
#       else:
#           return('Active')


    def get_est_or_init(self,name):
        print(name)
        v = self.get_estimate_item(name) 
        print(v)
    #   if (isNaN(v)):
    #       v = self.get_initpar_item(name)
    #       return(v)
    #   else:
    #       return(v)
        return(v)
    
    def get_initpar_item(self,pname):
        try:
            r = float(self.ests.loc[pname].init)
        except (KeyError):
            r = None
        return(r)         
       
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
                med = stats.median(np.exp(log_beta))
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
                med = stats.median(np.exp(self.diag['logmu']))
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

def get_mu_atD1(ext='.RData',fit_files = []):
    if (len(fit_files) < 1):
        fit_files = glob.glob(cv.fit_path+'*'+ext)
    else:
        for i,ff in enumerate(fit_files):
            fit_files[i] = cv.fit_path+fit_files[i]+ext

    mud_cols = ['moniker','lag','logmu','mu']
    tmp = np.empty(shape=(len(fit_files),4),dtype=object)
    for i,ff in enumerate(fit_files):
        moniker = os.path.splitext(os.path.basename(ff))[0]
        fit = Fit(ff)
        muD1 = -1.0
        obs_deaths = fit.diag['obs_deaths']
        for t in range(0,len(fit.diag.index)):
            if (muD1 < 0.0) and (obs_deaths[t] > 0.0):
                muD1 = np.exp(fit.diag['logmu'][t])
                lag = t-1
                tmp[i] = (moniker,lag,fit.diag['logmu'][t],muD1)

    mud = pd.DataFrame(tmp,columns=mud_cols)
    mud = mud.sort_values(by='lag',ascending=True)
    print(mud)
    mud.to_csv(cv.fit_path+'mud.csv',index=False)
    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    ax.set_ylabel('First '+r'$\ln\ \mu\ (da^{-1})$')
    ax.set_xlabel('Lag (days)') 
#   ax.hist(mud['logmu'],density=True,bins=3)
    ax.plot(mud['lag'],(mud['logmu']))
    plt.show(True)

def make_rate_plots(yvarname = 'logbeta',ext = '.RData', 
                    fit_files = [], show_medians = False, 
                    add_doubling_time = False, show_order_date = True, save=False):
    print(yvarname)
    if (yvarname == 'logbeta'):
        ylabel =r'$\ln \beta\ (da^{-1})$'
    elif (yvarname == 'logmu'):
        ylabel =r'$\ln \mu\ (da^{-1})$'
    elif (yvarname == 'gamma'):
        ylabel =r'$\gamma\ (da^{-1})$'
    else:
        sys.exit('Unknown yvarname '+yvarname)

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    if (len(fit_files) < 1):
        suffix = '_g'
        fit_files = glob.glob(cv.fit_path+'*'+ext)
    #   fit_files = glob.glob(cv.fit_path+'constrainID/'+'*'+ext)
    else:
        suffix = '_'+str(len(fit_files))
        for i,ff in enumerate(fit_files):
            fit_files[i] = cv.fit_path+fit_files[i]+ext
    for i,ff in enumerate(fit_files):
        print(i,ff)
        fit = Fit(ff)
        if (i == 0):
            GU.make_date_axis(ax)
            ax.set_ylabel(ylabel)
       
        pdate = []
        Date0 = datetime.strptime(fit.date0,'%Y-%m-%d')
        for t in range(0,len(fit.diag.index)):
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))

        yvar = fit.diag[yvarname]
        if (yvarname == 'gamma'):
            print(yvarname)
            print(min(yvar),max(yvar))
        ax.plot(pdate,yvar)

    #   sigma_logbeta is the standard deviation of the generating
    #   random walk, NOT the standard deviation of the estimated
    #   random effect
        if (yvarname == 'logbeta' and len(fit_files) <=4):
        #   sigma_logbeta = fit.get_est_or_init('logsigma_logbeta')
        #   plot_error(ax,pdate,yvar,sigma_logbeta,logscale=True)
            GU.plot_error(ax,pdate,yvar,fit.diag['SElogbeta'],logscale=True)

        sn = GG.short_name(fit.moniker)
        if (yvarname == 'logbeta'):
            GU.mark_ends(ax,pdate,yvar,sn,'b')
        else:
            GU.mark_ends(ax,pdate,yvar,sn,'b')

        if (show_medians):
            med = stats.median(np.exp(yvar))
            logmed = np.log(med)
            ax.plot(ax.get_xlim(),[logmed,logmed],linewidth=1,
                    color=ax.get_lines()[-1].get_color())

    if (show_order_date):
        GU.add_order_date(ax)

#   finagle doubling time axis at same scale as beta
    if (add_doubling_time):
        yticks = np.arange(-7,1,1)
        ax.set_ylim(min(yticks)-1,max(yticks)+1)
        ax.set_yticks(yticks)
        dtax = ax.twinx()
        dtax.set_ylim(ax.get_ylim())
        dtax.grid(False,axis='y') # omit grid lines
        dtax.set_ylabel('Doubling Time (da)')
    #   render now to get the tick positions and labels
    #   fig.canvas.draw()
        y2_ticks = dtax.get_yticks()
        labels = dtax.get_yticklabels()
        for i in range(0,len(y2_ticks)):
            y2_ticks[i] = np.log(2)/np.exp(y2_ticks[i])
        #   labels[i] = '%.1f'%y2_ticks[i]
            labels[i] = round(float(y2_ticks[i]),2)

        dtax.tick_params(length=0)
        dtax.set_yticklabels(labels)

    if save:
        gfile = cv.graphics_path+yvarname+'_summary'+suffix+'.png'
        fig.savefig(gfile)
        print('plot saved as',gfile)
        plt.show(block=False)
    else:
        plt.show(block=True)

def make_fit_plots(ext = '.RData'):
    fit_files = glob.glob(cv.fit_path+'*'+ext)
    print('found',len(fit_files),ext,'files in',cv.fit_path)
    plt.rc('figure', max_open_warning = 0)
    for ff in fit_files:
        fit = Fit(ff)
    #   fit.print_metadata()
        fit.plot(logscale=True)

    fit = Fit(cv.fit_path+'NassauNY'+ext)
    fit.plot(save=True,logscale=False)
    fit = Fit(cv.fit_path+'Miami-DadeFL'+ext)
    fit.plot(save=True,logscale=False)
    fit = Fit(cv.fit_path+'New_York_CityNY'+ext)
    fit.plot(save=True,logscale=False)
    fit = Fit(cv.fit_path+'Los_AngelesCA'+ext)
    fit.plot(save=True,logscale=False)

def make_fit_table(ext = '.RData'):

    def get_active(ests, name):
        print('get_active',name)
        try:
            a = ests.loc[name].map
        except (KeyError):
            a = None

        if (a is None):
            return('?Fixed')
        else:
            return('?Active')

    def get_initpar_item(self,pname):
        try:
            r = float(self.ests.loc[pname].init)
        except (KeyError):
            r = None
        return(r)         
       
 

    fit_files = glob.glob(cv.fit_path+'*'+ext)
    print('found',len(fit_files),ext,'files in',cv.fit_path)
#   md_cols = ['county','N0','ntime','prop_zero_deaths','fn']
    md_cols = ['county','ntime','prop_zero_deaths','fn','C']
    es_cols = ['logsigma_logCP','logsigma_logDP','logsigma_logbeta','logsigma_logmu',
               'logsigma_logC','logsigma_logD','mbeta','mmu','mgamma']
    tt_cols = md_cols + es_cols
    header = ['County','$n$','$p_0$','$f$','$C$',
              '$\sigma_{\eta_C}$', '$\sigma_{\eta_D}$', '$\sigma_\\beta$','$\sigma_\\mu$',
              '$\sigma_{\ln I}$','$\sigma_{\ln D}$','$\\tilde{\\beta}$','$\\tilde{\\mu}$',
              '$\\tilde\\gamma$']

    tt = pd.DataFrame(columns=tt_cols,dtype=None)

    func = pd.DataFrame(columns=['fn'])
    mgamma = pd.DataFrame(columns=['mgamma'])
    mbeta = pd.DataFrame(columns=['mbeta'])
    mmu = pd.DataFrame(columns=['mmu'])
    sigfigs = 5
    init_gamma = None
    active_gamma = None
    for k in range(0,len(fit_files)):
        ff = fit_files[k]
        print('adding fit',k,ff)
        fit = Fit(ff)
        ests  = fit.ests #['ests']
        meta = fit.md #['meta']
        diag = fit.diag #['diag']
        row = pd.Series(index=tt_cols)
        county = fit.get_metadata_item('county')  
        row['county'] = GG.pretty_county(county)
        if (active_gamma is None):
            active_gamma = get_active(ests,'loggamma')
            init_gamma = np.exp(fit.get_initpar_item('loggamma'))

        for k in range(1,len(tt_cols)):
            v = fit.get_estimate_item(tt_cols[k])
            row.iloc[k] = v

    #   row['N0'] = int(get_metadata('N0',meta))
        row['C'] = fit.get_metadata_item('convergence')
        row['ntime'] = int(fit.get_metadata_item('ntime'))
    #   row['prop_zero_deaths'] = round(float(fit.get_metadata_item('prop_zero_deaths')),sigfigs)
        row['prop_zero_deaths'] = float(fit.get_metadata_item('prop_zero_deaths'))
        tt = tt.append(row,ignore_index=True)
        func = np.append(func,float(fit.get_metadata_item('fn')))
        beta = np.exp(diag['logbeta'])
        mbeta = np.append(mbeta,stats.median(beta))
        mu = np.exp(diag['logmu'])
        mmu = np.append(mmu,mu.quantile(q=0.5))
        gamma = diag['gamma']
        mgamma = np.append(mgamma,gamma.quantile(q=0.5))

    tt['fn'] = func
    tt['mgamma'] = mgamma
    tt['mbeta'] = mbeta
    tt['mmu'] = mmu

    mtime = os.path.getmtime(cv.NYT_counties)
    dtime = datetime.fromtimestamp(mtime)
    ft_name = cv.fit_path+'fit_table_'+str(dtime.date())

    csv = ft_name+'.csv'
    tt.to_csv(csv,index=False)
    print('Fit table data written to file',csv)


    tt = tt.sort_values(by='mbeta',ascending=True)#,inplace=True)

    for r in range(0,tt.shape[0]):
        for c in range(3,len(tt.columns)):
           if (tt.iloc[r,c] != None):
               tt.iloc[r,c] = round(float(tt.iloc[r,c]),sigfigs)
        c = 2
        tt.iloc[r,c] = round(float(tt.iloc[r,c]),sigfigs)

    row = pd.Series(None,index=tt.columns)
    row['county'] = 'Median'
    for n in tt.columns:
        if (n != 'county'):
            mn = tt[n].quantile()
            row[n] = mn
    tt = tt.append(row,ignore_index=True)


    tex = ft_name+'.tex'
    ff = open(tex, 'w')
    caption_text = 'Model results. Estimating $\\beta$ and $\mu$ trends as random effects.\n' 
    caption_text = caption_text + 'Data updated ' + str(dtime.date()) + ' from https://github.com/nytimes/covid-19-data.git.\n'
    caption_text = caption_text + 'Initial $\gamma = '+str(init_gamma)+'$ ('+active_gamma+').\n'

    ff.write(caption_text)
#   ff.write(str(dtime.date())+'\n')
    ff.write(tabulate(tt, header, tablefmt="latex_raw",showindex=False))
#   tt.to_latex(buf=tex,index=False,index_names=False,longtable=False,
#               header=header,escape=False,#float_format='{:0.4f}'.format
#               na_rep='',column_format='lrrrrrrrrrrr')
    print('Fit table written to file',tex)

