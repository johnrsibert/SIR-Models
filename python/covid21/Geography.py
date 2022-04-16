from covid21 import config as cv
from covid21 import GraphicsUtilities as GU

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import sys
import re
from cycler import cycler
import scipy.stats as stats
from tabulate import tabulate

class Geography:

#   def __init__(self,name,enclosed_by,code):
    def __init__(self,**kwargs):
        self.gtype = None
        self.name = kwargs.get('name')
        self.enclosed_by = kwargs.get('enclosed_by')
        self.code = kwargs.get('code')
        self.population = None
        self.fips = None
        self.source = None
        if self.name is None:
            self.moniker = None
        else:
        #   self.moniker = str(self.name+self.code
        #   self.moniker =  self.moniker.replace(' ','_',5) 
            self.moniker = set_moniker(self.name,self.code)
    #   self.TMB_fit = None
    #   self.ADMB_fit = None
        if self.moniker is None:
            self.dat_file = None
        else:
            self.dat_file = cv.dat_path+self.moniker+'.dat'
        self.updated = None
        self.date0 = None
        self.ntime = None
        self.date = None
        self.cases = None
        self.deaths = None
        self.pdate = None # date for plotting on x-axis
        self.vax = None
        
    def print(self):
        print(self)
        
        
    def print_metadata(self):
        print('gtype:',self.gtype)
        print('gname:',self.name)
        print('genclosed_by:',self.enclosed_by)
        print('gcode:',self.code)
        print('fips:',self.fips)
        print('gpopulation:',self.population)
        print('gsource:',self.source)
        print('gmoniker:',self.moniker)
        print('gdat_file:',self.dat_file)
        print('gupdated:',self.updated)
        print('gdate0:',self.date0)
        print('gntime:',self.ntime)

    def print_data(self):
        self.get_pdate()
    #   if (len(self.deaths) > 0):
        if (self.deaths is not None):
            print('date','cases','deaths','pdate')
        else:
            print('date','cases','pdate')
        for k in range(0,len(self.date)):
            if (self.deaths is not None):
                print(self.date[k],self.cases[k],self.deaths[k],self.pdate[k])
            else:
                print(self.date[k],self.cases[k],self.pdate[k])
            
    def to_DataFrame(self):
        self.get_pdate()
        Gdf = pd.DataFrame(columns=('date','pdate','cases','deaths'))
        Gdf['date'] = self.date
        Gdf['pdate'] = self.pdate
        Gdf['cases'] = self.cases
        Gdf['deaths'] = self.deaths
        return(Gdf)

    def get_pdate(self):
        if (self.pdate is not None):
            return(self.pdate)

        else:
            self.pdate = []
            for d in self.date:
                self.pdate.append(mdates.date2num(datetime.strptime(d,'%Y-%m-%d').date()))
            return(self.pdate)

    def get_county_pop(self):
        """
        
        """

        dat = cv.GeoIndex
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
    #   COUNTY_filter = (dat['COUNTY']>0)
        County_rows = state_filter & county_filter #& COUNTY_filter
        try:
            population = int(pd.to_numeric(dat[County_rows]['population'].values))
        except:
            print('get_county_pop() failed for:')
            print(self.name, self.enclosed_by,self.code) 
            population = 1
        
        self.fips = int(pd.to_numeric(dat[County_rows]['fips'].values))
        
        return(population)   
        
    def read_nyt_data(self,gtype='county'):
        self.gtype = gtype
        mtime = 0.0
        if (gtype == 'county'):
            dat = cv.nyt_county_dat
        #   if (dat.empty):
        #      cspath = NYT_counties
        #      print('Reading',cspath)
        #      cv.nyt_county_dat = pd.read_csv(cspath,header=0)
        #      dat = cv.nyt_county_dat
        #      mtime = os.path.getmtime(cspath)
        else:
            sys.exit('No data for',gtype)
        
    #   dat = pd.read_csv(cspath,header=0)

        mtime = os.path.getmtime(cv.NYT_counties)
        dtime = datetime.fromtimestamp(mtime)
        self.updated = str(dtime.date())
        self.population = self.get_county_pop()
    
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
        County_rows = state_filter & county_filter
        if (len(County_rows) < 1):
            sys.exit(' * * * no records found for',self.name,self.surrounded_by)

   #    tmp = dat[County_rows]['date'].map(Strptime)
   #    self.pdate  = np.array(mdates.date2num(tmp))
        self.cases  = pd.array(dat[County_rows]['cases'])
        self.deaths = pd.array(dat[County_rows]['deaths'])
        self.date   = pd.array(dat[County_rows]['date'])
        self.date0 = self.date[0]
        self.ntime = len(self.date)
        self.source = 'New York Times, https://github.com/nytimes/covid-19-data.git.'
        
    def read_vax_data(self):
        #vax_name =  cv.CDC_home + 'vax.csv'
        #self.vax = pd.read_csv(vax_name,header=0,comment='#')
        #print('Vax data read from',vax_name)
        fips_filter = cv.cdc_vax_dat['fips'] == self.fips
        self.vax =  cv.cdc_vax_dat[fips_filter]
        #print(self.vax)
              
    def write_dat_file(self):
        print(len(self.date),'records found for',self.name,self.enclosed_by)
        O = open(self.dat_file,'w')
        O.write('# county\n %s\n'%(self.moniker))
        O.write('# updated from https://github.com/nytimes/covid-19-data.git\n')
        O.write(' %s\n'%self.updated)
        O.write('# population (N0)\n %10d\n'%self.population)
        O.write('# date zero\n %s\n'%self.date0)
        ntime = len(self.date)-1
        O.write('# ntime\n %4d\n'%ntime)
        O.write('#%6s %5s\n'%('cases','deaths'))
        for r in range(0,len(self.date)):
            O.write('%7d %6d\n'%(self.cases[r],self.deaths[r]))

    def read_BCHA_data(self,gtype='province'):
        self.gtype = gtype
        cspath = cv.BCHA_path
    #   print(cspath)

        dat = pd.read_csv(cspath,header=0)

        mtime = os.path.getmtime(cspath)
        dtime = datetime.fromtimestamp(mtime)
        self.updated = str(dtime.date())
    #   self.population = None
        self.population = self.get_county_pop()
    #   self.source = 'British Columbia Centre for Disease Control'\
        self.source = 'Government of British Columbia'\
                      ' www.bccdc.ca/Health-Info-Site/Documents/' 

        rdates = pd.DatetimeIndex(dat["Reported_Date"],normalize=True)
        self.date = pd.date_range(rdates[0],rdates[len(rdates)-1],normalize=True,freq='D')
        self.date = self.date.strftime('%Y-%m-%d') # NYT style
        self.date0 = self.date[0]
        self.ntime = len(self.date)
        self.get_pdate()


        daily_cases = pd.Series([0]*self.ntime)

        HA_filter = dat["HA"].isin([self.name])
        dat["rep_date"] = rdates
        for i in range(0,len(self.date)):
        #   date_mask = dat["Reported_Date"].isin([self.date[i]])
            date_mask = dat["rep_date"].isin([self.date[i]])
            mask = HA_filter & date_mask
            daily_cases[i] = len(dat[mask.values])

        self.cases = daily_cases.cumsum()
        


    def plot_prevalence(self,yscale='linear', per_capita=False, delta_ts=True,
                        window=[7], plot_dt = False, cumulative = False,
                        show_order_date = False,
                        show_superspreader = False,
                        low_prev = 0.0, qq=None,mult = 10000,
                        annotation = True, signature = False, 
                        save = True, dashboard = False, nax = 4):
        """ 
        Plots cases and deaths vs calendar date 

        scale: select linear of log scale 'linear'
        per_capita: plot numbers per 10000 (see mult)  False
        delta_ts: plot daily new cases True
        window: plot moving agerage window [11]
        plot_dt: plot initial doubling time slopes on log scale False
        annotations: add title and acknowledgements True
        save : save plot as file
        """

        firstDate = datetime.strptime(self.date[0],'%Y-%m-%d')
    #   orderDate = mdates.date2num(cv.CAOrderDate)

        if (self.deaths is None):
            nax = 1
   
        fig, pax = plt.subplots(nax,1,figsize=(6.5,nax*2.25))
        ax = [None]*nax
        if (nax == 1):
            ax[0] = pax
        else:
            ax=pax

        ylabel = ['Daily Cases','Daily Deaths','Case Fatality Ratio','Vaccinations (%)']
        end_marks = [r'$\Sigma$C','$\Sigma$D',''] # indicate cumulatives
        total_names = ['Cases','Deaths','']
        totals=[0]*3
        totals[0] = self.cases[self.ntime-1]
        gdf = pd.DataFrame()

        if (self.deaths is None):
            totals[1] = 0
            cfr = 0.0
        else:
            totals[1] = self.deaths[self.ntime-1]
            cfr = self.deaths/self.cases

        if (per_capita):
            for a in range(0,2):
                ylabel[a] = ylabel[a] +'/'+str(mult)
        
            cases =  mult*self.cases/self.population + cv.eps
            if (nax > 1):
                deaths =  mult*self.deaths/self.population + cv.eps
            else:
                deaths =  self.deaths
        else:
            cases  =  self.cases
            deaths =  self.deaths
        
        gdf['cases'] = cases
        gdf['deaths'] = deaths
        gdf['cfr'] = cfr

        '''
        ylim = [(0.0,1.2*gdf.iloc[:,0].max()),
                (0.0,1.2*gdf.iloc[:,1].max()),
                (0.0,1.2*gdf.iloc[:,2].max()),
                (0.0,101.0)]
        '''
        ylim = [(0.0,0.2*gdf.iloc[:,0].max()),
                (0.0,1.2*gdf.iloc[:,1].max()),
                (0.0,1.2*gdf.iloc[:,2].max()),
                (0.0,101.0)]
        
        ax2 = []
        for a in range(0,nax):
            GU.make_date_axis(ax[a],firstDate)
            if (a < 2):
                ax[a].set_yscale(yscale)
            ax[a].set_ylabel(ylabel[a])
            if (cumulative) & (a < 2):
                ax2.append(ax[a].twinx())
                ax2[a].set_ylabel('Cumulative')

        Date = pd.Series(self.get_pdate())

        for a in range(0,nax):
            if (a < 2):
                delta = np.diff(gdf.iloc[:,a]) #cases)
                ax[a].bar(Date[1:], delta)
                adc = None
                for w in range(0,len(window)):
                #   moving average
                    adc = pd.Series(delta).rolling(window=window[w]).mean()
                    ax[a].plot(Date[1:],adc,linewidth=2)
                    GU.mark_ends(ax[a],Date[1:],adc, str(window[w])+'da','r')

                if show_superspreader:
                    GU.add_superspreader_events(Date,adc,ax[a])

                if a == 0:
                    ylim[a] = (cv.eps,0.2*adc.max())
                else:
                    ylim[a] = (cv.eps,1.2*adc.max())
            #   print(ylim[a])
                ax[a].set_ylim(ylim[a])
                tx = GU.prop_scale(ax[a].get_xlim(), 0.5)
                ty = GU.prop_scale(ax[a].get_ylim(), 0.95)
            #   note = '{0:,} {1}'.format(int(gdf.iloc[-1,a]),total_names[a])
                note = '{0:,} {1}'.format(int(totals[a]),total_names[a])
                ax[a].text(tx,ty,note ,ha='center',va='top',fontsize=10)

                if (cumulative):
                    ax2[a].plot(Date, gdf.iloc[:,a], alpha=0.5, linewidth=1)#,label=cc)
                    GU.mark_ends(ax2[a],Date,gdf.iloc[:,a] ,end_marks[a],'r')
                    ax2[a].set_ylim(0.0,1.2*gdf.iloc[-1,a])

            elif a == 2:
                ax[a].plot(Date,gdf.iloc[:,a]) 
                ax[a].set_ylim(ylim[a])
                
            elif a == 3:
                ax[a].set_ylim(ylim[a])
                vmult = 100.0
                mdate = self.vax['mdate']
                pv = vmult*pd.Series(self.vax['first'])/self.population
                #print('number of vax estimates',len(pv),', max =',pv.max())
                #print(pv)
                if pv.max() > 1.0:
                    ax[a].plot(mdate,pv)
                    GU.mark_ends(ax[a],mdate,pv,'first','r')
                
                    pv = vmult*pd.Series(self.vax['full'])/self.population
                    ax[a].plot(mdate,pv)
                    GU.mark_ends(ax[a],mdate,pv,' full','r')
                
                    
               
                    if show_order_date:
                        GU.add_order_date(ax[a])
                else:
                    print('number of vax estimates',len(pv),', max =',pv.max())
                #    tx = mdate.iloc[0] + 0.5*(mdate.iloc[-1]-mdate.iloc[0])
                    tx = 0.5*(mdate.iloc[-1]+mdate.iloc[0])
                    ty = 50.0
                    #print(mdate)
                    #print(tx,ty)
                    ax[a].text(tx,ty,'Insufficient Data',ha='right',va='center',fontsize=14)

        if ((yscale == 'log') & (plot_dt) & (cumulative) ):
            # this is a bad idea
            ax[0] = GU.plot_dtslopes(ax[0],Date,gdf.iloc[:,0])

        if (annotation):
            if (self.gtype == 'county'):
                gname = 'County'
            else:
                gname = 'Region'
            title = 'Covid-19 Prevalence in '+self.name+' '+gname+', '+self.enclosed_by
            fig.text(0.5,1.0,title ,ha='center',va='top')
            if (nax < 4):
                GU.add_data_source(fig,self.source)
            else:
                GU.add_data_source(fig,'New York Times and Centers for Disease Control')


        if (signature):
            GU.add_signature(fig,'https://github.com/johnrsibert/SIR-Models')
        if low_prev > 0.0 and low_prev < ax[0].get_ylim()[1]:
            ax[0].plot((Date.iloc[0],Date.iloc[-1]),(low_prev,low_prev),
                       color='red',linewidth=1.5,linestyle=':')
            GU.mark_ends(ax[0],pd.Series([Date.iloc[0],Date.iloc[-1]]),
                         pd.Series([low_prev,low_prev]),
                         '$P_{0.1}$','r')
            GU.mark_ends(ax[0],pd.Series([Date.iloc[0],Date.iloc[-1]]),
                         pd.Series([low_prev,low_prev]),
                         '{:.2f}'.format(low_prev),'l')

        if qq is not None:
           nq = qq.shape[0]
           for a in range(0,nax):
               v = qq.columns[a]
               print(a,v)
           #   print(qq[v])
               for iq in range(0,nq):
                   q = qq[v].iloc[iq]
                   print('    ',iq,q,qq.index[iq])

                   ax[a].plot((Date.iloc[0],Date.iloc[-1]),(q,q),
                       color='green',linewidth=1.5,linestyle=':')
                   GU.mark_ends(ax[a],pd.Series([Date.iloc[0],Date.iloc[-1]]),
                        pd.Series([q,q]), '$P_{:.1f}$'.format(qq.index[iq]),'r')
                   GU.mark_ends(ax[a],pd.Series([Date.iloc[0],Date.iloc[-1]]),
                         pd.Series([q,q]), '{:.2f}'.format(q),'l')



        if (dashboard):
        #   out_img = BytesIO()
        #   plt.savefig(out_img, format='png')
        #   out_img.seek(0)  # rewind file
        #   encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        #   return "data:image/png;base64,{}".format(encoded)
            gfile = cv.graphics_path+'test.png'
            
            print('saving fig for dash:',gfile)
            plt.savefig(gfile,format='png',dpi=300)
            plt.close()
            print('saved')

        else:
            if save:
                gfile = cv.graphics_path+self.moniker+'_prevalence.png'
                plt.savefig(gfile,dpi=300)
                plt.show(block=False)
                plt.pause(2)
                plt.close()
                print('plot saved as',gfile)
            else:
                plt.show()

    def plot_per_capita_curvature(self,mult = 10000,save=False):
        pc_cases = self.cases/self.population * mult
        
        fig, ax = plt.subplots(1,figsize=(6.5,4.5))
        self.make_date_axis(ax)
        
        grad = np.gradient(pc_cases)
    #   d1 = np.diff(pc_cases)
    #   print(len(d1))
    #   grad = np.diff(d1)
    #   print(len(grad))

    #   ax.plot(self.get_pdate()[2:],grad)
        ax.plot(self.get_pdate(),grad)
        print(min(grad),max(grad))

        plt.show()

# end of class Geography:
    

def plot_prevalence_comp_TS(flag=None,per_capita=True, mult = 10000, delta_ts=True,
                    window=7, plot_dt = False, cumulative = False,
                    show_order_date = False, 
                    show_superspreader = False,
                    annotation = True, signature = True, 
                    ymaxdefault = None,
                    show_SE = False,
                    low_prev = 0.0,
#                   ymax = [None,None,None], #[0.2,0.01,0.04],
                    save = True, nax = 4):
    """ 
    Plots cases and deaths vs calendar date 

    per_capita: plot numbers per 10000 (see mult)  False
    delta_ts: plot daily new cases True
    window: plot moving agerage window [11]
    plot_dt: plot initial doubling time slopes on log scale False
    annotations: add title and acknowledgements True
    save : save plot as file
    """
    firstDate = mdates.date2num(cv.FirstNYTDate)

    #firstDate = mdates.date2num(datetime.strptime('2021-07-01','%Y-%m-%d'))
    lastDate  = mdates.date2num(cv.EndOfTime)
#   print('GG lastDate:',lastDate)
    orderDate = mdates.date2num(cv.CAOrderDate)
#   print('GG firstDate,lastDate:',firstDate,lastDate)
 
    fig, ax = plt.subplots(nax,1,figsize=(6.5,nax*2.25))
    plt.rcParams['lines.linewidth'] = 1.5

    ylabel = ['Daily Cases','Daily Deaths','Case Fatality Ratio','Vaccinations %']
    if (per_capita):
        for a in range(0,2):
            ylabel[a] = ylabel[a] +'/'+str(mult)
    total_names = ['Cases','Deaths','']
    save_path = cv.graphics_path

#   nyt_counties = pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
    nyt_counties = cv.GeoIndex

    if flag.isnumeric():
        gg_filter = nyt_counties['population'] > float(flag)
    else:
        gg_filter = nyt_counties['flag'].str.contains(flag)
    gg = nyt_counties[gg_filter]


    nG = len(gg)

    cv.EndOfTime = cv.dtime.date()+timedelta(days=7)
    firstDate = cv.FirstNYTDate 
    
    ylim = [(0.0,20.0),
            (0.0,0.6),
            (0.0,0.04),
            (0.0,101.0)]
  
    if flag == 'B':
        ylim[0] = (0.0,30.0)
        ylim[1] = (0.0,0.2)
        ylim[2] = (0.0,0.02)

    for g in range(0,nG):
        print(g,gg['county'].iloc[g],gg['code'].iloc[g],gg['fips'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data('province')
        else:
            tmpG.read_nyt_data('county')
            tmpG.read_vax_data()
 
        do_plot = [True]*4
        if (tmpG.deaths is None):
            cfr = None
            deaths = None
        else:
            cfr = tmpG.deaths/tmpG.cases

        if (per_capita):
            cases  =  mult*tmpG.cases/tmpG.population
            if tmpG.deaths is not None:
                deaths =  mult*tmpG.deaths/tmpG.population
        else:
            cases  =  tmpG.cases
            deaths =  tmpG.deaths
    
        gdf = pd.DataFrame()
        gdf['cases'] = cases
        gdf['deaths'] = deaths
        gdf['cfr'] = cfr

        for a in range(0,3):
            if (any(pd.isna(gdf.iloc[:,a]))):
                do_plot[a] = False
        
        nn = tmpG.ntime-1
        max_cases = cases[nn]

        Date = pd.Series(tmpG.get_pdate())
        df_correction = np.sqrt(window-1)
        for a in range(0,nax):
            GU.make_date_axis(ax[a],firstDate)
            ax[a].set_ylabel(ylabel[a])
            if (do_plot[a]):
                if (a < 2):
                    delta = np.diff(gdf.iloc[:,a]) # first differences
                #   moving average:
                    yvar = pd.Series(delta).rolling(window=window).mean()
                    ax[a].plot(Date[1:],yvar)
                #   print(g,short_name(tmpG.moniker),a,Date.iloc[-1],delta[-1])
                    if show_SE:
                        serr = pd.Series(delta).rolling(window=window).std()/df_correction
                        GU.plot_error(ax[a],Date[1:],yvar,serr,logscale=True,mult=2)
                #   GU.mark_ends(ax[a],Date[1:],yvar,' '+short_name(tmpG.moniker),'r')
#                   if (ymaxdefault is None):
#                   #   ymax[a] = max(ymax[a],1.2*yvar.max())
#                       ymax[a] = max(ymax[a],2*yvar.iloc[-1])

                elif a == 2:
                    ax[a].plot(Date,gdf.iloc[:,a]) 
                    if flag == 'B':
                        GU.mark_ends(ax[a],Date,gdf.iloc[:,a],' '+short_name(tmpG.moniker),'r')
                #   ax[a].set_ylim(ylim[a])

                elif a == 3:
                    vmult = 100.0
                    mdate = tmpG.vax['mdate']
                    pv = vmult*pd.Series(tmpG.vax['first'])/tmpG.population
                    #print('number of vax estimates',len(pv),', max =',pv.max())
                    #print(pv)
                    if pv.max() > 1.0:
                        ax[a].plot(mdate,pv)
                        if flag == 'B':
                            GU.mark_ends(ax[a],mdate,pv,' '+short_name(tmpG.moniker),'r')
                
                    #   pv = vmult*pd.Series(self.vax['full'])/self.population
                    #   ax[a].plot(mdate,pv)
                    #   GU.mark_ends(ax[a],mdate,pv,' full','r')
            # if (do_plot[a]):
        #for a in range(0,nax):


    
    # loop: for g in range(0,len(gg)):

    for a in range(0,nax):
        ax[a].set_ylim(ylim[a]) #ymax[a]))
    
    if low_prev > 0.0 and low_prev < ax[0].get_ylim()[1]:
        ax[0].plot((Date.iloc[0],Date.iloc[-1]),(low_prev,low_prev),
                   color='red',linewidth=1.5,linestyle=':')
        GU.mark_ends(ax[0],pd.Series([Date.iloc[0],Date.iloc[-1]]),
                     pd.Series([low_prev,low_prev]),
                     '$P_{05}$','r')
        GU.mark_ends(ax[0],pd.Series([Date.iloc[0],Date.iloc[-1]]),
                     pd.Series([low_prev,low_prev]),
                     '{:.1f}'.format(low_prev),'l')

    if (annotation):
        if flag == 'B':
            title = 'Covid-19 Prevalence Comparison, SF Bay Area'
        else:
            title = 'Covid-19 Prevalence Comparison ('+str(nG)+' Places)'
        fig.text(0.5,0.5,title ,ha='center',va='top')
        GU.add_data_source(fig, 'New York Times and Centers for Disease Control')

    if (signature):
        GU.add_signature(fig,'johnrsibert@gmail.com')

    if save:
        gfile = cv.graphics_path+'prevalence_comp_TS_'+flag+'.png'
        plt.savefig(gfile,dpi=300)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        print('plot saved as',gfile)
    else:
        plt.show()

    
    #cv.EndOfTime = oldEndOfTime = cv.EndOfTime

def plot_prevalence_comp_histo(flag=None,per_capita=True, mult = 10000, delta_ts=True,
                    window=7, plot_dt = False, cumulative = False,
                    show_order_date = False, 
                    annotation = True, signature = False, 
                    ymaxdefault = None,
                    show_SE = False,
#                   ymax = [None,None,None], #[0.2,0.01,0.04],
                    save = True):
    """ 
    Plots frequency distribtuion of recent a case prevalence 

    window: agerage window for recent prevalence
    annotations: add title and acknowledgements True
    save : save plot as file

    returns quantiles
    """
    #firstDate = mdates.date2num(cv.FirstNYTDate)
    #lastDate  = mdates.date2num(cv.EndOfTime)
    #orderDate = mdates.date2num(cv.CAOrderDate)

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    plt.rcParams['lines.linewidth'] = 1.5

    #save_path = cv.graphics_path

#   nyt_counties = pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
    nyt_counties = cv.GeoIndex

    if flag.isnumeric():
        gg_filter = nyt_counties['population'] > float(flag)
        gg = nyt_counties[gg_filter]
        nG = len(gg)
        file = 'recent_prevalence_histo_pop'
        note = '{0} Regions with population greater than {1:,}'.format(nG,int(flag))
    else:
        gg_filter = nyt_counties['flag'].str.contains(flag)
        gg = nyt_counties[gg_filter]
        file = 'recent_prevalence_histo_'+flag
        note = '{0} Counties'.format(nG)



    if (ymaxdefault is None):
        ymax = [0.0]*3
    else:
        ymax = ymaxdefault

    recent = pd.DataFrame(index=range(0,nG),
                          columns=('county_code','fips','cases','population','sname'))
    print('nG =',nG,'flag =',flag,'recent',recent)

    for g in range(0,nG):
        print(g,gg['county'].iloc[g],gg['code'].iloc[g],gg['fips'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data('province')
        else:
            tmpG.read_nyt_data('county')
            tmpG.read_vax_data()
 

        #Date = pd.Series(tmpG.get_pdate())
        #df_correction = np.sqrt(window-1)
        recent['county_code'][g] = gg['county'][g] +' '+ gg['code'][g] 
    #   recent['code'][g] = gg['code'][g] 
        recent['fips'][g] = gg['fips'][g] 
        recent['population'][g] = tmpG.population
        recent['sname'][g] = short_name(tmpG.moniker)

        delta = pd.Series(np.diff(tmpG.cases)) # first differences
        window_delta = delta.tail(window)
        pd_filter = (window_delta > 0.0)
        window_delta = window_delta[pd_filter]
        recent['cases'][g] = mult*window_delta.sum()/window/tmpG.population

    #   window_len = window #len(window_delta)
    #   
    #   if window_len > 0:
    #       recent['cases'][g] = mult*window_delta.sum()/window_len/tmpG.population
    #   recent['cases'][g] = mult*delta.tail(window).mean()/tmpG.population
    #   if (recent['cases'][g] < 0.0):
    #       print('negative cases:')
    #       print(g,tmpG.moniker)
    #       print(delta.tail(window))
    #       print(pd.DataFrame(tmpG.cases).tail(window))
    #       print('  ')

#   ---- loop: for g in range(0,len(gg)):
    print(recent)
    zmask = recent['cases'] > 0.0

#   select records with cases < 0
    precent = recent[zmask]
    print(precent)

    summary = str(stats.describe(precent['cases']))
    print('recent summary:',summary)           

#   bins = np.linspace(0.0,0.5,50)
    bins = np.linspace(0.0,5.0,50)
    print('bins:',bins)
    nbin = len(bins)
    #weights = np.ones_like(recent['cases']) / len(recent)
    weights = np.ones_like(precent['cases']) / len(precent)

    #hist,edges,patches = ax.hist(recent['cases'],bins,density=False)
    hist,edges,patches = ax.hist(precent['cases'],bins,density=False)
    print(hist)

#   sort recent cases
    precent = precent.sort_values(by='cases',ascending=True)
    precent.to_csv(file+'.csv',index=False)

#   tfig, tax = plt.subplots(1,figsize=(6.5,4.5))
#   tax.scatter(recent['population'],recent['cases'])
#   plt.show()


    table = pd.DataFrame(columns=('rank','county_code','cases'))
    table['county_code'] = precent['county_code']
    table['cases'] = precent['cases']
    table['rank'] = range(0,len(precent))
    print('table')
    print(table)

    html_file = cv.assets_path+file+'.html' 
    with open(html_file,'w') as hh:
        hh.write('<!--- recent summary\n')
        hh.write(summary) 
        hh.write('\n--->\n')
         
        tab = pd.DataFrame(columns=table.columns,dtype='object')
        nreg=20
        tab = tab.append(table.head(nreg),ignore_index=True)
        # insertion of '...' intoduces nan into cases column
        tab = tab.append(pd.Series(['...']*3,index=table.columns),ignore_index=True)
        tab = tab.append(table.tail(nreg),ignore_index=True)
    
        tab['cases'] = pd.to_numeric(tab['cases'],errors='coerce')
        hh.write('<!---START TABLE--->\n')
        # make tabulation string and then replace nan with ...
        tt = tabulate(tab,headers=['Rank','Region','Prevalence'],tablefmt='html',
                      numalign="right", floatfmt=".3f",
                      stralign='left', showindex=False).replace(' nan',' ...')
        hh.write(tt)
        mtime = os.path.getmtime(cv.NYT_home+'us-counties.csv')
        dtime = datetime.fromtimestamp(mtime)

        hh.write('\n<br>\nUpdated '+str(dtime.date())+'\n<br>\n')
        hh.write('<!---END TABLE--->\n')
        print('prevalence rankings saved as',html_file)

    html_file2 ='all_prev.html' 
    with open(html_file2,'w') as h2:
        tt = tabulate(table,headers=['Rank','Region','Prevalence'],tablefmt='html',
                      numalign="right", floatfmt=".3f",
                      stralign='left', showindex=False).replace(' nan',' ...')
        h2.write(tt)
        print('all prevalence rankings saved as',html_file2)
    

    pref = np.quantile(precent['cases'],q=0.05)
    t_filter = (precent['cases'] <= pref) & (precent['cases'] >= 0.0)
    tt = precent[t_filter]

    print('quantiles:')     
    qq = [0.01,0.05,0.10,0.9,0.95,0.99]
    quantiles = pd.Series(index=qq)

    for q in qq:
        quantiles[q] = np.quantile(precent['cases'],q=q)
    print(quantiles)
   

    for k in tt.index:
        print(k,tt['sname'][k],tt['cases'][k])#,tt['sname'][k])
        GU.vline(ax,tt['cases'][k],tt['sname'][k],pos='left')


#   GU.vline(ax,pref,'q=0.1',pos='right')
    ax.axvline(pref,linewidth=3,color='green',alpha=0.5)
    ax.set_xlabel('Mean '+str(window)+' Day Prevalence'+' per '+str(mult))
    ax.set_ylabel('Number of Areas')
    tx = ax.get_xlim()[1]
    ylim = ax.get_ylim()
    ty = ylim[0]+0.90*(ylim[1]-ylim[0])
#   if flag.isnumeric():
#   else:

    print(note)
    ax.text(tx,ty,note,ha='right',va='center',fontsize=10)

    plt.title('Recent Prevalence Frequency')
#   plt.title('Most Recent '+str(window)+' Day Prevalence')

 
    if (signature):
        GU.add_signature(fig,'https://github.com/johnrsibert/SIR-Models')
        GU.add_data_source(fig)

    if save:
        gfile = cv.graphics_path+file+'.png'
        plt.savefig(gfile,dpi=300)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        print('plot saved as',gfile)
    else:
        plt.show()

    return(quantiles)
   
def short_name(s):
    """
    Create 4 byte abbreviation for getography names
    """
    w = re.split(r'[ _-]',s)
    if (len(w)<2):
        sn = s[0:2]+s[-2:]
    else:
        sn = w[0][0]+w[1][0]+s[-2:]
    if (sn == 'DoDC'):
        sn = 'WaDC'
    return(sn)  

def set_moniker(name, code):
    mon = str(name+code)
    return(mon.replace(' ','_',5))
    
def pretty_county(s):
    ls = len(s)
    pretty = s[0:(ls-2)]+', '+s[(ls-2):]
    return(pretty.replace('_',' ',5))



def qcomp(flag=None,per_capita=True, mult = 10000, delta_ts=True, window=7):
    #dtime = datetime.fromtimestamp(mtime)
    #EndOfTime = dtime.date()+timedelta(days=21)
    #firstDate = mdates.date2num(cv.FirstNYTDate)
    #lastDate = mdates.date2num(EndOfTime)
    #print(firstDate, cv.FirstNYTDate)
    #print(lastDate, EndOfTime)
   
    if flag.isnumeric():
        print('   numeric')
        gg_filter = cv.GeoIndex['population'] > float(flag)
        gg = cv.GeoIndex[gg_filter]
        nG = len(gg)
    #   file = 'recent_prevalence_histo_pop'
        note = '{0} Regions with population greater than {1:,}'.format(nG,int(flag))
    else:
        print('   character')
        print(cv.GeoIndex['flag'])
        gg_filter = cv.GeoIndex['flag'].str.contains(flag)
        gg = cv.GeoIndex[gg_filter]
        nG = len(gg) # ??
    #   file = 'recent_prevalence_histo_'+flag
        note = '{0} Counties'.format(nG)
    print(note)
    
    recent = pd.DataFrame(index=range(0,nG),
                          columns=('county_code','fips','population','sname','cases','deaths','cfr','vax'))
    print('nG =',nG,'flag =',flag,'recent:')
    #print(recent)

    for g in range(0,nG):
        print(g,gg['county'].iloc[g],gg['code'].iloc[g],gg['fips'].iloc[g],gg['population'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data('province')
        else:
            tmpG.read_nyt_data('county')
            tmpG.read_vax_data()
            
        tmpG.get_pdate()
      

        #Date = pd.Series(tmpG.get_pdate())
        #df_correction = np.sqrt(window-1)
        recent['county_code'][g] = gg['county'][g] +' '+ gg['code'][g] 
    #   recent['code'][g] = gg['code'][g] 
        recent['fips'][g] = gg['fips'][g] 
        recent['population'][g] = tmpG.population
        recent['sname'][g] = short_name(tmpG.moniker)

        dcases = pd.Series(np.diff(tmpG.cases)) # first differences
#        print(dcases)
        if tmpG.deaths is not None:
            ddeaths = pd.Series(np.diff(tmpG.deaths)) # first differences
        else:
            ddeaths = pd.Series(0.0,index=ddeaths.index)
        #if g == 0:
        #    print(tmpG.enclosed_by,list(pd.Series(tmpG.pdate).tail(10)))
        #if tmpG.enclosed_by == 'Florida':
        #    print(tmpG.to_DataFrame().tail(window))
        #    
        #    #sys.exit(1)
        
#        window_delta = delta.tail(window)
#        pd_filter = (window_delta > 0.0)
#        window_delta = window_delta[pd_filter]
#        recent['cases'][g] = mult*window_delta.sum()/window/tmpG.population
        
        #print(tmpG.deaths)
        tcases = pd.Series(tmpG.cases)
        if tmpG.deaths is not None:
            tdeaths = pd.Series(tmpG.deaths)
        else:
            tdeaths = pd.Series(0.0,index=tcases.index)        #print(tmpG.vax)
        #tmpG.vax = pd.Series(tmpG.vax)
        #print(tmpG.vax.tail(window))
        
        den = window*tmpG.population/10000.0
        recent['cases'][g] = dcases.tail(window).sum()/den
        if tmpG.deaths is not None:
   #     print('deaths')
            recent['deaths'][g] = ddeaths.tail(window).sum()/den
   #     print('cfr')
            recent['cfr'][g] = pd.Series(tmpG.deaths).tail(window).sum()/pd.Series(tmpG.cases).tail(window).sum()
        if tmpG.vax is not None:   #     print('vax')
            recent['vax'][g] = tmpG.vax['full'].tail(window).mean()/tmpG.population
        

    recent.to_csv('recent.csv',index=False)
    '''
    fig, ax = plt.subplots(2,figsize=(6.5,6.5)) 
    ax[0].scatter(recent['vax'],recent['cases'])
    ax[0].set_ylabel('Cases per 10k')
    ax[0].set_xlabel('% Vaccinated')
    
    ax[1].scatter(recent['vax'],recent['deaths'])
    ax[1].set_xlabel('% Vaccinated')
    ax[1].set_ylabel('Deaths per 10k')
    plt.show()
    '''
    
    vnames = ['cases','deaths','cfr','vax']
    qq = [0.01,0.05,0.1,0.2,0.8,0.9,0.95,0.99]
    quantiles = pd.DataFrame(index=qq, columns=vnames)
    #print(quantiles)

    for v in vnames:
        for q in qq:
        #    print(v,q,np.quantile(recent[v],q=q))
            quantiles[v][q] = np.quantile(recent[v],q=q)

    quantiles['vax'].fillna(0.999,inplace=True)
    quantiles['vax'] = 100.0*quantiles['vax'] 
    #print(quantiles)
    return quantiles
