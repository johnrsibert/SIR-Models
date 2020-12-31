from covid21 import config as cv
from covid21 import GraphicsUtilities as GU

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

class Geography:

#   def __init__(self,name,enclosed_by,code):
    def __init__(self,**kwargs):
        self.gtype = None
        self.name = kwargs.get('name')
        self.enclosed_by = kwargs.get('enclosed_by')
        self.code = kwargs.get('code')
        self.population = None
        self.source = None
        if self.name is None:
            self.moniker = None
        else:
            self.moniker = str(self.name+self.code)
            self.moniker =  self.moniker.replace(' ','_',5) 
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
        
    def print(self):
        print(self)
        
        
    def print_metadata(self):
        print('gtype:',self.gtype)
        print('gname:',self.name)
        print('genclosed_by:',self.enclosed_by)
        print('gcode:',self.code)
        print('gpopulation:',self.population)
        print('gsource:',self.source)
        print('gmoniker:',self.moniker)
        print('gdat_file:',self.dat_file)
        print('gupdated:',self.updated)
        print('gdate0:',self.date0)
        print('gntime:',self.ntime)

    def print_data(self):
        self.get_pdate()
    #   if (self.deaths != None):
        if (len(self.deaths) > 0):
            print('date','cases','deaths','pdate')
        else:
            print('date','cases','pdate')
        for k in range(0,len(self.date)):
        #   if (self.deaths != None):
            if (len(self.deaths) > 0):
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
        if (self.pdate != None):
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
    #   if (dat.empty):
    #       print('Reading',census_data_path)
    #       cv.population_dat = pd.read_csv(census_data_path,header=0,comment='#')
    #       dat = cv.population_dat
    #   else:
    #       print('Using current "dat" object')

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
        return(population)   
        
    def read_nyt_data(self,gtype='county'):
        self.gtype = gtype
        mtime = 0.0
        if (gtype == 'county'):
            dat = cv.nyt_county_dat
            if (dat.empty):
               cspath = NYT_counties
               print('Reading',cspath)
               cv.nyt_county_dat = pd.read_csv(cspath,header=0)
               dat = cv.nyt_county_dat
               mtime = os.path.getmtime(cspath)
        else:
            sys.exit('No data for',gtype)
        
    #   dat = pd.read_csv(cspath,header=0)

    #   mtime = os.path.getmtime(cspath)
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

    def read_BCHA_data(self,gtype='hsa'):
        self.gtype = gtype
        cspath = BCHA_path

        dat = pd.read_csv(cspath,header=0)

        mtime = os.path.getmtime(cspath)
        dtime = datetime.fromtimestamp(mtime)
        self.updated = str(dtime.date())
        self.population = None
   #    self.source = 'British Columbia Centre for Disease Control'\
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
                        annotation = True, signature = False, 
                        save = True, dashboard = False, nax = 3):
        
        """ 
        Plots cases and deaths vs calendar date 

        scale: select linear of log scale 'linear'
        per_capita: plot numbers per 1000 (see mult)  False
        delta_ts: plot daily new cases True
        window: plot moving agerage window [11]
        plot_dt: plot initial doubling time slopes on log scale False
        annotations: add title and acknowledgements True
        save : save plot as file
        """
        mult = 1000

        firstDate = datetime.strptime(self.date[0],'%Y-%m-%d')
        orderDate = mdates.date2num(cv.CAOrderDate)

        if (self.deaths is None):
            nax = 1
    
        ax = [None]*nax
        fig, pax = plt.subplots(nax,1,figsize=(6.5,nax*2.25))
        ax = [None]*nax
        if (nax == 1):
            ax[0] = pax
        else:
            ax=pax

        ylabel = ['Daily Cases','Daily Deaths','Case Fatality Ratio']
        end_marks = [r'$\Sigma$C','$\Sigma$D','']
        total_names = ['Cases','Deaths','']
        gdf = pd.DataFrame()

        cfr = self.deaths/self.cases

        if (per_capita):
            for a in range(0,2):
                ylabel[a] = ylabel[a] +' per '+str(mult)
        
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

        ylim = [(0.0,1.2*gdf.iloc[:,0].max()),
                (0.0,1.2*gdf.iloc[:,1].max()),
                (0.0,1.2*gdf.iloc[:,2].max())]

        nn = self.ntime-1
        max_cases = cases[nn]
        if (self.deaths is None):
            max_deaths = 0.0
        else:
            max_deaths = deaths[nn]

        ax2 = []
        for a in range(0,nax):
            GU.make_date_axis(ax[a],firstDate)
            if (a < 2):
                ax[a].set_yscale(yscale)
            ax[a].set_ylabel(ylabel[a])
            if (cumulative) & (a < 2):
                ax2.append(ax[a].twinx())
                ax2[a].set_ylabel('Cumulative')

        Date = self.get_pdate()

        for a in range(0,nax):
            if (a < 2):
                delta = np.diff(gdf.iloc[:,a]) #cases)
                ax[a].bar(Date[1:], delta)
                for w in range(0,len(window)):
                #   moving average
                    adc = pd.Series(delta).rolling(window=window[w]).mean()
                    ax[a].plot(Date[1:],adc,linewidth=2)
                    GU.mark_ends(ax[a],Date[1:],adc, str(window[w])+'da','r')

                ylim[a] = (0.0,1.2*adc.max())
                tx = GU.prop_scale(ax[a].get_xlim(), 0.5)
                ty = GU.prop_scale(ax[a].get_ylim(), 0.95)
                note = '{0:,} {1}'.format(int(gdf.iloc[-1,a]),total_names[a])
                ax[a].text(tx,ty,note ,ha='center',va='top',fontsize=10)

                if (cumulative):
                    ax2[a].plot(Date, gdf.iloc[:,a], alpha=0.5, linewidth=1)#,label=cc)
                    GU.mark_ends(ax2[a],Date,gdf.iloc[:,a] ,end_marks[a],'r')
                    ax2[a].set_ylim(0.0,1.2*gdf.iloc[-1,a])

            else:
                ax[a].plot(Date,gdf.iloc[:,a]) 
                ax[a].set_ylim(ylim[a])

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
            GU.add_data_source(fig,self.source)

        if (signature):
            GU.add_signature(fig,'https://github.com/johnrsibert/SIR-Models/tree/master/PlotsToShare')

        if (dashboard):
        #   out_img = BytesIO()
        #   plt.savefig(out_img, format='png')
        #   out_img.seek(0)  # rewind file
        #   encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        #   return "data:image/png;base64,{}".format(encoded)
            gfile = graphics_path+'test.png'
            
            print('saving fig for dash:',gfile)
            plt.savefig(gfile,format='png',dpi=300)
            plt.close()
            print('saved')

        else:
            if save:
                graphics_path = '../'
                gfile = graphics_path+self.moniker+'_prevalence.png'
                plt.savefig(gfile,dpi=300)
                plt.show(block=False)
                plt.pause(3)
                plt.close()
                print('plot saved as',gfile)
            else:
                plt.show()

    def plot_per_capita_curvature(self,mult = 1000,save=False):
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


