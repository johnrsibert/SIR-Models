SIR_path = '/home/jsibert/Projects/SIR-Models/'
fit_path = paste(SIR_path,'fits/',sep='')
dat_path = paste(SIR_path,'dat/',sep='')
TMB_path = paste(SIR_path,'TMB/',sep='')
source(paste(TMB_path,'plot_state.R',sep=''))
source(paste(TMB_path,'SIR_read_dat.R',sep=''))
source(paste(TMB_path,'fit_to_df.R',sep=''))
require(TMB)
require(gtools)


do_one_run = function(County = "Santa Clara",model.name = 'simpleSIR4',do.plot=TRUE)
{
cname = sub(" ","_",County)
datfile=paste(dat_path,cname,'.dat',sep='')
separator = "#############################"
print(paste(separator,County,separator),quote=FALSE)
dat = read.dat.file(datfile)
eps = 1e-8
data=dat$data
print(names(data))
print(data)
data$log_obs_cases = log(data$obs_cases+eps)
data$log_obs_deaths = log(data$obs_deaths+eps)
print("-data:")
print(data)

init = list(
    logsigma_logP = log(0.2),
    logsigma_logbeta = 0.4, #log(0.7),
    logsigma_logmu = 0.2, #log(2.0),
    logmu = log(0.00005),
    logsigma_logC = log(log(1.25)),
    logsigma_logD = log(log(1.1)),
    logbeta = log(0.05)
)
print("--init parameter values:")
print(init)

par = list(
    logsigma_logP = init$logsigma_logP,
    logsigma_logbeta = init$logsigma_logbeta,
    logsigma_logmu = init$logsigma_logmu,
    logmu    = rep(init$logmu,data$ntime+1),
    logsigma_logC = init$logsigma_logC,
    logsigma_logD = init$logsigma_logD,
    logbeta = rep(init$logbeta,(data$ntime+1))
)
print(paste("---initial model parameters: ", length(par)))
print(par)

map = list(
           "logsigma_logP" = as.factor(1),
           "logsigma_logbeta" = as.factor(1),
           "logsigma_logmu" = as.factor(1),
           "logsigma_logC" = as.factor(NA),
           "logsigma_logD" = as.factor(NA)
)

print(paste("---- estimation map:",length(map),"variables"))
print(map)

cpp.name = paste(model.name,'.cpp',sep='')

print(paste("Compiling",cpp.name,"-------------------------"),quote=FALSE)
compile(cpp.name)
print(paste("Loading",model.name,"-------------------------"),quote=FALSE)
dyn.load(dynlib(model.name))
print("Finished compilation and dyn.load-------------",quote=FALSE)
print("Calling MakeADFun-----------------------------",quote=FALSE)
obj = MakeADFun(data,par,random=c("logbeta","logmu"), 
                map=map,DLL=model.name)
print("--------MakeADFun Finished--------------------",quote=FALSE)
print("obj$par (1):")
print(obj$par)
lb <- obj$par*0-Inf
ub <- obj$par*0+Inf

print("Starting minimization-------------------------",quote=FALSE)
opt = nlminb(obj$par,obj$fn,obj$gr)
#opt = optim(obj$par,obj$fn,obj$gr,method="BFGS")
#opt = optim(obj$par,obj$fn,obj$gr) #,control=list(maxit=1000))
#opt = optim(obj$par,obj$fn,obj$gr,method="L-BFGS-B",arg="L-BFGS-B")

print("Done minimization-----------------------------",quote=FALSE)
print(paste("Objective function value =",opt$objective))
print(paste("Objective function value =",opt$value))
print(paste("Convergence ",opt$convergence))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
print("parameters:",quote=FALSE)
print(opt$par)
print(exp(opt$par))

mlogbeta = median(obj$report()$logbeta)
print(paste("median logbeta:",mlogbeta))
mlogmu = median(obj$report()$logmu)
print(paste("median logmu:",mlogmu))

if (do.plot){
    x11()
    plot.log.state(data,par,obj,opt,map,np=4)
    dev.file = paste(fit_path,data$county,'.pdf',sep='')
    dev.copy2pdf(file=dev.file,width=6.5,height=6)
    dev.off()
}

fit = list(data=data,map=map,par=par,obj=obj,opt=opt,init=init)
#rd.file = paste(fit_path,data$county,'.RData',sep='')
#save.fit(data,obj,opt,map,init,rd.file)
save.fit(fit,file=data$county)#   rd.file) #"t.RData")

return(fit)

} # do_one_run = function(


county_list = list("Alameda", "Contra_Costa", "San_Francisco", "San_Mateo",
                    "Santa_Clara")
CA_county_list = list("Alameda", "Contra_Costa", "Los_Angeles", "Marin",
                       "Napa", "Orange", "Riverside", "Sacramento",
                       "San_Bernardino", "San_Diego", "San_Francisco",
                       "San_Mateo", "Santa_Clara", "Sonoma")

big_county_list = list(
                       "New_York_City",
                       "Los_Angeles","San_Diego",
                       "Orange", "Riverside",
                       "San_Bernardino","Santa_Clara",
                       "Alameda",
                       "Sacramento","Contra_Costa","Fresno", "Kern",
                       "San_Francisco",
                       "Ventura","San_Mateo","San_Joaquin",
                       "Stanislaus","Sonoma","Marin")

largest_us_counties = list(
"AlamedaCA",
"BexarTX",
"BrowardFL",
"ClarkNV",
"Contra_CostaCA",
"CookIL",
"DallasTX",
"FranklinOH",
"HarrisTX",
"HennepinMN",
"HillsboroughFL",
"KingWA",
"Los_AngelesCA",
"MaricopaAZ",
"Miami-DadeFL",
"MiddlesexMA",
"NassauNY",
"New_York_CityNY",
"OaklandMI",
"OrangeCA",
"OrangeFL",
"Palm_BeachFL",
"PhiladelphiaPA",
"RiversideCA",
"SacramentoCA",
"San_BernardinoCA",
"San_DiegoCA",
"San_FranciscoCA",
"San_MateoCA",
"Santa_ClaraCA",
"TarrantTX",
"TompkinsNY",
"TravisTX",
"WayneMI"
)

fit_examples = list(
"New_York_CityNY",
"Miami-DadeFL", "BrowardFL", 
"Palm_BeachFL","HillsboroughFL","NassauNY",
"HarrisTX",
"DallasTX",
"TarrantTX",
"BexarTX",
"TravisTX",
"MaricopaAZ",
"PhiladelphiaPA",
"TravisTX",
"HonoluluHI",
"CookIL",
"AlamedaCA"
)
                       
nrun = 1
if (nrun < 2) {
#   do_one_run(County="Los_AngelesCA")->fit
#   do_one_run(County="AlamedaCA")->fit
#   do_one_run(County="HonoluluHI")->fit
    do_one_run(County="NassauNY")->fit
#   do_one_run(County="BrowardFL")->fit
} else {
   sink( paste(fit_path,'SIR_model.log',sep=''), type = c("output", "message"))
   dp =paste(dat_path,'*.dat',sep='')
   print(dp)
   print(paste('globbing',dp))
   cc = Sys.glob(dp)
   for (c in cc)
   {
       County = sub("\\.dat","",c)
       c = basename(County)
       print(paste("starting",c))
       do_one_run(County=c,do.plot=TRUE)->fit
   }
   sink()
}

