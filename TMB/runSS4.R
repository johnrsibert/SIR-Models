source('plot_state.R')
source('SIR_read_dat.R')
source('fit_to_df.R')
fit_path = '/home/jsibert/Projects/SIR-Models/fits/'
require(TMB)
require(gtools)


do_one_run = function(County = "Santa Clara",model.name = 'simpleSIR4')
{
cname = sub(" ","_",County)
datfile=paste(fit_path,cname,'.dat',sep='')
separator = "#############################"
print(paste(separator,County,separator),quote=FALSE)
print(paste(separator,County,separator),quote=FALSE)
print(paste(separator,County,separator),quote=FALSE)
dat = read.dat.file(datfile)
eps = 1e-8
data=dat$data
print(names(data))
print(data)
data$log_obs_cases = log(data$obs_cases+eps)
data$log_obs_deaths = log(data$obs_deaths+eps)
data$beta_a = eps
data$beta_b = 2.0
data$mu_a = eps
data$mu_b = 2.0
print("-data:")
print(data)

init = list(
    logsigma_logP = log(0.2),
    logsigma_beta = log(0.02),
    logsigma_mu = log(0.001),
    logitmu = logit(0.001,data$mu_a,data$mu_b),
    loggamma = log(0.001),
    logsigma_logC = log(0.25),
    logsigma_logD = log(0.25),
    logitbeta = logit(0.1,data$beta_a,data$beta_b)
)
print("--initial parameter values:")
print(init)

par = list(
    logsigma_logP = init$logsigma_logP,
    logsigma_beta = init$logsigma_beta,
    logsigma_mu = init$logsigma_mu,
    logitmu    = rep(init$logitmu,data$ntime+1),
    loggamma = init$loggamma,
    logsigma_logC = init$logsigma_logC,
    logsigma_logD = init$logsigma_logD,
    logitbeta = rep(init$logitbeta,(data$ntime+1))
)
print(paste("---model parameters: ", length(par)))
print(par)

map = list(
           "logsigma_logP" = as.factor(1),
           "logsigma_beta" = as.factor(1),
           "logsigma_mu" = as.factor(1),
           "loggamma"  = as.factor(1),
           "logsigma_logC" = as.factor(1),
           "logsigma_logD" = as.factor(1)
)
#          "logitmu" = rep(as.factor(NA),data$ntime),

print(paste("---- estimation map:",length(map),"variables"))
print(map)

cpp.name = paste(model.name,'.cpp',sep='')

print(paste("Compiling",cpp.name,"-------------------------"),quote=FALSE)
compile(cpp.name)
dyn.load(dynlib(model.name))
print("Finished compilation and dyn.load-------------",quote=FALSE)
print("Calling MakeADFun-----------------------------",quote=FALSE)
obj = MakeADFun(data,par,random=c("logitbeta","logitmu"), 
                map=map,DLL=model.name)
print("--------MakeADFun Finished--------------------",quote=FALSE)
print("obj$par (1):")
print(obj$par)
lb <- obj$par*0-Inf
ub <- obj$par*0+Inf
#lb["sigma_logD"] =  0.0
#lb["sigma_logD"] =  0.0
#lb["loggamma"] = -8.0

print("Starting minimization-------------------------",quote=FALSE)
options(warn=2,verbose=FALSE)
#obj$control=list(eval.max=500,iter.max=10)
#opt = nlminb(obj$par,obj$fn,obj$gr,
#    control=list('trace'=1,'abs.tol'=1e-3,'rel.tol'=1e-3))
#opt = optim(obj$par,obj$fn,obj$gr,method="BFGS")
 opt = optim(obj$par,obj$fn,obj$gr)
#opt = optim(obj$par,obj$fn,obj$gr,method="L-BFGS-B",arg="L-BFGS-B")

print("Done minimization-----------------------------",quote=FALSE)
print(paste("Objective function value =",opt$objective))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
print("parameters:",quote=FALSE)
print(opt$par)
print(exp(opt$par))
mbeta = mean(obj$report()$beta)
print(paste("mean beta:",mbeta))
mmu = mean(obj$report()$mu)
print(paste("mean mu:",mmu))

plot.log.state(data,par,obj,opt,map,np=4)
dev.file = paste(fit_path,data$county,'.pdf',sep='')
dev.copy2pdf(file=dev.file,width=6.5,height=6)

rd.file = paste(fit_path,data$county,'.RData',sep='')
save.fit(data,obj,opt,map,init,rd.file)

return(list(data=data,map=map,par=par,obj=obj,opt=opt,init=init))

} # do_one_run = function((County = "Santa Clara",model.name = 'simpleSIR4')


county_list = list("Alameda", "Contra_Costa", "San_Francisco", "San_Mateo",
                    "Santa_Clara")
CA_county_list = list("Alameda", "Contra_Costa", "Los_Angeles", "Marin",
                       "Napa", "Orange", "Riverside", "Sacramento",
                       "San_Bernardino", "San_Diego", "San_Francisco",
                       "San_Mateo", "Santa_Clara", "Sonoma")

big_county_list = list(
                      #"New_York_City",
                       "Los_Angeles","San_Diego",
                       "Orange", "Riverside",
                       "San_Bernardino","Santa_Clara",
                       "Alameda",
                       "Sacramento","Contra_Costa","Fresno", "Kern",
                       "San_Francisco",
                       "Ventura","San_Mateo","San_Joaquin",
                       "Stanislaus","Sonoma","Marin")

largest_us_counties = list(
"AlamedaCA", "BexarTX",
"BrowardFL", "ClarkNV", 
"Contra_CostaCA",
"CookIL", "DallasTX", "FresnoCA", "HarrisTX", "KernCA",
"KingWA", 
"Los_AngelesCA", 
"MaricopaAZ", "MarinCA", 
"Miami-DadeFL",
"New_York_CityNY", 
"OrangeCA", "RiversideCA", "SacramentoCA",
"San_BernardinoCA", 
"San_DiegoCA", 
"San_FranciscoCA", 
"San_JoaquinCA",
"San_MateoCA",
"Santa_ClaraCA", 
"SonomaCA", "StanislausCA",
"SuffolkMA", "TarrantTX", "VenturaCA", "WayneMI")
                       
nrun = 2
if (nrun < 2) {
    do_one_run(County="AlamedaCA")->fit
#   do_one_run(County="New_York_CityNY")->fit
#   do_one_run(County="Contra_CostaCA")->fit
} else {
   sink( paste(fit_path,'SIR_model.log',sep=''), type = c("output", "message"))
   for (c in 1:length(largest_us_counties))
   {
       print(paste('starting',largest_us_counties[c]))
       do_one_run(County=largest_us_counties[c])->junk
       print(paste('finished',largest_us_counties[c]))
   }
   sink()
}

