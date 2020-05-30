source('plot_state.R')
source('SIR_read_dat.R')
source('fit_to_df.R')
fit_path = '/home/jsibert/Projects/SIR-Models/fits/'
require(TMB)


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
print("-data:")
print(data)

init = list(
    sigma_logP = 0.1,
    sigma_logbeta = 0.05,
    sigma_logmu = 0.01,
    logmu = log(0.005),
    loggamma = log(0.05),
    sigma_logC = log(0.25),
    sigma_logD = log(0.25),
    logbeta = log(0.1)
)
print("--initial parameter values:")
print(init)

par = list(
    sigma_logP = init$sigma_logP,
    sigma_logbeta = init$sigma_logbeta,
    sigma_logmu = init$sigma_logmu,
    logmu    = rep(init$logmu,data$ntime),
    loggamma = init$loggamma,
    sigma_logC = init$sigma_logC,
    sigma_logD = init$sigma_logD,
    logbeta = rep(init$logbeta,data$ntime)
)
print(paste("---model parameters: ", length(par)))
print(par)

map = list(
           "sigma_logP" = as.factor(1),
           "sigma_logbeta" = as.factor(1),
           "sigma_logmu" = as.factor(NA),
           "loggamma"  = as.factor(1),
           "sigma_logC" = as.factor(1),
           "sigma_logD" = as.factor(1),
           "sigma_logbeta" = as.factor(1)
)
#          "logmu"  = as.factor(1),
#          "logbeta" = rep(factor(1),data$ntime))

print(paste("---- estimation map:",length(map),"variables"))
print(map)

cpp.name = paste(model.name,'.cpp',sep='')

print(paste("Compiling",cpp.name,"-------------------------"),quote=FALSE)
compile(cpp.name)
dyn.load(dynlib(model.name))
print("Finished compilation and dyn.load-------------",quote=FALSE)
print("Calling MakeADFun-----------------------------",quote=FALSE)
obj = MakeADFun(data,par,random=c("logbeta","logmu"), 
                map=map,DLL=model.name)
print("--------MakeADFun Finished--------------------",quote=FALSE)
print("obj$par (1):")
print(obj$par)
lower <- obj$par*0-Inf
upper <- obj$par*0+Inf
lower["sigma_logC"] =  0.0
lower["sigma_logD"] =  0.0
#lower["loggamma"] = -8.0

print("Starting minimization-------------------------",quote=FALSE)
options(warn=2,verbose=FALSE)
obj$control=list(eval.max=500,iter.max=10)
 opt = nlminb(obj$par,obj$fn,obj$gr)#,lower=lower)#,upper=upper)
#opt = optim(obj$par,obj$fn,obj$gr,method="BFGS")
#opt = optim(obj$par,obj$fn,obj$gr,method="L-BFGS-B",arg="L-BFGS-B",lower=lower)

print("Done minimization-----------------------------",quote=FALSE)
print(paste("Objective function value =",opt$objective))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
print("parameters:",quote=FALSE)
print(opt$par)
print(exp(opt$par))
gmbeta = exp(mean(obj$report()$logbeta))
print(paste("geometric mean beta:",gmbeta))
gmmu = exp(mean(obj$report()$logmu))
print(paste("geometric mean mu:",gmmu))

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

big_county_list = list("New_York_City","Los_Angeles","San_Diego",
                       "Riverside","San_Bernardino","Santa_Clara","Alameda",
                       "Sacramento","Contra_Costa","Fresno","Kern",
                       "San_Francisco","Ventura","San_Mateo","San_Joaquin",
                       "Stanislaus","Sonoma","Marin","Orange")
                       
nrun = 1
if (nrun < 2) {
    do_one_run(County='Alameda')->fit
} else {
   sink( paste(fit_path,'SIR_model.log',sep=''), type = c("output", "message"))
   for (c in 1:length(big_county_list))
   {
       print(paste('starting',big_county_list[c]))
       do_one_run(County=big_county_list[c])->junk
       print(paste('finished',big_county_list[c]))
   }
   sink()
}

