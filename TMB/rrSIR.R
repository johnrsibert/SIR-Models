SIR_path = '/home/jsibert/Projects/SIR-Models/'
fit_path = paste(SIR_path,'fits/',sep='')
dat_path = paste(SIR_path,'dat/',sep='')
TMB_path = paste(SIR_path,'TMB/',sep='')
source(paste(TMB_path,'plot_rrSIR.R',sep=''))
source(paste(TMB_path,'SIR_read_dat.R',sep=''))
source(paste(TMB_path,'fit_to_df.R',sep=''))
require(TMB)
require(gtools)
print('starting')


do_one_run = function(County = "Santa Clara",model.name = 'rrSIR',do.plot=TRUE)
{
separator = "#############################"
print(separator)

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
    logsigma_logP = log(0.1),

    logsigma_logbeta = log(0.2),
    logsigma_logmu = log(0.2),

    bias_logbeta = 0.0,
    bias_logmu = 0.0,

    logprop_immune = log(1.0),

    logsigma_logC = log(0.223),
    logsigma_logD = log(0.105),

    logbeta = -0.4, #log(0.05),
    logmu = -7.0, #log(0.05),

    priorlogmu = -4.0,
    sigma_priorlogmu = 0.8,

    logsigma_loggamma = log(0.2),
    loggamma = log(0.5) 
)
print("--init parameter values:")
print(init)

par = list(
    logsigma_logP = init$logsigma_logP,

    logsigma_logbeta = init$logsigma_logbeta,
    logsigma_loggamma = init$logsigma_loggamma,
    logsigma_logmu = init$logsigma_logmu,

    bias_logbeta = init$bias_logbeta,
#   bias_loggamma = init$bias_loggamma,
    bias_logmu = init$bias_logmu,

    logprop_immune = init$logprop_immune,

    logsigma_logC = init$logsigma_logC,
    logsigma_logD = init$logsigma_logD,

    logbeta  = rep(init$logbeta,(data$ntime+1)),
    loggamma = rep(init$loggamma,data$ntime+1),
    logmu    = rep(init$logmu,data$ntime+1),

    priorlogmu = init$priorlogmu,
    sigma_priorlogmu = init$sigma_priorlogmu
)
print(paste("---initial model parameters: ", length(par)))
print(par)

map = list(
           "loggamma" = rep(as.factor(NA),data$ntime+1),

           "logsigma_logP" = as.factor(1),
           "logsigma_logbeta" = as.factor(1),
           "logsigma_loggamma" = as.factor(1),
           "logsigma_logmu" = as.factor(1),

           "bias_logbeta" = as.factor(NA),
#          "bias_loggamma" = as.factor(NA),
           "bias_logmu" = as.factor(NA),

           "logprop_immune" = as.factor(NA),

           "logsigma_logC" = as.factor(NA),
           "logsigma_logD" = as.factor(NA),

           
            "priorlogmu" = as.factor(NA),
            "sigma_priorlogmu" = as.factor(NA)
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
#opt = nlminb(obj$par,obj$fn,obj$gr)
opt = optim(obj$par,obj$fn,obj$gr)

print("Done minimization-----------------------------",quote=FALSE)
print(paste("Function objective =",opt$objective))
print(paste("Function value =",opt$value))
print(paste("Convergence ",opt$convergence))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
print("parameters:",quote=FALSE)
print(opt$par)
print("exp(parameters):",quote=FALSE)
print(exp(opt$par))

mlogbeta = median(obj$report()$logbeta)
print(paste("median logbeta:",mlogbeta))
mlogmu = median(obj$report()$logmu)
print(paste("median logmu:",mlogmu))
mgamma = median(obj$report()$gamma)
print(paste("median gamma:",mgamma))

#print('data')
#print(data)
#print('map')
#print(map)
#print('par')
#print(par)
#print('obj')
#print(obj)
#print('opt')
#print(opt)
#print('init')
#print(init)
#print(model.name)

fit = list(dat=data,map=map,par=par,obj=obj,opt=opt,init=init,
           model.name=model.name)
if (do.plot){
#   x11()
    plot.log.state(fit)
#   dev.file = paste(fit_path,data$county,'.pdf',sep='')
    dev.file = paste(data$county,'_ests.pdf',sep='')
    dev.copy2pdf(file=dev.file,width=6.5,height=9)
#   dev.off()
}

#rd.file = paste(fit_path,data$county,'.RData',sep='')
#save.fit(data,obj,opt,map,init,rd.file)
#save.fit(fit,file=data$county)#   rd.file) #"t.RData")

return(fit)

} # do_one_run = function(County = ...)


nrun = 1
if (nrun < 2) {
#   sink('test.log', type = c("output", "message"))
#   do_one_run(County="Los_AngelesCA")->fit
#   do_one_run(County="AlamedaCA")->fit
#   do_one_run(County="HonoluluHI")->fit
#   do_one_run(County="NassauNY",do.plot=TRUE)->fit
    do_one_run(County="Miami-DadeFL",do.plot=TRUE)->fit
#   sink()
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

