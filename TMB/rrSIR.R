SIR_path = '/home/jsibert/Projects/SIR-Models/'
fit_path = paste(SIR_path,'fits/',sep='')
dat_path = paste(SIR_path,'dat/',sep='')
TMB_path = paste(SIR_path,'TMB/',sep='')
source(paste(TMB_path,'plot_rrSIR.R',sep=''))
source(paste(TMB_path,'SIR_read_dat.R',sep=''))
source(paste(TMB_path,'fit_to_df.R',sep=''))
require(TMB)
require(gtools)
rm(fit)
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
data$obs_R = vector(len=data$ntime,mode="numeric")
data$obs_R[1] = 0.0
for (t in 2:(data$ntime+1))
{
    data$obs_R[t] = data$obs_cases[t-1] - data$obs_deaths[t-1]
}
data$log_obs_R = log(data$obs_R+eps)

print("-data:")
print(data)

#                       init       est  map
# names                                    
# logsigma_logCP   -2.120264 -2.690276    1
# logsigma_logDP   -2.343407 -3.267401    1
# logsigma_logC    -1.499940 -1.499940  NaN
# logsigma_logD    -2.350619 -2.350619  NaN
# logsigma_logbeta  0.400000 -4.838245    1
# logsigma_logmu    0.200000 -1.977164    1
# loggamma         -0.051293 -0.051293  NaN
# def SIRsim(mu = 0.001, beta = 0.2, gamma = 0.075, N0 = 2e6, sigma_eye = 0.01,
 

init = list(
    logsigma_logP = log(0.223),# 3.0, #log(0.1),#logsigma_logP = 0.5,

    logsigma_logbeta = log(0.223), #0.4,
    logsigma_loggamma = log(0.223), #0.4,
    logsigma_logmu = log(0.223), #0.4,-2.0,


    logsigma_logC = log(0.223),
    logsigma_logD = log(0.105),
    logsigma_logR = log(0.105),

    logbeta  = log(0.2),
    loggamma =  log(0.075),
    logmu    = log(0.00025)
)
print("--init parameter values:")
print(init)

par = list(
    logsigma_logP = init$logsigma_logP,

    logsigma_logbeta = init$logsigma_logbeta,
    logsigma_loggamma = init$logsigma_loggamma,
    logsigma_logmu = init$logsigma_logmu,

    logsigma_logC   = init$logsigma_logC,
    logsigma_logD   = init$logsigma_logD,
    logsigma_logR   = init$logsigma_logR,

    logbeta  = rep(init$logbeta,(data$ntime+1)),
    loggamma = rep(init$loggamma,data$ntime+1),
    logmu    = rep(init$logmu,data$ntime+1) 
)

print(paste("---initial model parameters: ", length(par)))
print(par)

map = list(
           "logsigma_logP" = as.factor(1),

           "logsigma_logbeta" = as.factor(1),
           "logsigma_loggamma" = as.factor(1),
           "logsigma_logmu" = as.factor(1),

           "logsigma_logC" = as.factor(NA),
           "logsigma_logD" = as.factor(NA),
           "logsigma_logR" = as.factor(NA) 
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
obj = MakeADFun(data,par, random=c("logbeta","logmu"), 
                map=map,DLL=model.name)
print("--------MakeADFun Finished--------------------",quote=FALSE)
print("obj$par (1):")
print(obj$par)
hide_obj = obj
lb <- obj$par*0-Inf
ub <- obj$par*0+Inf

#   cmd = 'Rscript --verbose simpleSIR4.R'

obj$env$inner.control$tol10 <- 0

print("Starting minimization-------------------------",quote=FALSE)
opt = nlminb(obj$par,obj$fn,obj$gr)
#opt = optim(obj$par,obj$fn,obj$gr)

print("Done minimization-----------------------------",quote=FALSE)
print(paste("Function objective =",opt$objective))
print(paste("Function value =",opt$value))
print(paste("Convergence ",opt$convergence))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
#print("parameters:",quote=FALSE)
#print(opt$par)
#print("exp(parameters):",quote=FALSE)
#print(exp(opt$par))

#mlogbeta = median(obj$report()$logbeta)
#print(paste("median logbeta:",mlogbeta,exp(mlogbeta)))
#mlogmu = median(obj$report()$logmu)
#print(paste("median logmu:",mlogmu,exp(mlogmu)))
#mloggamma = median(obj$report()$loggamma)
#print(paste("median loggamma:",mloggamma,exp(mloggamma)))

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
#   print(fit)
    plot.log.state(fit)
#   dev.file = paste(fit_path,data$county,'.pdf',sep='')
    dev.file = paste(data$county,'_ests.pdf',sep='')
    dev.copy2pdf(file=dev.file,width=6.5,height=9)
#   dev.off()
}

#rd.file = paste(fit_path,data$county,'.RData',sep='')
#save.fit(data,obj,opt,map,init,rd.file)
#save.fit(fit,file="t.RData")

return(fit)

} # do_one_run = function(County = ...)


nrun = 1
if (nrun < 2) {
#   sink('test.log', type = c("output", "message"))
#   do_one_run(County="Los_AngelesCA")->fit
    do_one_run(County="AlamedaCA")->fit
#   do_one_run(County="HonoluluHI")->fit
#   do_one_run(County="NassauNY",do.plot=TRUE)->fit
#   do_one_run(County="Miami-DadeFL",do.plot=TRUE)->fit
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

