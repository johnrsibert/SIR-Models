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
    logsigma_logCP = log(0.12),
    logsigma_logDP = log(0.096),
    logsigma_logbeta = 0.4, #log(0.7),
    logsigma_logmu = 0.2, #log(2.0),
    logsigma_logC = log(log(1.25)), # log(0.1), 
    logsigma_logD = log(log(1.1)), # log(0.05), 

    logbeta = log(0.05),
    loggamma = log(0.9),
    logmu = log(0.00005) 
)
print("--init parameter values:")
print(init)
init
par = list(
    logsigma_logCP = init$logsigma_logCP,
    logsigma_logDP = init$logsigma_logDP,
    logsigma_logbeta = init$logsigma_logbeta,
    logsigma_logmu = init$logsigma_logmu,
    logsigma_logC = init$logsigma_logC,
    logsigma_logD = init$logsigma_logD,

    logbeta  = rep(init$logbeta,(data$ntime+1)),
    loggamma = init$loggamma,
    logmu    = rep(init$logmu,data$ntime+1) 
)
print(paste("---initial model parameters: ", length(par)))
print(par)

map = list(
           "logsigma_logCP" = as.factor(1),
           "logsigma_logDP" = as.factor(1),
           "logsigma_logC" = as.factor(NA),
           "logsigma_logD" = as.factor(NA),
           "logsigma_logbeta" = as.factor(1),
           "logsigma_logmu" = as.factor(1),

           "loggamma" = as.factor(NA)
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
#opt = optim(obj$par,obj$fn,obj$gr)

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

fit = list(data=data,map=map,par=par,obj=obj,opt=opt,init=init,
           model.name=model.name)
if (do.plot){
    x11()
    plot.log.state(data,par,obj,opt,map,np=4)
    dev.file = paste(fit_path,data$county,'.pdf',sep='')
    dev.copy2pdf(file=dev.file,width=6.5,height=6)
    dev.off()
}

#rd.file = paste(fit_path,data$county,'.RData',sep='')
#save.fit(data,obj,opt,map,init,rd.file)
save.fit(fit,file=data$county)#   rd.file) #"t.RData")

return(fit)

} # do_one_run = function(



nrun = 1
if (nrun < 2) {
#   do_one_run(County="Los_AngelesCA")->fit
    do_one_run(County="AlamedaCA")->fit
#   do_one_run(County="HonoluluHI")->fit
#   do_one_run(County="NassauNY")->fit
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
       do_one_run(County=c,do.plot=FALSE)->fit
   }
   sink()
}

