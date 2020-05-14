#rm(list=ls())
#gc(verb=TRUE)
eps = 1e-8
source('./SIR_read_dat.R')

print(" ",quote=FALSE)
print("##############################################",quote=FALSE)
SIR.dat = read.dat.file('simpleSIR.dat')
print(names(SIR.dat))
print(names(SIR.dat$data))
data = SIR.dat$data
data$log_obs_cases = log(data$obs_cases+eps)
data$log_obs_deaths = log(data$obs_deaths+eps)
#data$obs_cases = ""
#data$obs_deaths = ""
print("----data:")
print(data)
# from python simulator
# def SIRsim(mu = 0.001, beta = 0.5, gamma = 0.075, N0 = 1e5, sigma_eye = 0.3,

init = list(
    sigma_logP = 0.1,
    sigma_logbeta = 0.05,
    mu = 0.5,
    gamma = 0.5,
    sigma_logC = 0.69,
    sigma_logD = 0.69,
    beta = 0.5
)
print("-initial values:")
print(init)

parameters = list(
    sigma_logP = init$sigma_logP,
    sigma_logbeta = init$sigma_logbeta,
    logmu    = log(init$mu),
    loggamma = log(init$gamma),
    sigma_logC = init$sigma_logC,
    sigma_logD = init$sigma_logD,
    logbeta = rep(log(init$beta),data$ntime)
#   logEye = rep(log(10.0),data$ntime),
#   logD = rep(log(eps),data$ntime)
)
#   logbeta = rep(log(init$beta),data$ntime),
#print(paste("--parameters: ", length(parameters)))
#print(paste(data$obs_cases[1],data$obs_deaths[1]))
#print(paste(data$obs_cases[1],data$obs_deaths[1]))
#parameters$Eye[1] = data$obs_cases[1]
#parameters$D[1] =  data$obs_deaths[1]
#parameters$logbeta[1] = -2.0

print(paste("--parameters: ", length(parameters)))
print(parameters)

map = list()# "sigma_logP" = as.factor(NA))
#          "sigma_logbeta" = as.factor(NA))
#           "sigma_logC" = as.factor(NA),
#           "sigma_logD" = as.factor(NA))
#          "sigma_logbeta" = as.factor(NA))
#          "logmu"  = as.factor(NA))
#          "loggamma"  = as.factor(NA),
print(paste("---map: ",length(map)))
print(map)

require(TMB)

print("Compiling-------------------------------------",quote=FALSE)
compile("simpleSIR.cpp")
dyn.load(dynlib("simpleSIR"))
print("Finished compilation and dyn.load-------------",quote=FALSE)
print("Calling MakeADFun-----------------------------",quote=FALSE)
obj = MakeADFun(data,parameters,random=c("logbeta"), #"logEye","logD"),
                map=map,DLL="simpleSIR")
print("--------MakeADFun Finished--------------------",quote=FALSE)
print(paste("par (1): ",obj$par))

print("Starting minimization-------------------------",quote=FALSE)
options(warn=2,verbose=FALSE)
#obj$env$tracepar = TRUE
#obj$control=list(trace=2,eval.max=1,iter.max=1,rel.tol=1e-3,x.tol=1e-3)
obj$control=list(eval.max=500)#,iter.max=10)
opt = nlminb(obj$par,obj$fn,obj$gr,control=obj$control)

print("Done minimization-----------------------------",quote=FALSE)
print(paste("Objective function value =",opt$objective))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
print("parameters:",quote=FALSE)
print(obj$par)
print(opt$par)
#print(exp(opt$par))
gmbeta = exp(mean(obj$report()$logbeta))
print(paste("geometric mean beta:",gmbeta))

source('plot_state.R')
plot.log.state(data,obj,parameters,np=3)
#dev.copy2pdf(file=paste(data$county,'.pdf',sep=''),width=6.5,height=6)
#dev.off()

source('fit_to_df.R')
save.fit(data,obj,opt)
