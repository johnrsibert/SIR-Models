#rm(list=ls())
#gc(verb=TRUE)
eps = 1e-8
source('./SIR_read_dat.R')
print(" ",quote=FALSE)
print("##############################################",quote=FALSE)
SIR.dat = read.dat.file()
print(names(SIR.dat))
data = SIR.dat$data
print("----data:")
print(data)
# from python simulator
# def SIRsim(mu = 0.001, beta = 0.5, gamma = 0.075, N0 = 1e5, sigma_eye = 0.3,

init = list(
    sigma_P = 0.5, #1e3,
    sigma_beta = 0.01, #2.0,
    mu = 0.05,
    gamma = 0.075,
    sigma_C = 0.25,
    sigma_D = 0.25,
    beta = 0.2
)
print("-initial values:")
print(init)

parameters = list(
    logsigma_P = log(init$sigma_P),
    logsigma_beta = log(init$sigma_beta),
    logmu    = log(init$mu),
    loggamma = log(init$gamma),
    logsigma_C = log(init$sigma_C),
    logsigma_D = log(init$sigma_D),
    logbeta = rep(log(init$beta),data$ntime)
)
print(paste("--parameters: ", length(parameters)))
print(parameters)
#print(names(parameters))
#print(names(map))
#print(paste("map: ",length(map)))
map = list("logsigma_C" = as.factor(NA),
           "logsigma_D" = as.factor(NA))
#          "logsigma_beta" = as.factor(NA),
#          "loggamma"  = as.factor(NA),
#          "logmu"  = as.factor(NA),
#map = list(
#      logsigma_P = as.factor(1),
#      logsigma_beta = as.factor(1),
#      logmu    = as.factor(NA),
#      loggamma = as.factor(NA),
#      logsigma_C = as.factor(NA),
#      logsigma_D = as.factor(NA),
#      logbeta = as.factor(rep(1,length(parameters$logbeta)))
#)
#map = list(
#      "logsigma_P" = as.factor(1),
#      "logsigma_beta" = as.factor(1),
#      "logmu"    = as.factor(NA),
#      "loggamma" = as.factor(NA),
#      "logsigma_C" = as.factor(NA),
#      "logsigma_D" = as.factor(NA),
#      "logbeta" = as.factor(rep(1,length(parameters$logbeta)))
#)
print(paste("map: ",length(map)))
print(map)

require(TMB)

print("\nCompiling-------------------------------------",quote=FALSE)
compile("SIRest1.cpp")
dyn.load(dynlib("SIRest1"))
print("\nFinished compilation and dyn.load-------------",quote=FALSE)
print("\nCalling MakeADFun-----------------------------",quote=FALSE)
##mody=MakeADFun(data, parsy, random=c("rei", "rey"), DLL="REiy", map=mapy)
obj = MakeADFun(data,parameters,random=c("logbeta"),map=map,DLL="SIRest1")
#obj = MakeADFun(data,parameters,map=map,DLL="SIRest1")
print("\n--------MakeADFun Finished--------------------",quote=FALSE)
print(paste("par: ",obj$par))

print("\nStarting minimization-------------------------",quote=FALSE)
obj$control=list(trace=1,eval.max=1,iter.max=2)
opt <- nlminb(obj$par,obj$fn,obj$gr,control=obj$control)
##opty= nlminb(mody $par, mody $fn, mody $gr)
print("\nDone minimization-----------------------------",quote=FALSE)
print(paste("Objective function value =",opt$objective))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
print("parameters:\n",quote=FALSE)
print(exp(opt$par))

plot.state(data,obj)
