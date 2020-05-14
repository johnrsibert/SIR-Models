source('plot_state.R')
source('SIR_read_dat.R')
source('fit_to_df.R')
fit_path = '/home/jsibert/Projects/Covid-19/fits/'
require(TMB)


do_one_run = function(County = "Santa Clara",model.name = 'simpleSIR4')
{
cname = sub(" ","_",County)
datfile=paste('../',cname,'.dat',sep='')
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
    sigma_logmu = 0.05,
    mu = 0.5,
    gamma = 1.0,
    sigma_logC = log(2.5),
    sigma_logD = log(1.5),
    beta = 0.5
)
print("--initial parameter values:")
print(init)

par = list(
    sigma_logP = init$sigma_logP,
    sigma_logbeta = init$sigma_logbeta,
    sigma_logmu = init$sigma_logmu,
    logmu    = rep(log(init$mu),data$ntime),
    loggamma = log(init$gamma),
    sigma_logC = init$sigma_logC,
    sigma_logD = init$sigma_logD,
    logbeta = rep(log(init$beta),data$ntime)
)
print(paste("---model parameters: ", length(par)))
print(par)

map = list(
           "sigma_logP" = as.factor(1),
           "sigma_logbeta" = as.factor(1),
           "sigma_logmu" = as.factor(1),
           "loggamma"  = as.factor(1),
           "sigma_logC" = as.factor(NA),
           "sigma_logD" = as.factor(NA),
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

print("Starting minimization-------------------------",quote=FALSE)
options(warn=2,verbose=FALSE)
obj$control=list(eval.max=500,iter.max=10)
#opt = nlminb(obj$par,obj$fn,obj$gr,control=obj$control)
 opt = optim(obj$par,obj$fn,obj$gr,method="BFGS")
tod = format(Sys.time(), "%Y%m%d%H%M%S")

print("Done minimization-----------------------------",quote=FALSE)
print(paste("Objective function value =",opt$objective))
print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)
print("parameters:",quote=FALSE)
print(obj$par)
print(opt$par)
print(exp(opt$par))
gmbeta = exp(mean(obj$report()$logbeta))
print(paste("geometric mean beta:",gmbeta))
gmmu = exp(mean(obj$report()$logmu))
print(paste("geometric mean mu:",gmmu))

plot.log.state(data,par,obj,opt,map,np=4)
dev.file = paste(fit_path,data$county,'_',tod,'.pdf',sep='')
dev.copy2pdf(file=dev.file,width=6.5,height=6)

rd.file = paste(fit_path,data$county,'_',tod,'.RData',sep='')
save.fit(data,obj,opt,map,rd.file)

return(list(data=data,par=par,obj=obj,opt=opt))

} # do_one_run = function((County = "Santa Clara",model.name = 'simpleSIR4')

#do_one_run(County='Contra Costa')->fit
sink( paste(fit_path,'SIR_model.log',sep=''), type = c("output", "message"))
county_list = list("Alameda", "Contra_Costa", "San_Francisco", "San_Mateo",
                    "Santa_Clara")
for (c in 1:length(county_list))
{
    do_one_run(County=county_list[c])->junk
}
#sink()
