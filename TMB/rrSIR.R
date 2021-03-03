SIR_path = '/home/jsibert/Projects/SIR-Models/'
graphics_path = paste(SIR_path,'Graphics/rrSIR/',sep='')
fit_path = paste(SIR_path,'fits/rrSIR/',sep='')
dat_path = paste(SIR_path,'dat/',sep='')
TMB_path = paste(SIR_path,'TMB/',sep='')
source(paste(TMB_path,'SIR_read_dat.R',sep=''))
source(paste(TMB_path,'plot_rrSIR_state.R',sep=''))
source(paste(TMB_path,'rrSIR_fit_to_df.R',sep=''))
require(TMB)
require(gtools)
#rm(fit)
print('starting')

# ------------------ do_one run -------------------

do_one_run = function(County = "Santa Clara",model.name = 'rrSIR',do.plot=TRUE)
{
    cname = sub(" ","_",County)
    datfile=paste(dat_path,cname,'.dat',sep='')
    print(paste(separator,County,separator),quote=FALSE)
    dat = read.dat.file(datfile)##,max_ntime = 200)
    eps = 1e-8
    data=dat$data
    print(names(data))
    print(data)
    data$log_obs_cases = log(data$obs_cases+eps)
    data$log_obs_deaths = log(data$obs_deaths+eps)
    print("-data:")
    print(data)
    #                       est                 init

    
    init = list(
    #   compartment process errors
        logsigma_logCP    = log(0.105),
        logsigma_logRP    = log(0.105),
        logsigma_logDP    = log(0.105),

    #   rate random walks
        logsigma_logbeta  = log(0.105),
        logsigma_loggamma = -1.0, #log(0.105),
        logsigma_logmu    = -1.0, #log(0.105),
    
    #   observation errors 
        logsigma_logC = log(0.097),
        logsigma_logD = log(0.097),

    #   simpleSIR4.R:
    #   logsigma_logC = log(log(1.25)),
    #   logsigma_logD = log(log(1.1)),
 
    #   rate parameter random effects
        logbeta  = -4.0, #log(0.01),
        loggamma = -7.5, #log(0.005),
        logmu    = -5.0 #log(0.0001)
    )
    print("--init parameter values:")
    print(init)
    
    par = list(
        logsigma_logCP = init$logsigma_logCP,
        logsigma_logRP = init$logsigma_logRP,
        logsigma_logDP = init$logsigma_logDP,
    
        logsigma_logbeta  = init$logsigma_logbeta,
        logsigma_loggamma = init$logsigma_loggamma,
        logsigma_logmu    = init$logsigma_logmu,
    
        logsigma_logC   = init$logsigma_logC,
        logsigma_logD   = init$logsigma_logD,
    
        logbeta  = rep(init$logbeta,data$ntime+1),
        loggamma = rep(init$loggamma,data$ntime+1),
        logmu    = rep(init$logmu,  data$ntime+1) 
    )
    
    print(paste("---initial model parameters: ", length(par)))
    print(par)
    
    map = list(
               "logsigma_logCP" = as.factor(1),
               "logsigma_logRP" = as.factor(1),
               "logsigma_logDP" = as.factor(1),
    
    #          "logsigma_logbeta" = as.factor(NA),
    #          "logsigma_loggamma" = as.factor(NA),
    #          "logsigma_logmu" = as.factor(NA),

    #          "logbeta" = rep(as.factor(NA), data$ntime+1),
               "loggamma"    = rep(as.factor(NA), data$ntime+1),
    #          "logmu"   = rep(as.factor(NA), data$ntime+1),

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
    obj = MakeADFun(data,par,  random=c("logbeta","loggamma","logmu"), 
                    map=map,DLL=model.name)
    print("--------MakeADFun Finished--------------------",quote=FALSE)
    print("obj$par (1):")
    print(obj$par)
    lb <- obj$par*0-10
    ub <- obj$par*0-1
#   lb["logR"] <- 0.0
#   print(paste('ub:',length(ub)))
#   ub["logbeta"] <- -1.0
#   lb["logbeta"] <- -5.0
#   ub["loggamma"] <- -1.2
#   lb["loggamma"] <- -7.5
#   ub["logmu"] <- -5.0
#   lb["logmu"] <- -10.0
#   lb["logR"] <- 0.0
    print(paste('ub:',length(ub)))
    print(ub)
    print(paste('lb:',length(lb)))
    print(lb)


    
    #   cmd = 'Rscript --verbose simpleSIR4.R'
    

    control=list(trace=1,eval.max=1,iter.max=4,rel.tol=1e-4,abs.tol=1e-3)

    
    print("Starting minimization-------------------------",quote=FALSE)
    opt = nlminb(obj$par,obj$fn,obj$gr,lower=lb, upper=ub,control=control)
                 


    #opt = optim(obj$par,obj$fn,obj$gr)
    
    print("Done minimization-----------------------------",quote=FALSE)
    print(paste("Function objective =",opt$objective))
    print(paste("Function value =",opt$value))
    print(paste("Convergence ",opt$convergence))
    print(paste("Number of parameters = ",length(opt$par)),quote=FALSE)

    fit = list(data=data,map=map,par=par,obj=obj,opt=opt,init=init,
               model.name=model.name)
    if (do.plot){
    #   x11()
    #   print(fit)
        print('plotting')
        plot.log.state(fit)
    #   dev.file = paste(fit_path,data$county,'.pdf',sep='')
    #   dev.copy2pdf(file=dev.file,width=6.5,height=9)
        dev.file = paste(graphics_path,data$county,init$loggamma,'.png',sep='')
        print(paste('Attempting to save plot as',dev.file))
        dev.copy(png,file=dev.file,width=6.5,height=9,unit='in',res=300) #bg='white', 
        print(paste('plot saved as',dev.file))
        dev.off()
#> dev.copy(png,'myplot.png')
#> dev.off()
    }
    
#   rd.file = paste(fit_path,data$county,'.RData',sep='')
#   save.fit(fit,file=data$county,mod=model.name)#   rd.file) #"t.RData")
    save.fit(fit,file=paste(data$county,init$loggamma,sep=''),mod=model.name)#   rd.file) #"t.RData")
    
    return(fit)

} # do_one_run = function(County = ...)


separator = "#############################"
print(separator)


nrun = 1
print(paste('nrun =',nrun))
if (nrun < 2) {
#   County="Miami-DadeFL"
#   County="KingWA"
    County="AlamedaCA"
#   County = "Los_AngelesCA"
#   County = "New_York_CityNY"
    print(paste('----nrun =',nrun,County))
    do_one_run(County=County)->fit
    for (n in 1:length(fit$opt$par))
    {
    #   print(paste('name:',names(fit$opt$par[n])))
    #   print(paste('     ',fit$opt$par[n]))
        name = names(fit$opt$par[n])
        w = which(names(fit$par)==name)
    #   print(paste(name,w))
        print(paste(name,fit$opt$par[n],fit$par[w]))
    }
    print(paste('median logbeta',median(fit$obj$report()$logbeta)))
    print(paste('median loggamma',median(fit$obj$report()$loggamma)))
    print(paste('median logmu',median(fit$obj$report()$logmu)))
} else {
    print(paste('----nrun =',nrun))
    sink( paste(fit_path,'SIR_model.log',sep=''), type = c("output", "message"))
    dp =paste(dat_path,'*.dat',sep='')
    print(dp)
#   print(paste('globbing',dp))
#   cc = Sys.glob(dp)
cc = list(
'AlamedaCA',
'BexarTX',
'BrowardFL',
'ClarkNV',
'CookIL',
'DallasTX',
'FranklinOH',
'HarrisTX',
'HennepinMN',
'HillsboroughFL',
"Los_AngelesCA", 
"MaricopaAZ", 
"Miami-DadeFL", 
"MiddlesexMA", 
"NassauNY", 
"New_York_CityNY", 
"OrangeCA", 
"OrangeFL", 
"Palm_BeachFL", 
"PhiladelphiaPA", 
"RiversideCA", 
"SacramentoCA", 
"San_BernardinoCA", 
"San_DiegoCA", 
"Santa_ClaraCA", 
"SuffolkNY", 
"TarrantTX", 
"TravisTX", 
"WayneMI") 

    for (c in cc)
    {
        County = sub("\\.dat","",c)
        c = basename(County)
        print(paste("starting",c))
        do_one_run(County=c,do.plot=FALSE)
    }
    sink()
 }


#'KingWA',
