SIR_path = '/home/jsibert/Projects/SIR-Models/'
fit_path = paste(SIR_path,'fits/',sep='')
dat_path = paste(SIR_path,'dat/',sep='')
TMB_path = paste(SIR_path,'TMB/',sep='')
SIRmodel.name = 'simpleSIR4'
source(paste(TMB_path,'plot_state.R',sep=''))
source(paste(TMB_path,'SIR_read_dat.R',sep=''))
source(paste(TMB_path,'fit_to_df.R',sep=''))
require(TMB)
require(gtools)

make_data=function(County)
{
    cname = sub(" ","_",County)
    datfile=paste(dat_path,cname,'.dat',sep='')
    separator = "#############################"
    print(paste(separator,County,separator),quote=FALSE)
    
    dat = read.dat.file(datfile)#,max_ntime = 196)
    
    eps = 1e-8
    data=dat$dat
    print(names(data))
    print(data)
    data$log_obs_cases = log(data$obs_cases+eps)
    data$log_obs_deaths = log(data$obs_deaths+eps)
    print("-data:")
    print(data)
#   print(data$ntime)
    return(data)
}
                      #init       est  map
#names                                    
#logsigma_logCP   -2.000000 -2.450247    1
#logsigma_logDP   -3.300000 -3.216701    1
#logsigma_logbeta -0.800000 -0.800147    1
#logsigma_logmu   -1.500000 -1.513849    1
#logsigma_logC    -2.995732 -2.995732  NaN
#logsigma_logD    -2.995732 -2.995732  NaN


make_init=function()
{
    init = list(
        logsigma_logCP = -2.0, #log(0.12),
        logsigma_logDP = -3.3, #log(0.096),
        logsigma_logbeta = -0.8, # 0.4, #log(0.7),
        logsigma_logmu = -1.5, #log(2.0),
    
        logsigma_logC = log(0.05),
        logsigma_logD = log(0.05)
    )
    print("--init parameter values:")
    print(names(init))
    print(init)
    return(init)
}

make_par=function(data,init)
{
    par = list(
        logsigma_logCP = init$logsigma_logCP,
        logsigma_logDP = init$logsigma_logDP,
        logsigma_logbeta = init$logsigma_logbeta,
        logsigma_logmu = init$logsigma_logmu,
        logsigma_logC = init$logsigma_logC,
        logsigma_logD = init$logsigma_logD,
    
        logbeta  = rep(0.0,data$ntime+1),
        logmu    = rep(0.0,data$ntime+1) 
    )
#   print(paste("---initial model parameters: ", length(par)))
#   print(par)
    return(par)
}

make_map=function()
{
   map = list(
         "logsigma_logCP" = as.factor(1),
         "logsigma_logDP" = as.factor(1),
         "logsigma_logbeta" = as.factor(1),
         "logsigma_logmu" = as.factor(1),

         "logsigma_logC" = as.factor(NA),
         "logsigma_logD" = as.factor(NA)
    )
#   print(paste("---- estimation map:",length(map),"variables"))
#   print(map)
    return(map)
}

build_mode=function(mmeodel.name,do.compile=FALSE)
{
    cpp.name = paste(SIRmodel.name,'.cpp',sep='')
    if (do.compile)
    {
        print(paste("Compiling",cpp.name,"-------------------------"),quote=FALSE)
        compile(cpp.name)
        print("Finished compilation-------------",quote=FALSE)
    } 
#   print(paste("Loading",SIRmodel.name,"-------------------------"),quote=FALSE)
#   dyn.load(dynlib(SIRmodel.name))
#   print("Finished dyn.load-------------",quote=FALSE)
}

make_model_function=function(SIRmodel.name,data,par,map,rand=c("logbeta","logmu")) 
{
    print("Calling MakeADFun-----------------------------",quote=FALSE)
    obj = MakeADFun(data,par,random=rand, map=map,DLL=SIRmodel.name)
    print("--------MakeADFun Finished--------------------",quote=FALSE)
    return(obj)
}

run_model=function(obj)
{
    print("Starting minimization-------------------------",quote=FALSE)
    print("obj$par (1):")
    print(obj$par)
    #lb <- obj$par*0-Inf
    #ub <- obj$par*0+Inf
    #control=list(trace=1,eval.max=1,iter.max=4,rel.tol=1e-4,abs.tol=1e-3)
    #control=list(trace=1,iter.max=6)
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
    mrho = median(obj$report()$rho)
    print(paste("median rho:",mrho))
    print(names(opt))
    return(opt)
}


test = function(moniker='AlamedaCA')
{
    data = make_data(moniker)
    init  = make_init()
    par  = make_par(data,init)
    map  = make_map()
    build_mode(SIRmodel.name,do.compile=TRUE)
    print(paste("Loading",SIRmodel.name,"-------------------------"),quote=FALSE)
    dyn.load(dynlib(SIRmodel.name))
    print("Finished dyn.load-------------",quote=FALSE)

    obj  = make_model_function(SIRmodel.name,data,par,map,rand=c("logbeta","logmu")) 

    opt = run_model(obj)


    fit = list(data=data,map=map,par=par,obj=obj,opt=opt,init=init,
               model.name=SIRmodel.name)

    plot.log.state(fit,file=data$county,np=5)
    save.fit(fit,file=data$county)
    save(fit,file='last_tfit.RData')
    return(fit)
}

obs_error_runs=function(moniker='AlamedaCA',err=c(0.01,0.02,0.05,0.1,0.2))
{
    data = make_data(moniker)
    init  = make_init()
    par  = make_par(data,init)
    map  = make_map()
    print(paste("Loading",SIRmodel.name,"-------------------------"),quote=FALSE)
    dyn.load(dynlib(SIRmodel.name))
    print("Finished dyn.load-------------",quote=FALSE)
    print(err)
    nrun = length(err)
    for (i in 1:(nrun+1))
    {
        if (i <= nrun)
        {
            file = paste(moniker,i,sep='')
            s = err[i]
            print(paste('==================',file,'s=',s))
            init$logsigma_logC = log(s)
            init$logsigma_logD = log(s)
            par$logsigma_logC = init$logsigma_logC 
            par$logsigma_logD = init$logsigma_logD 
            print('    par:')
            print(par)
            obj  = make_model_function(SIRmodel.name,data,par,map,
                                       rand=c("logbeta","logmu")) 
        }
        else
        {
            file = paste(moniker,0,sep='')
            print(paste('==================',file,'s= ?'))
            map$logsigma_logC = as.factor(1)
            map$logsigma_logD = as.factor(1)

        }
        opt = run_model(obj)

        fit = list(data=data,map=map,par=par,obj=obj,opt=opt,init=init,
                   model.name=SIRmodel.name)

        plot.log.state(fit,file_root=file,np=5)
        save.fit(fit,file_root=file)

    }
}

glob_runs = function(do.plot=FALSE)
{

    init  = make_init()
    dp =paste(dat_path,'*.dat',sep='')
    print(dp)
    print(paste('globbing',dp))
    cc = Sys.glob(dp)
    print(paste('    found:',cc))

    log_file = paste(fit_path,SIRmodel.name,'.log',sep='') 
    print(paste('Redirecting output to',log_file))
    sink(log_file, type = c("output", "message"))

    for (c in cc)
    {
        tmp = sub("\\.dat","",c)
        print(tmp)
        moniker = basename(tmp)
        print(paste("Glob starting",moniker))
        data = make_data(moniker)
    #   init  = make_init()
        par  = make_par(data,init)
        map  = make_map()
        print(paste("Loading",SIRmodel.name,"-------------------------"),quote=FALSE)
        dyn.load(dynlib(SIRmodel.name))
        print("Finished dyn.load-------------",quote=FALSE)
 
        obj  = make_model_function(SIRmodel.name,data,par,map,
                                   rand=c("logbeta","logmu")) 
        opt = run_model(obj)

        fit = list(data=data,map=map,par=par,obj=obj,opt=opt,init=init,
                   model.name=SIRmodel.name)
        if (do.plot)
        {
            plot.log.state(fit,file_root=data$county,np=5)
        }
        save.fit(fit,file_root=data$county)

    }

    sink()
    print(paste('Closed log file,',log_file))
}

#do_one_run = function(County = "Santa Clara",model.name = 'simpleSIR4',do.plot=TRUE)
#{
#   
#    
##   fit = list(data=data,map=map,par=par,obj=obj,opt=opt,init=init,
##              model.name=model.name)
#    if (do.plot)
#    {
#        x11()
#        plot.log.state(fit,np=5)
#        dev.file = paste(fit_path,data$county,'.png',sep='')
#        print(paste('Attempting to save plot as',dev.file))
#        dev.copy(png,file=dev.file,width=6.5,height=9,unit='in',res=300)
#        print(paste('plot saved as',dev.file))
#        dev.off()
#    }
#    
#    save.fit(fit,file=data$county)#   rd.file) #"t.RData")
#    
#    return(fit)
#
#} # do_one_run = function(
#
#
#
#nrun = 1
#if (nrun < 2) {
##   do_one_run(County="Los_AngelesCA")->fit
#    do_one_run(County="AlamedaCA")->fit
##   do_one_run(County="HonoluluHI")->fit
##   do_one_run(County="NassauNY")->fit
##   do_one_run(County="BrowardFL")->fit
#} else {
#   sink( paste(fit_path,'SIR_model.log',sep=''), type = c("output", "message"))
#   dp =paste(dat_path,'*.dat',sep='')
#   print(dp)
#   print(paste('globbing',dp))
#   cc = Sys.glob(dp)
#   for (c in cc)
#   {
#       County = sub("\\.dat","",c)
#       c = basename(County)
#       print(paste("starting",c))
#       do_one_run(County=c,do.plot=FALSE)->fit
#   }
#   sink()
#}
#
