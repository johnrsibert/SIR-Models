SIR_path = '/home/jsibert/Projects/SIR-Models/'
fit_path = paste(SIR_path,'fits/remSIR/',sep='')
dat_path = paste(SIR_path,'dat/',sep='')
TMB_path = paste(SIR_path,'TMB/',sep='')
source(paste(TMB_path,'SIR_read_dat.R',sep=''))
#source(paste(TMB_path,'remSIR_fit_to_df.R',sep=''))
require(TMB)
require(gtools)
#rm(fit)
print('starting')

# ------------------ support functions ------------

make.ytext = function(ylim,prop)
{
    ytext = prop*(ylim[2]-ylim[1])+ylim[1]
    return(ytext)
}

get.error = function(par,opt,map,tn)
{
    if ( is.na(map[tn]) )
    {   
        err = par[tn] 
    }
    else
    {   
        err = opt$par[tn] 
    } 

    print(paste('get.error',tn,err))
    return(as.numeric(err))
}

plot.rv = function(x, obs_y, pred_y, ylab, err, err_name, ylim,pch='+', 
                   obs_col='red', note_col='blue')
{
    plot(x, obs_y, ylab=ylab, ylim=ylim, pch=pch)
    lines(x, pred_y, col=obs_col,lwd=3)
    plot.error(x, pred_y, err)
    ytext = make.ytext(ylim,0.9)
#   print(paste('mean(err)',mean(err)))
    print(paste(ylab,err_name,err))
    if (err > 0.0)
    {
        note = paste(err_name,'~',sprintf("%.5g",err))
    }
    else
    {
        note = err_name
    }
    ttext = 0.1*(length(obs_y)-1)
    text(ttext,ytext,note,col=note_col,pos=4)
}

plot.log.state = function(fit) #,np=5)
{
    attach(fit)

    bias.note = function(name,err=NA)
    {
        w = which(names(obj$report()) == name)
        val = as.numeric(obj$report()[w])
        note = paste('bias = ',signif(val,3),sep='')
        return(note)
    }

    fn = opt$value
    if (is.null(fn))
        fn = opt$objective
    title =  paste(data$county,", f = ",sprintf('%.5g',fn),
                   ", ",model.name,
                   ", C = ", opt$convergence,
                   " (",(opt$convergence==0),")",sep='')
    width = 8.0
    height = 8.0
    x11(width=width,height=height,title=title)#,xpos=100)

    old.par = par(no.readonly = TRUE) 
    par(mar=c(2.0,4.0,0.5,0.5)+0.1) #,oma = c(5,4,0,0) + 0.1)#mfcol=c(3,2),
 
    note.color='blue'
    point.symb = '+'

    lm = layout(matrix(c(1:8),ncol=2,byrow=FALSE))
    layout.show(lm)
#   mtext(title,outer=TRUE,side=1)


    tt = seq(0,(length(data$log_obs_cases)-1))
    ttext = 0.1*(length(data$log_obs_cases)-1)

    poplim = c(0.0,1.2*log(data$N0)) #range(obj$report()$logS)

    plot(tt,obj$report()$logS,ylab='ln S',ylim=poplim, pch=point.symb)
#   err = SElogS
#   logspace.plot.error(tt,obj$report()$logS,err)
    gmlogS = median(obj$report()$logS)
    abline(h=gmlogS,lty='dashed')


    err = exp(get.error(par,opt,map,'logsigma_logC'))
    plot.rv(tt,data$log_obs_cases, obj$report()$logEye,
            ylab='ln cases,', err, err_name='sigma_logC',ylim=poplim)

#   err = exp(get.error(par,opt,map,'logsigma_logR'))
#   plot.rv(tt,data$log_obs_R, obj$report()$logR,
#           ylab='ln R', err, err_name='sigma_logR',ylim=poplim)

    plot(tt,obj$report()$logR,ylab='ln R',ylim=poplim, col='red',type='l',lwd=3)

    ylim=c(0.0,1.2*max(data$log_obs_deaths,obj$report()$logD))
    if (hasName(par,'logsigma_logD'))
    {
    #   Zero infltated log normal
        err = exp(get.error(par,opt,map,'logsigma_logD'))
        ename = 'sigma_logD'
    }
    else
    {
    #   Poisson error
        err = 0.0
        ename = 'lambda'
    }
    plot.rv(tt,data$log_obs_deaths, obj$report()$logD,
            ylab='ln deaths', err, err_name=ename,ylim=ylim)

    rlim = c(-10.0,1.0)
    err = exp(get.error(par,opt,map,'logsigma_logbeta'))
    plot.rv(tt,obj$report()$logbeta, obj$report()$logbeta,
            ylab='ln beta', err=err,
            err_name='sigma_logbeta',ylim=rlim)
 
#   rlim = c(-10.0,1.0)
#   err = exp(get.error(par,opt,map,'logsigma_logmu'))
#   plot.rv(tt,obj$report()$logmu, obj$report()$logmu,
#           ylab='ln mu', err=err,
#           err_name='sigma_logmu',ylim=rlim)

#   rlim = range(obj$report()$logZ)
#   err = exp(get.error(par,opt,map,'logsigma_logZ'))
#   plot.rv(tt,obj$report()$logZ, obj$report()$logZ,
#           ylab='ln Z', err=err,
#           err_name='sigma_logZ',ylim=rlim)


    plot(tt,obj$report()$loggamma,ylab='ln gamma',ylim=rlim,  col='red',type='l',lwd=3)

#   plot.new()
#   mtext(title,outer=FALSE)

#   print(dev.cur())
#   dev.copy2pdf(file='ests.pdf',width=6.5,height=6.5)
#   print(dev.cur())

    par(old.par)
    detach(fit)
    return(dev.cur())
}

plot.error=function(x,y,sd,bcol='black',fcol='gray',mult=2)
{
   if (capabilities("cairo"))
   {
      sdyu = y+mult*sd
      sdyl = y-mult*sd
      frgb = col2rgb(fcol)/255
      
      polygon(c(x,rev(x)),c(sdyl,rev(sdyu)),
              border=bcol,lty="dashed",lwd=1,
              col=rgb(frgb[1],frgb[2],frgb[3],0.5))
   }
   else
      polygon(c(x,rev(x)),c(sdyl,rev(sdyu)),
              border=bcol,lty="dashed",lwd=1,col=fcol)

}


# ------------------ do_one run -------------------

do_one_run = function(County = "Santa Clara",model.name = 'remSIR',do.plot=TRUE)
{
    cname = sub(" ","_",County)
    datfile=paste(dat_path,cname,'.dat',sep='')
    print(paste(separator,County,separator),quote=FALSE)
    dat = read.dat.file(datfile, max_ntime = 200)
    eps = 1e-8
    data=dat$data
    print(names(data))
    print(data)
    data$log_obs_cases = log(data$obs_cases+eps)
    data$log_obs_deaths = log(data$obs_deaths+eps)
    data$logmu_prior = log(0.02)
    data$sigma_logmu_prior = 0.223

    print("-data:")
    print(data)
    print(paste('cases:',length(data$obs_cases)))

    init = list(
    #   logsigma_logCP    = -0.5, #log(0.105),
        logsigma_logRP    = log(0.223),
    #   logsigma_logDP    = -5.0, #log(0.105),

        logsigma_logbeta = log(0.223),
        logsigma_loggamma =  log(0.223),
    #   logsigma_logmu   = -0.5, #log(0.223),
    
    
        logsigma_logC = log(0.05),
        logsigma_logD = log(0.05),

        logbeta  = -2.0, #log(0.01),
        loggamma = -2.0, #log(0.005),
        logmu    = data$logmu_prior 
    #   logR     =  data$log_obs_cases
    )
    print("--init parameter values:")
    print(init)
    
    par = list(
    #   logsigma_logCP = init$logsigma_logCP,
        logsigma_logRP = init$logsigma_logRP,
    #   logsigma_logDP = init$logsigma_logDP,
    
        logsigma_logbeta = init$logsigma_logbeta,
        logsigma_loggamma = init$logsigma_loggamma,
    #   logsigma_logmu = init$logsigma_logmu,
    
        logsigma_logC   = init$logsigma_logC,
        logsigma_logD   = init$logsigma_logD,
    
        logbeta  = rep(init$logbeta,data$ntime+1),
        loggamma = rep(init$loggamma,data$ntime+1),
        logmu    = init$logmu
    #   logR     = rep(init$logR,data$ntime+1) 
    #   logR     = init$logR
    ) 
    print(paste("---initial model parameters: ", length(par)))
    print(par)
    
    map = list(
    #          "logsigma_logCP" = as.factor(1),
               "logsigma_logRP" = as.factor(1),
    #          "logsigma_logDP" = as.factor(1),
    
    #          "logsigma_logbeta" = as.factor(1),
    #          "logsigma_loggamma" = as.factor(NA),
    #          "logsigma_logmu" = as.factor(1),
    
    #          "logsigma_logC" = as.factor(NA),
    #          "logsigma_logD" = as.factor(NA),

    #          "logbeta" = rep(as.factor(NA), data$ntime+1),
    #          "loggamma"    = rep(as.factor(NA), data$ntime+1),
    #          "logR"    = rep(as.factor(NA), data$ntime+1),
               "logmu"   = as.factor(1)
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
    obj = MakeADFun(data,par, random=c("loggamma","logbeta"), 
                    map=map,DLL=model.name)
    print("--------MakeADFun Finished--------------------",quote=FALSE)
    print("obj$par (1):")
    print(obj$par)
    hide_obj = obj
    lb <- obj$par*0-Inf
    ub <- obj$par*0+Inf
    
    #   cmd = 'Rscript --verbose simpleSIR4.R'
    
    #obj$env$inner.control$tol10 <- 0

    #nlminb.con=list(x.tol = 1e-5) #eval.max=5000,iter.max=5000)
    inner.control = list(smartsearch=FALSE,maxit=5)
    #control=list(trace=1,eval.max=1,iter.max=5)
    control=list(eval.max=10,iter.max=5)
    
    print("Starting minimization-------------------------",quote=FALSE)
    opt = nlminb(obj$par,obj$fn,obj$gr,control=control)
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
        dev.file = paste(fit_path,data$county,'.pdf',sep='')
    #   dev.file = paste(data$county,'_ests.pdf',sep='')
        dev.copy2pdf(file=dev.file,width=6.5,height=9)
    #   dev.off()
    }
    
#   rd.file = paste(fit_path,data$county,'.RData',sep='')
#   save.fit(fit,file=data$county,mod=model.name)#   rd.file) #"t.RData")
    
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
#   print(paste('median logbeta',median(fit$obj$report()$logbeta)))
#   print(paste('median loggamma',median(fit$obj$report()$loggamma)))
    print(paste('opt$par',fit$opt$par['logmu'],exp(fit$opt$par['logmu'])))
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
