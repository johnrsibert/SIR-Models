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

    print(paste(tn,err))
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
        note = err_name

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

#   sdr = sdreport(obj)
#   SElogbeta = as.list(sdr,"Std. Error")$logbeta
#   SEloggamma   = as.list(sdr,"Std. Error")$loggamma
#   SElogmu   = as.list(sdr,"Std. Error")$logmu
#   print(paste(SElogbeta,SEloggamma,SElogmu))
#   print(SElogbeta)
#   print(mean(SElogbeta))

    fn = opt$value
    if (is.null(fn))
        fn = opt$objective
    title =  paste(dat$county,", f = ",sprintf('%.5g',fn),
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


    tt = seq(0,(length(dat$log_obs_cases)-1))
    ttext = 0.1*(length(dat$log_obs_cases)-1)

    poplim = c(0.0,1.2*log(dat$N0)) #range(obj$report()$logS)
    plot(tt,obj$report()$logS,ylab='ln S',ylim=poplim, pch=point.symb)
#   err = SElogS
#   logspace.plot.error(tt,obj$report()$logS,err)
    gmlogS = median(obj$report()$logS)
    abline(h=gmlogS,lty='dashed')


    err = exp(get.error(par,opt,map,'logsigma_logC'))
    plot.rv(tt,dat$log_obs_cases, obj$report()$logEye,
            ylab='ln cases,', err, err_name='sigma_logC',ylim=poplim)

    err = exp(get.error(par,opt,map,'logsigma_logR'))
    plot.rv(tt,dat$log_obs_R, obj$report()$logR,
            ylab='ln R', err, err_name='sigma_logR',ylim=poplim)

    ylim=c(0.0,1.2*max(dat$log_obs_deaths,obj$report()$logD))
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
    plot.rv(tt,dat$log_obs_deaths, obj$report()$logD,
            ylab='ln deaths', err, err_name=ename,ylim=ylim)

    rlim = c(-10.0,1.0)
    err = exp(get.error(par,opt,map,'logsigma_logbeta'))
    plot.rv(tt,obj$report()$logbeta, obj$report()$logbeta,
            ylab='ln beta', err=err,
            err_name='sigma_logbeta',ylim=rlim)
 
    rlim = c(-10.0,1.0)
    err = exp(get.error(par,opt,map,'logsigma_logmu'))
#   print(paste(length(tt),length(obj$report()$logmu),length(obj$report()$sigma_logmu)))
    plot.rv(tt,obj$report()$logmu, obj$report()$logmu,
            ylab='ln mu', err=err,
            err_name='sigma_logmu',ylim=rlim)

    glim = range(obj$report()$loggamma)
    err = exp(get.error(par,opt,map,'logsigma_loggamma'))
    plot.rv(tt,obj$report()$loggamma, obj$report()$loggamma,
            ylab='ln gamma', err=err,
            err_name='sigma_loggamma',ylim=rlim)

    cfrlim = 0.4
    prd_cfr = exp(obj$report()$logD - obj$report()$logEye)
    obs_cfr = dat$obs_deaths / dat$obs_cases
    plot(tt, obs_cfr,ylab='Deaths/Cases',ylim=(c(0.0,cfrlim)), pch=point.symb)
    lines(tt, prd_cfr,col='red')
    ttext = 0.1*(length(dat$log_obs_cases)-1)
    ytext = cfrlim*0.9
    print(obj$report()$cfrpen)
    note = paste('cfr penalty =',sprintf("%g",obj$report()$cfrpen))
    print(note)
    text(ttext,ytext,note,col=note.color,pos=4)
    

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

