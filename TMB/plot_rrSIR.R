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

    return(as.numeric(err))
}

plot.log.state = function(fit) #,np=5)
{
    attach(fit)

    bias.note = function(name,err=NA)
    {
        w = which(names(obj$report()) == name)
        val = log(as.numeric(obj$report()[w]))
        note = paste('b = ',signif(val,3),sep='')
        return(note)
    }

    sdr = sdreport(obj)
    SElogbeta = as.list(sdr,"Std. Error")$logbeta
    SEloggamma   = as.list(sdr,"Std. Error")$loggamma
    SElogmu   = as.list(sdr,"Std. Error")$logmu

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
    par(mar=c(2.0,4.0,0.5,0.5)+0.1)
 
    note.color='blue'
    point.symb = '+'

    lm = layout(matrix(c(1:8),ncol=2,byrow=FALSE))
    layout.show(lm)

    tt = seq(0,(length(dat$log_obs_cases)-1))
    ttext = 0.1*(length(dat$log_obs_cases)-1)
    ylim=c(0.0,1.2*max(dat$log_obs_cases,obj$report()$logEye))
    plot(tt,dat$log_obs_cases,ylab='ln cases,',ylim=ylim, pch=point.symb)
    lines(tt,obj$report()$logEye,col='red')
    err = exp(get.error(par,opt,map,'logsigma_logC'))
    plot.error(tt,obj$report()$logEye,err)
    ytext = make.ytext(ylim,0.9)
    note = paste('sigma_logC ~',sprintf("%.5g",err))
    text(ttext,ytext,note,col=note.color,pos=4)
#   title(main=title,sub='sub')

    ylim=c(0.0,1.2*max(dat$log_obs_deaths,obj$report()$logD))
    plot(tt,log(dat$obs_deaths),ylab='ln deaths',ylim=ylim,pch=point.symb)
    lines(tt,obj$report()$logD,col='red')
    err = exp(sqrt(obj$report()$logD)/dat$ntime) # Poisson
    plot.error(tt,obj$report()$logD,err)

    ylim = range(obj$report()$logbeta)
    plot(tt,obj$report()$logbeta,ylab='ln beta',pch=point.symb)
    ttext = 0.95*(length(dat$log_obs_cases)-1)
    ytext = make.ytext(ylim,0.9)
    text(ttext,ytext,bias.note("bias_logbeta"),col=note.color,pos=2) 
    err = SElogbeta
    plot.error(tt,obj$report()$logbeta,err)
    gmlogbeta = median(obj$report()$logbeta)
    abline(h=gmlogbeta,lty='dashed')
 
    ylim = range(obj$report()$logmu)
    plot(tt,obj$report()$logmu,ylab='ln mu',ylim=ylim, pch=point.symb)
    ttext = 0.95*(length(dat$log_obs_cases)-1)
    ytext = make.ytext(ylim,0.9)
    text(ttext,ytext,bias.note("bias_logmu"),col=note.color,pos=2) 
    err = SElogmu
    plot.error(tt,obj$report()$logmu,err)
    gmlogmu = median(obj$report()$logmu)
    abline(h=gmlogmu,lty='dashed')

    ylim = c(0.0,log(dat$N0)) #range(obj$report()$logS)
    plot(tt,obj$report()$logS,ylab='ln S',ylim=ylim)#, pch=point.symb)
#   err = SElogS
#   logspace.plot.error(tt,obj$report()$logS,err)
    gmlogS = median(obj$report()$logS)
    abline(h=gmlogS,lty='dashed')

    ylim = range(obj$report()$logR)
    plot(tt,obj$report()$logR,ylab='ln R',ylim=ylim, pch=point.symb)
#   err = SElogR
#   logspace.plot.error(tt,obj$report()$logR,err)
    gmlogR = median(obj$report()$logR)
    abline(h=gmlogR,lty='dashed')

    ylim = range(obj$report()$loggamma)
#   print(ylim)
    plot(tt,obj$report()$loggamma,ylab='ln gamma',ylim=ylim, pch=point.symb)
    ttext = 0.95*(length(dat$log_obs_cases)-1)
    ytext = make.ytext(ylim,0.9)
    text(ttext,ytext,bias.note("bias_loggamma"),col=note.color,pos=2) 
    err = SEloggamma
#   plot.error(tt,obj$report()$loggamma,err)
    gmloggamma = median(obj$report()$loggamma)
    abline(h=gmloggamma,lty='dashed')

    print(dev.cur())
    dev.copy2pdf(file='ests.pdf',width=6.5,height=6.5)
    print(dev.cur())

    par(old.par)
    detach(fit)
#   return(dev.cur())
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

