logspace.plot.error=function(x,logy,sd,bcol='black',fcol='gray',mult=2)
{
   if (capabilities("cairo"))
   {
      sdyu = logy+mult*sd
      sdyl = logy-mult*sd
      frgb = col2rgb(fcol)/255
      
      polygon(c(x,rev(x)),c(sdyl,rev(sdyu)),
              border=bcol,lty="solid",lwd=1,
              col=rgb(frgb[1],frgb[2],frgb[3],0.5))
   }
   else
      polygon(c(x,rev(x)),c(sdyl,rev(sdyu)),
              border=bcol,lty="dashed",lwd=1,col=fcol)

}

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

plot.log.state = function(dat,par,obj,opt,map,np = 4)
{
    fn = opt$value
    if (is.null(fn))
        fn = opt$objective
    title =  paste(dat$county,", f = ",sprintf('%.5g',fn),
                   ", converge = ", opt$convergence,
                   " (",(opt$convergence==0),")",sep='')
    old.par = par(no.readonly = TRUE) 
    par(mar=c(2,4.5,2,4)+0.1)
    note.color='blue'
    point.symb = '+'

    lm = layout(matrix(c(1:np),ncol=1,byrow=TRUE))
    layout.show(lm)

    tt = seq(0,(length(dat$log_obs_cases)-1))
    ttext = 0.1*(length(dat$log_obs_cases)-1)

    ylim=c(0.0,max(dat$log_obs_cases,obj$report()$logEye))
    poplim = ylim
    plot(tt,dat$log_obs_cases,ylab='log cases,',ylim=ylim, pch=point.symb)
    lines(tt,obj$report()$logEye,col='red')
    err = exp(get.error(par,opt,map,'logsigma_logC'))
    logspace.plot.error(tt,obj$report()$logEye,err)
    ytext = make.ytext(ylim,0.9)
    note = paste('sigma_logC ~',sprintf("%.5g",err))
    text(ttext,ytext,note,col=note.color,pos=4)
    title(main=title,sub='sub')

#   ylim = c(0.0,dat$beta_b)#range(obj$report()$beta)
    ylim = c(0.0,max(obj$report()$beta))
    plot(tt,obj$report()$beta,ylab='beta',ylim=ylim, pch=point.symb)
    err = exp(get.error(par,opt,map,'logsigma_beta'))
    plot.error(tt,obj$report()$beta,err)
    gmbeta = median(obj$report()$beta)
    abline(h=gmbeta,lty='dashed')
    ytext = make.ytext(ylim,0.9)
    note = paste('sigma_beta ~',sprintf("%.5g",err))
    text(ttext,ytext,note,col=note.color,pos=4)
 
    ylim=poplim #c(0.0,max(dat$log_obs_deaths,obj$report()$logD))
    plot(tt,dat$log_obs_deaths,ylab='log deaths',ylim=ylim, pch=point.symb)
    lines(tt,obj$report()$logD,col='red')
    err = exp(get.error(par,opt,map,'logsigma_logD'))
    logspace.plot.error(tt,obj$report()$logD,err)
    ytext = make.ytext(ylim,0.9)
    note = paste('sigma_logD ~',sprintf("%.5g",err))
    text(ttext,ytext,note,col=note.color,pos=4)

    if (np > 3)
    {
        ylim = c(0.0,max(obj$report()$mu))
        plot(tt,obj$report()$mu,ylab='mu',ylim=ylim, pch=point.symb)
        err = exp(get.error(par,opt,map,'logsigma_mu'))
        plot.error(tt,obj$report()$mu,err)
        gmmu = median(obj$report()$mu)
        abline(h=gmmu,lty='dashed')
        ytext = make.ytext(ylim,0.9)
        note = paste('sigma_mu ~',sprintf("%.5g",err))
        text(ttext,ytext,note,col=note.color,pos=4)
    }
#   print(dev.cur())
#   dev.copy2pdf(file='ests.pdf',width=6.5,height=6.5)
#   print(dev.cur())

    par(old.par)
#   return(dev.cur())
}

plot.state=function(dat,oo,mod.par,np = 5)
{
    old.par = par(no.readonly = TRUE) 
    par(mar=c(3,4.5,0,4)+0.1)

    lm = layout(matrix(c(1:np),ncol=1,byrow=TRUE))
    layout.show(lm)

    tt = seq(0,(length(dat$obs_cases)-1))

    plot(tt,log(dat$obs_cases),ylab='log cases,',
            ylim=c(0.0,max(log(dat$obs_cases),log(oo$report()$Eye))))
    lines(tt,log(oo$report()$Eye),col='red')
    plot.error(tt,log(oo$report()$Eye),mod.par$logsigma_C)

    plot(tt,log(dat$obs_deaths),ylab='log deaths',
            ylim=c(0.0,max(dat$obs_deaths,oo$report()$D)))
    lines(tt,log(oo$report()$D),col='red')
    plot.error(tt,log(oo$report()$D),mod.par$logsigma_D)

    plot(tt,exp(oo$report()$logbeta),ylab='beta',
         ylim = range(exp(oo$report()$logbeta)))
    plot.error(tt,exp(oo$report()$logbeta),exp(mod.par$logsigma_beta))
 
    if (np > 3)
    {
        plot(oo$report()$S,ylab='susceptible', ylim=c(0,dat$N0))

        plot(oo$report()$R,ylab='recovered')
    }
    dev.copy2pdf(file='ests.pdf',width=6.5,height=6.5)

    par(old.par)
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


write.report=function(dat,obj,opt)
{
   repfile = paste(dat$county,".rep",sep="")
   print(paste("writing",repfile))
   cat("# parameters:",file=repfile,fill=TRUE,append=FALSE)
   npar = length(opt$par)
   cat(paste("parameter","initial","estimate",sep=","),
         file=repfile,fill=TRUE,append=TRUE)
   for (p in 1:npar)
   {
       cat(paste(names(obj$par[p]),obj$par[p],opt$par[p],sep=","),
             file=repfile,fill=TRUE,append=TRUE)
   } 
  
   
   ntime = dat$ntime
   cat(paste("log_obs_cases","log_est_cases","log_obs_deaths","log_est_deaths",
             "log_beta",sep=","),file=repfile,fill=TRUE,append=TRUE)
   for (t in 1:ntime)
   {
       cat(paste(dat$log_obs_cases[t],obj$report()$logEye[t],
                 dat$log_obs_deaths[t],obj$report()$logD[t],
                 obj$report()$logbeta[t], sep=","),
             file=repfile,fill=TRUE,append=TRUE)
   }


}
