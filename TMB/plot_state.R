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

plot.log.state = function(fit,file_root,np = 5,remove_plot=FALSE)
{
    attach(fit)
    fn = opt$value
    if (is.null(fn))
        fn = opt$objective
    title =  paste(data$county,", f = ",sprintf('%.5g',fn),
                   ", converge = ", opt$convergence,
                   " (",(opt$convergence==0),")",sep='')
    width = 6.5
    height = 9.0
    x11(width=width,height=height,title=title)#,xpos=100)

    old.par = par(no.readonly = TRUE) 
    par(mar=c(2,4.5,2,4)+0.1)
    note.color='blue'
    point.symb = '+'

    lm = layout(matrix(c(1:np),ncol=1,byrow=TRUE))
    layout.show(lm)
    tt = seq(0,(length(data$log_obs_cases)-1))
    ttext = 0.1*(length(data$log_obs_cases)-1)

    ylim=c(0.0,max(data$log_obs_cases,obj$report()$logEye))
    poplim = ylim
    plot(tt,data$log_obs_cases,ylab='ln cases,',ylim=ylim, pch=point.symb)
    lines(tt,obj$report()$logEye,col='red')
    err = exp(get.error(par,opt,map,'logsigma_logC'))
    logspace.plot.error(tt,obj$report()$logEye,err)
    ytext = make.ytext(ylim,0.9)
    note = paste('sigma_logC ~',sprintf("%.5g",err))
    text(ttext,ytext,note,col=note.color,pos=4)
    title(main=title,sub='sub')

    ylim=poplim #c(0.0,max(data$log_obs_deaths,obj$report()$logD))
    plot(tt,data$log_obs_deaths,ylab='ln deaths',ylim=ylim, pch=point.symb)
    lines(tt,obj$report()$logD,col='red')
    err = exp(get.error(par,opt,map,'logsigma_logD'))
    logspace.plot.error(tt,obj$report()$logD,err)
    ytext = make.ytext(ylim,0.9)
    note = paste('sigma_logD ~',sprintf("%.5g",err))
    text(ttext,ytext,note,col=note.color,pos=4)

    ylim = 1.2*range(obj$report()$logbeta)
    print(ylim)
    plot(tt,obj$report()$logbeta,ylab='ln beta',ylim=ylim, pch=point.symb)
    err = exp(get.error(par,opt,map,'logsigma_logbeta'))
    logspace.plot.error(tt,obj$report()$logbeta,err)
    gmlogbeta = median(obj$report()$logbeta)
    abline(h=gmlogbeta,lty='dashed')
    ytext = make.ytext(ylim,0.9)
    note = paste('sigma_logbeta ~',sprintf("%.5g",err))
    text(ttext,ytext,note,col=note.color,pos=4)
 
    if (np > 3)
    {
        ylim = 1.2*range(obj$report()$logmu)
        plot(tt,obj$report()$logmu,ylab='ln mu', pch=point.symb)
        err = exp(get.error(par,opt,map,'logsigma_logmu'))
        logspace.plot.error(tt,obj$report()$logmu,err)
        gmlogmu = median(obj$report()$logmu)
        abline(h=gmlogmu,lty='dashed')
        ytext = make.ytext(ylim,0.9)
        note = paste('sigma_logmu ~',sprintf("%.5g",err))
        text(ttext,ytext,note,col=note.color,pos=4)

        ylim = 1.2*range(obj$report()$rho)
        plot(tt,obj$report()$rho,ylab='rho', ylim=c(1e-1,5000),pch=point.symb, log='y')
        gmrho = median(obj$report()$rho)
        abline(h=gmrho,lty='dashed')
        abline(h=1.0,lty='dashed',col='red')
    }

    file =paste(fit_path,file_root,'.png',sep='')
    print(paste('Attempting to save plot as',file))
    dev.copy(png,file=file,width=6.5,height=9,unit='in',res=300)
    print(paste('State plot saved as',file))
    if (remove_plot)
        graphics.off()



    par(old.par)
    detach(fit)
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
