save.fit = function(fit,file_root,mod='rrSIR')
{
    print(paste("saving fit for",file_root))
    attach(fit)

    sdr = sdreport(obj)
    SElogbeta = as.list(sdr,"Std. Error")$logbeta
    SEloggamma    = as.list(sdr,"Std. Error")$loggamma
    SElogmu   = as.list(sdr,"Std. Error")$logmu
    print('finished sdr; names:')
    print(names(sdr))
    diag = data.frame(stringsAsFactors = FALSE,
           obs_cases = data$obs_cases,
           obs_deaths = data$obs_deaths,
           log_obs_cases = data$log_obs_cases,
           log_obs_deaths = data$log_obs_deaths,
           log_pred_cases = obj$report()$logEye,
           log_pred_deaths = obj$report()$logD,
           logbeta = obj$report()$logbeta,
           loggamma = obj$report()$loggamma,
           logmu = obj$report()$logmu,
           logR = obj$report()$logR,
	   SElogbeta = NA,
           SEloggamma = NA,
           SElogmu   = NA 
    )
     print('created diag data.frame')
 
    for (r in 1:nrow(diag))
    {
        diag[r,'SElogbeta'] = SElogbeta[r]
        diag[r,'SEloggamma']    = SEloggamma[r]
        diag[r,'SElogmu']   = SElogmu[r]
    }
    print('---- finished diag:')
#   print(diag)

    if (is.null(opt$value))
        data = c(data$county,data$update_stamp,data$N0,data$Date0,data$ntime,
                       data$prop_zero_deaths,opt$objective,opt$convergence,mod)
    else
        data = c(data$county,data$update_stamp,data$N0,data$Date0,data$ntime,
                        data$prop_zero_deaths,opt$value,opt$convergence,mod)
    meta = data.frame(stringsAsFactors = FALSE,
           names = c("county","update_stamp","N0","Date0","ntime","prop_zero_deaths",
                      "fn","convergence","model"),
           data = data
    ) 
    print('---- finished meta:')
    print(meta)

#   print(paste('length of map',length(map)))
#   tinit = vector(length=length(map),mode='numeric')
#   test = vector(length=length(map),mode='numeric')
#   for (n in 1:length(map))
#   {
#       nn = names(map)[n]
#       tinit[n] = unlist(init)[nn] 
#       print(paste(n,nn,tinit[n],is.na(map[nn]),dim(tinit)))
#       if (is.na(map[nn]))
#       {
#           test[n] = tinit[nn] 
#       }
#       else
#       {
#           test[n] = opt$par[nn]
#       }   
#   }

#   print('est names:')
#   print(names(map))
#   ests = data.frame(stringsAsFactors = FALSE,
#                    names = names(map),
#                    init = tinit,
#                    est  = test,
#                    map = map
#   )

    enames = names(obj$par)
    einit = obj$par
    est   = opt$par
    ests = data.frame(stringsAsFactors = FALSE,
                      names = enames,
                      init = einit,
                      est  = est
    )
    print('---- finished ests:')
    print(ests)

    like_names = c('f','betanll', 'gammanll', 'munll', 'Pnll','cnll','dnll')
    like = vector(length=length(like_names))
    like[1] = obj$report()$f
    like[2] = obj$report()$betanll
    if (mod == 'simpleSIR3')
        like[3] = NA
    else
        like[3] = obj$report()$gammanll
    like[4] = obj$report()$munll
    like[5] = obj$report()$Pnll
    like[6] = obj$report()$cnll
    like[7] = obj$report()$dnll
    
    like_comp = data.frame(stringsAsFactors = FALSE,
                            names = like_names,
                            like = like)
    print('---- finished like:')
    print(like)

    pyreadr_kludge = FALSE
    if (pyreadr_kludge)
    {
        stderror = data.frame(stringsAsFactors = FALSE,
                              SElogbeta = SElogbeta,
                              SElogmu   = SElogmu)
        csv.file=paste(fit_path,file_root,'_stderror.csv',sep='')
        print(paste('saving stderror:',csv.file))
        write.csv(stderror,csv.file)
    }


    rd.file = paste(fit_path,file_root,'.RData',sep='')
    print(paste('saving fit:',rd.file))
    save(diag,meta,ests,like_comp,file=rd.file)
    print('     saved')


#   stderror = data.frame(stringsAsFactors = FALSE,
#                             SElogbeta = as.vector(unlist(SElogbeta)),
#                             SElogmu   = as.vector(unlist(SElogmu)))

#   rd.file = paste(fit_path,'stderror.RData',sep='')
#   print(paste('Now',rd.file))
#   print(stderror)
#   save(stderror,file=rd.file)

    detach(fit)
}
