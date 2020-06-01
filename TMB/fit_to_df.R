save.fit = function(data,obj,opt,map,init,file)
{
    diag = data.frame(stringsAsFactors = FALSE,
           obs_cases = data$obs_cases,
           obs_deaths = data$obs_deaths,
           log_obs_cases = data$log_obs_cases,
           log_obs_deaths = data$log_obs_deaths,
           log_pred_cases = obj$report()$logEye,
           log_pred_deaths = obj$report()$logD,
           beta = obj$report()$beta,
           mu = obj$report()$mu
    )

    if (is.null(opt$value))
        data = c(data$county,data$update_stamp,data$N0,data$Date0,data$ntime,
                       data$prop_zero_deaths,opt$objective,opt$convergence)
    else
        data = c(data$county,data$update_stamp,data$N0,data$Date0,data$ntime,
                        data$prop_zero_deaths,opt$value,opt$convergence)
    meta = data.frame(stringsAsFactors = FALSE,
           names = c("county","update_stamp","N0","Date0","ntime","prop_zero_deaths",
                      "fn","convergence"),
           data = data
    ) 

    est_names = names(init)
    init = unlist(init)
    est = vector(length=length(init))
    est = init
    for (n in names(opt$par))
    {
         est[n] = opt$par[n]
    }

    ests = data.frame(stringsAsFactors = FALSE,
                     names = est_names,
                     init = init,
                     est = est)

    like_names = c('f','betanll', 'munll', 'Pnll','cnll','dnll')
    like = vector(length=length(like_names))
    like[1] = obj$report()$f
    like[2] = obj$report()$betanll
    like[3] = obj$report()$munll
    like[4] = obj$report()$Pnll
    like[5] = obj$report()$cnll
    like[6] = obj$report()$dnll

    print(paste(like[1],sum(like)))
    
    like_comp = data.frame(stringsAsFactors = FALSE,
                            names = like_names,
                            like = like)

#   tod = format(Sys.time(), "%Y%m%d%H%M%S")
#   file = paste(data$county,'_',tod,'.RData',sep='')
#   file = "Los_Angeles_20200512111526.RData"

    print(paste('saving fit:',file))
    save(diag,meta,ests,like_comp,file=file)
}

# tnames=names(fit$obj$par)
