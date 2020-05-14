save.fit = function(data,obj,opt,map,file)
{
    diag = data.frame(
           stringsAsFactors = FALSE,
           obs_cases = data$obs_cases,
           obs_deaths = data$obs_deaths,
           log_obs_cases = data$log_obs_cases,
           log_obs_deaths = data$log_obs_deaths,
           log_pred_cases = obj$report()$logEye,
           log_pred_deaths = obj$report()$logD,
           logbeta = obj$report()$logbeta,
           logmu = obj$report()$logmu
    )

    meta = data.frame(stringsAsFactors = FALSE,
           names = c("county","N0","Date0","ntime","prop_zero_deaths",
                      "fn","convergence"),
           data = c(data$county,data$N0,data$Date0,data$ntime,
                    data$prop_zero_deaths,opt$value,opt$convergence)
    ) 

    est = data.frame(stringsAsFactors = FALSE,
         names = c("sigma_logP","sigma_logbeta","logmu","loggamma",
                   "sigma_logC","sigma_logD"),
         obs  = c(obj$par['sigma_logP'],obj$par['sigma_logbeta'],obj$par['logmu'],
                  obj$par['loggamma'],obj$par['sigma_logC'],obj$par['sigma_logD']),
         est = c(opt$par['sigma_logP'],opt$par['sigma_logbeta'],opt$par['logmu'],
                  opt$par['loggamma'], opt$par['sigma_logC'],opt$par['sigma_logD'])
    )


#   tod = format(Sys.time(), "%Y%m%d%H%M%S")
#   file = paste(data$county,'_',tod,'.RData',sep='')
#   file = "Los_Angeles_20200512111526.RData"

#   print(paste('saving fit:',file))

    save(diag,meta,est,file=file)
}
