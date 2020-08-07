#save.fit = function(data,obj,opt,map,init,file,mod='simpleSIR4')
save.fit = function(fit,file.name,mod='simpleSIR4')
{
    attach(fit)

    sdr = sdreport(obj)
    SElogbeta = as.list(sdr,"Std. Error")$logbeta
    SElogmu   = as.list(sdr,"Std. Error")$logmu
    print(head(SElogbeta))
    print(typeof(SElogbeta))

    diag = data.frame(stringsAsFactors = FALSE,
           obs_cases = data$obs_cases,
           obs_deaths = data$obs_deaths,
           log_obs_cases = data$log_obs_cases,
           log_obs_deaths = data$log_obs_deaths,
           log_pred_cases = obj$report()$logEye,
           log_pred_deaths = obj$report()$logD,
           gamma = obj$report()$gamma,
           logbeta = obj$report()$logbeta,
           logmu = obj$report()$logmu
    #      SElogbeta = SElogbeta,
    #      SElogmu = SElogmu
    )
    print(head(diag))
    print(tail(diag))
    print(typeof(diag))
    print(head(obj$report()$logbeta))
    print(typeof(obj$report()$logbeta))

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

    tinit = vector(length=length(map),mode='numeric')
    test = vector(length=length(map),mode='numeric')
    for (n in 1:length(map))
    {
        nn = names(map)[n]
        tinit[n] = unlist(init)[nn] 
        if (is.na(map[nn]))
        {
            test[n] = unlist(init)[nn] 
        }
        else
        {
            test[n] = opt$par[nn]
        }   
    }

    ests = data.frame(stringsAsFactors = FALSE,
                     names = names(unlist(map)),
                     init = tinit,
                     est  = test
    )

    like_names = c('f','betanll', 'munll', 'Pnll','cnll','dnll')
    like = vector(length=length(like_names))
    like[1] = obj$report()$f
    like[2] = obj$report()$betanll
    if (mod == 'simpleSIR3')
        like[3] = NA
    else
        like[3] = obj$report()$munll
    like[4] = obj$report()$Pnll
    like[5] = obj$report()$cnll
    like[6] = obj$report()$dnll
    
    like_comp = data.frame(stringsAsFactors = FALSE,
                            names = like_names,
                            like = like)

    stderror = data.frame(stringsAsFactors = FALSE,
                           logbeta = SElogbeta,
                           logmu   = SElogmu)
    print(head(stderror))
    print(tail(stderror))

    print(paste('saving fit:',file))
    save(diag,meta,ests,like_comp,file=file)
    save(diag,file="diag.RData")
    save(stderror,file="stderror.RData")
    write.csv(stderror,"stderror.csv")
    write.csv(diag,"diag.csv")

    detach(fit)
}
