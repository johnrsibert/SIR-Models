#save.fit = function(data,obj,opt,map,init,file,mod='simpleSIR4')
save.fit = function(fit,file_root,mod='simpleSIR4')
{
    attach(fit)

    sdr = sdreport(obj)
 
    SElogbeta = as.vector(unlist(as.list(sdr,"Std. Error")$logbeta))
    SElogmu   = as.list(sdr,"Std. Error")$logmu
#   print("finished SElogmu")
#   SErho     = as.list(sdr,"Std. Error")$rho
#   print("finished SErho")
#   print(SErho)

    diag = data.frame(stringsAsFactors = FALSE,
           obs_cases = data$obs_cases,
           obs_deaths = data$obs_deaths,
           log_obs_cases = data$log_obs_cases,
           log_obs_deaths = data$log_obs_deaths,
           log_pred_cases = obj$report()$logEye,
           log_pred_deaths = obj$report()$logD,
           logbeta = obj$report()$logbeta,
           logmu = obj$report()$logmu,
           rho = obj$report()$rho,
	   SElogbeta = NA,
           SElogmu = NA
    #      SErho = NA
    )
 
    for (r in 1:nrow(diag))
    {
        diag[r,'SElogbeta'] = SElogbeta[r]
        diag[r,'SElogmu'] = SElogmu[r]
    #   print(paste("finished SElogmu, row",r))
    #   diag[r,'SErho'] = SErho[r]
    #   print("finished SErho 2")
    }

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

    print(paste('init:',names(init)))
    print(paste('opt$par"',names(opt$par)))
#   HPnames = list('logsigma_logCP', 'logsigma_logDP',
#                  'logsigma_logbeta', 'logsigma_logmu',
#                  'logsigma_logC', 'logsigma_logD ')
    HPnames = names(init)

    nHP = length(HPnames)
    tinit = as.vector(unlist(init))
    test  = vector(length=nHP)
    tmap  = vector(length=nHP)
    print(init)

    for (n in 1:nHP)
    {
        pname = HPnames[n]
        print(paste(n,pname))
        tmap[n] = 'NA'
    #   tinit[n] = init[pname]
        if (is.na(map[pname]))
        {
            test[n] = tinit[n]
            tmap[n] = 'NA'
        }
        else
        {
            test[n] = opt$par[pname]
            tmap[n] = 1
        }
    }
    print(tinit)
    print(test)
    print(tmap)
        
    ests = data.frame(stringsAsFactors = FALSE,
                     names = HPnames,
                     init = tinit,
                     est  = test,
                     map = tmap
    )
    print('ests:')
    print(ests)
    print('not ests:')

#   tinit = vector(length=length(map),mode='numeric')
#   test = vector(length=length(map),mode='numeric')
#   for (n in 1:length(map))
#   {
#       nn = names(map)[n]
#       tinit[n] = unlist(init)[nn] 
#       if (is.na(map[nn]))
#       {
#           test[n] = unlist(init)[nn] 
#       }
#       else
#       {
#           test[n] = opt$par[nn]
#       }   
#   }

#   ests = data.frame(stringsAsFactors = FALSE,
#                    names = names(unlist(map)),
#                    init = tinit,
#                    est  = test,
#                    map = unlist(map)
#   )
#   print(ests)

    like_names = c('f','betanll', 'munll', 'Pnll','cnll','dnll')
    like = vector(length=length(like_names))
    like[1] = obj$report()$f
    like[2] = obj$report()$betanll
    like[3] = obj$report()$munll
    like[4] = obj$report()$Pnll
    like[5] = obj$report()$cnll
    like[6] = obj$report()$dnll
    
    like_comp = data.frame(stringsAsFactors = FALSE,
                            names = like_names,
                            like = like)

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


#   stderror = data.frame(stringsAsFactors = FALSE,
#                             SElogbeta = as.vector(unlist(SElogbeta)),
#                             SElogmu   = as.vector(unlist(SElogmu)))

#   rd.file = paste(fit_path,'stderror.RData',sep='')
#   print(paste('Now',rd.file))
#   print(stderror)
#   save(stderror,file=rd.file)

    detach(fit)
}
