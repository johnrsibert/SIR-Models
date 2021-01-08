
read.dat.file=function(dat.file = "",max_ntime = 1000)
{
   field.counter <<- 0
   print(paste("Reading ",dat.file))

   get.field = function()
   {
     field.counter <<- field.counter + 1
     field = sca[field.counter]
   # print(paste(field.counter,field))
     return(field)
   }
   
   get.numeric.field<-function()
   {
      ret = as.numeric(get.field())
      return(ret)
   }

   sca = scan(file=dat.file,comment.char="#",what="raw",quiet=TRUE)
   print(paste("Read",length(sca),"items from ",dat.file))

   data = list()
   data$county = get.field()        # county name
   data$update_stamp = get.field()  # date data downloaded from source
   data$N0 = get.numeric.field()    # total population
   data$Date0 = get.field()         # calander date of first case
   data$ntime = get.numeric.field() # number of dat records

   if (data$ntime > max_ntime)
      data$ntime = max_ntime

   ntime = data$ntime
   nobs  = ntime + 1
   data$obs_cases  = vector(len=ntime,mode="numeric")
   data$obs_deaths = vector(len=ntime,mode="numeric")

   for (t in 1:nobs)
   {
      data$obs_cases[t]  = get.numeric.field()# + tiny
      data$obs_deaths[t] = get.numeric.field()# + tiny
   }

   data$prop_zero_deaths = length(which(data$obs_deaths < 1.0))/length(data$obs_deaths)

   return(list(data=data))
}   
