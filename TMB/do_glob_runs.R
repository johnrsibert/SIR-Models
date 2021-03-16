args = commandArgs(trailingOnly=TRUE)
source('./simpleSIR4.R')
#test = function(moniker='AlamedaCA')
print(args)
print(paste('running  SIR model on',args[1]))
glob_runs(TRUE)->fit
Sys.sleep(10)
print(paste('Finished SIR model on',args[1]))
