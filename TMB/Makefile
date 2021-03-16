#make -j4 -Otarget

.SUFFIXES: .RData .dat
.PHONY: ALL CLEAN RULES file_names
FIT_DIR = /home/jsibert/Projects/SIR-Models/fits
DAT_DIR = /home/jsibert/Projects/SIR-Models/dat
TMB_DIR = /home/jsibert/Projects/SIR-Models/TMB
vpath %.RData $(FIT_DIR)
vpath %.dat $(DAT_DIR)

RR=Rscript --vanilla do_make_runs.R 

FIT_LIST = $(shell find $(DAT_DIR) -name *.dat -type f -printf "%f\n" | sed 's-.dat-.RData-g')
#DAT_LIST = $(shell find $(DAT_DIR) -name *.dat -type f -printf "%f\n")
#MONIKERS = $(shell find $(DAT_DIR) -name *.dat -type f -printf "%f\n" | sed 's-.dat--g')

all_fits: $(FIT_LIST)

%.RData : %.dat
	$(RR) $*

RULES:
	@echo "FIT_LIST: "$(FIT_LIST)
#@echo "MONIKERS: "$(MONIKERS)
#@echo "DAT_LIST: "$(DAT_LIST)
	@echo "RR: "$(RR)

CLEAN:
	@rm -fv *.RData




