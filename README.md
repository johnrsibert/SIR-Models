# SIR-Models
Statistical models derived from classic SIR epidemiology models and adapted to be suitable for estimation using readily available data on the Covid-19 epidemic in the United States.

## Files and directores

* Directories
  * ADMB - ADMB and C++ source code for development of statiostical SIR models using ADModel Builder. See http://www.admb-project.org/
  * TMB - R scripts and C++ source code for development of statiostical SIR models using Template Model Builder. See https://github.com/kaskr/adcomp and http://www.admb-project.org/
  * References - a few literature references to SIR models and Covid-19
  * Reports - preliminary write up of simpleSIR4 model in TMB, drafted August 2019
  * assets - graphics files suporting https://johnrsibert.github.io/JonzPandemic/
  * dash - abandoned attempt to generate interactive plot selection
  * plots to share - primarily prevelence graphs of the pandemic in selected geographies (flagged 's' in GeogIndex.csv)
  * python - Python 3.x scripts and packages for extracting data and making pictures
* Root Data File
  * GeogIndex.csv - list of New York Times "geographies" (primarily US counties) with names, state postal codes, and FIPS codes that are consistent with those used by the United State Census. Also contains census estimates of population size for each US county. The 'flag' column is used by data selection software for various purposes.
