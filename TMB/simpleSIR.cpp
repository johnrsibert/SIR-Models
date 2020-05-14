#include <TMB.hpp>
#include <math.h>
#include <fenv.h> 
const double TWO_M_PI = 2.0*M_PI;
const double LOG_M_PI = log(M_PI);
const double eps = 1e-8;
const double logeps = log(eps);

//#include <fstream>
//std::ofstream Clogf;
//Clogf.open("TMB_error.log");
#include "trace.h"

/*
void report_error(const int line)
{
    std::cerr << "********* exception at line " << line << std::endl;
}
*/


template < class Type > Type square(Type x)
{
    return x * x;
}

// zero-inflated log-normal error
template <class Type>
Type ZILNerr(Type logobs, Type logpred, Type var, Type prop0 = 0.15)
{
    Type tmp;

    if (logobs > logeps)  //log zero deaths
    {
        tmp = (1.0-prop0)*0.5*(log(TWO_M_PI*var) + square(logobs - logpred)/var);
    }
    else
    {
        tmp = prop0*0.5*(log(TWO_M_PI*var));
    }


    return tmp;
}

// log-normal error
template <class Type>
Type NLerr(Type logobs, Type logpred, Type var)
{
    Type nll = 0.5*(log(TWO_M_PI*var) + square(logobs-logpred)/var);
    return nll;
}

template < class Type > Type isNaN(Type x, const int line)
{
    if (x != x)
    {
         std::cerr << "NaN at line " << line << std::endl;
         exit(1);
    }
    return x;
}


template<class Type> 
Type objective_function <Type>::operator()()
{
 //feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO | FE_UNDERFLOW);

    DATA_SCALAR(N0)
    DATA_INTEGER(ntime)
    DATA_VECTOR(log_obs_cases)
    DATA_VECTOR(log_obs_deaths)
    DATA_SCALAR(prop_zero_deaths)

    PARAMETER(sigma_logP);          // SIR process error
    PARAMETER(sigma_logbeta);       // beta random walk sd
    PARAMETER(sigma_logmu);
    PARAMETER(loggamma);            // recovery rate of infection population
    PARAMETER(sigma_logC);          // cases observation error
    PARAMETER(sigma_logD);          // deaths observation error

    PARAMETER_VECTOR(logbeta);      // infection rate time series
    PARAMETER_VECTOR(logmu);               // mortality rate of infection population

    // state variables
    vector <Type> logEye(ntime);       // number of infections
    vector <Type> logD(ntime);         // number of deaths from infected population

  //Type mu = exp(logmu);
    Type gamma = exp(loggamma);
  //Type sigma_P = exp(logsigma_logP);
  //Type sigma_C = exp(logsigma_logC);
  //Type sigma_D = exp(logsigma_logD);
  //Type sigma_beta = exp(logsigma_logbeta);

    Type var_logbeta = square(sigma_logbeta);
    Type var_logmu = square(sigma_logmu);
    Type var_logPop = square(sigma_logP);
    Type var_logcases = square(sigma_logC);
    Type var_logdeaths = square(sigma_logD);

    Type f = 0.0;

    //  loop over time
    //logEye(0) = log_obs_cases(0);
    for (int t = 1; t <  ntime; t++)
    {
         // infection rate random walk
         Type betanll = 0.0;
         betanll += isNaN(NLerr(logbeta(t-1),logbeta(t),var_logbeta),__LINE__);

         Type munll = 0.0;
         munll += isNaN(NLerr(logmu(t-1),logmu(t),var_logmu),__LINE__);

         Type Pnll = 0.0;
         Type prevEye = exp(logEye(t-1));
         logEye(t) = log(prevEye*(1.0 + (exp(logbeta(t-1)) - gamma - exp(logmu(t-1))))+eps);
         Pnll += isNaN(NLerr(logEye(t-1), logEye(t),var_logPop),__LINE__);

         logD(t) = logmu(t-1) + logEye(t-1);
       //Pnll += isNaN(ZILNerr(logD(t-1), logD(t), var_logPop),__LINE__);

         f += isNaN((betanll + munll + Pnll),__LINE__);
     }
 
     // compute observation likelihoods
     Type cnll = 0.0;
     Type dnll = 0.0;
     for (int t = 0; t < ntime; t++)
     {   
         cnll += isNaN(NLerr(log_obs_cases(t),logEye(t),var_logcases),__LINE__);

         dnll += isNaN(ZILNerr(log_obs_deaths(t),logD(t),var_logdeaths, prop_zero_deaths),__LINE__);
     }

     f += isNaN((cnll + dnll),__LINE__);

     REPORT(logEye)
     REPORT(logD)
     REPORT(logbeta)
     REPORT(logmu)
   //REPORT(loggamma)
     /* 
     ADREPORT(sigma_logP);
     ADREPORT(sigma_logbeta);
     ADREPORT(logmu);
     ADREPORT(loggamma);
     ADREPORT(sigma_logC);
     ADREPORT(sigma_logD); 
     */
     return isNaN(f,__LINE__);
}
