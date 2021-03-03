#include <TMB.hpp>
#include <math.h>
#include <fenv.h> 
#include "trace.h"

const double TWO_M_PI = 2.0*M_PI;
const double eps = 1e-8;
const double logeps = log(eps);

template < class Type > Type square(Type x)
{
    return x * x;
}

// log-normal error
template <class Type>
Type NLerr(Type logobs, Type logpred, Type var)
{
    Type nll = 0.5*(log(TWO_M_PI*var) + square(logobs-logpred)/var);
    return nll;
}

// zero-inflated log-normal error
template <class Type>
Type ZILNerr(Type logobs, Type logpred, Type var, Type prop0 = 0.15)
{
    Type nll;

    if (logobs > logeps)  //log zero deaths
    {
        nll = (1.0-prop0)*0.5*(log(TWO_M_PI*var) + square(logobs - logpred)/var);
    }
    else
    {
        nll = prop0*0.5*(log(TWO_M_PI*var));
    }
    return nll;
}

/*
template < class Type > Type isNaN(Type x, const int line)
{
    if (x != x)
    {
         std::cerr << "NaN at line " << line << std::endl;
         exit(1);
    }
    return x;
}
*/

template<class Type> 
Type objective_function <Type>::operator()()
{
// feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO | FE_UNDERFLOW);

    DATA_SCALAR(N0)
    DATA_INTEGER(ntime)
    DATA_VECTOR(log_obs_cases)
    DATA_VECTOR(log_obs_deaths)
    DATA_SCALAR(prop_zero_deaths)
//  DATA_SCALAR(beta_a)
//  DATA_SCALAR(beta_b)
//  DATA_SCALAR(mu_a)
//  DATA_SCALAR(mu_b)


    PARAMETER(logsigma_logCP);      // SIR process error
    PARAMETER(logsigma_logDP);      // SIR process error
    PARAMETER(logsigma_logbeta);    // beta random walk sd
    PARAMETER(logsigma_logmu);      // mu randomwalk sd

    PARAMETER(logsigma_logC);       // cases observation error
    PARAMETER(logsigma_logD);       // deaths observation error

    PARAMETER_VECTOR(logbeta);      // infection rate time series
  //PARAMETER(loggamma);            // recovery rate time series
    PARAMETER_VECTOR(logmu);        // mortality rate of infection population

    // state variables
    vector <Type> logEye(ntime+1);  // number of infections
    vector <Type> logD(ntime+1);    // number of deaths from infected population

  //vector <Type> gamma(ntime+1);   // recovery rate
    vector <Type> rho(ntime+1);     // fake 'reproduction number'

    Type sigma_logbeta = exp(logsigma_logbeta); 
    Type sigma_logmu = exp(logsigma_logmu); 
    Type sigma_logCP = exp(logsigma_logCP);
    Type sigma_logDP = exp(logsigma_logDP);
    Type sigma_logC = exp(logsigma_logC);
    Type sigma_logD = exp(logsigma_logD);

    Type var_logbeta = square(sigma_logbeta);
    Type var_logmu = square(sigma_logmu);
    Type var_logCP = square(sigma_logCP);
    Type var_logDP = square(sigma_logDP);
    Type var_logC = square(sigma_logC);
    Type var_logD = square(sigma_logD);

    Type f = 0.0;
    Type betanll = 0.0;
    Type munll = 0.0;
    Type Pnll = 0.0;
    Type cnll = 0.0;
    Type dnll = 0.0;

    //  loop over time
    //gamma[0] = exp(loggamma);
    for (int t = 1; t <= ntime; t++)
    {
         // infection rate random walk
       //betanll += isNaN(NLerr(logbeta(t-1),logbeta(t),var_logbeta),__LINE__);
         betanll += NLerr(logbeta(t-1),logbeta(t),var_logbeta);

         // mortality rate random walk
       //munll += isNaN(NLerr(logmu(t-1),logmu(t),var_logmu),__LINE__);
         munll += NLerr(logmu(t-1),logmu(t),var_logmu);

         // cases process error
         Type prevEye = exp(logEye(t-1));
         logEye(t) = log(prevEye*(1.0 + (exp(logbeta(t-1)) -  //gamma(t-1) - 
                         exp(logmu(t-1))))+eps);
       //Pnll += isNaN(NLerr(logEye(t-1), logEye(t),var_logCP),__LINE__);
         Pnll += NLerr(logEye(t-1), logEye(t),var_logCP);

       //gamma(t) = exp(loggamma);

         // deaths process error
         Type prevD = exp(logD(t-1));
         logD(t) = log(prevD + exp(logmu(t-1))*exp(logEye(t-1))+eps);
       //Pnll += isNaN(ZILNerr(logD(t-1), logD(t), var_logDP, prop_zero_deaths),__LINE__);
         Pnll += ZILNerr(logD(t-1), logD(t), var_logDP, prop_zero_deaths);

     }
 
     // compute observation likelihoods
     for (int t = 0; t <= ntime; t++)
     {   
       //cnll += isNaN(  NLerr(log_obs_cases(t),logEye(t),var_logC),__LINE__);
         cnll += NLerr(log_obs_cases(t),logEye(t),var_logC);

       //dnll += isNaN(ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths),__LINE__);
         dnll += ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths);
     }

     // total likelihood
   //f += isNaN((betanll + munll + Pnll + cnll + dnll),__LINE__);
     f += (betanll + munll + Pnll + cnll + dnll);

     REPORT(logEye)
     REPORT(logD)
     REPORT(logbeta)
     REPORT(logmu)
     rho = exp(logbeta -logmu);
     REPORT(rho)

     REPORT(sigma_logCP);
     REPORT(sigma_logDP);
     REPORT(sigma_logbeta);
     REPORT(sigma_logmu);
  // REPORT(gamma);

     REPORT(f);
     REPORT(betanll);
     REPORT(munll);
     REPORT(Pnll);
     REPORT(cnll);
     REPORT(dnll);

   //return isNaN(f,__LINE__);
     return (f);
}
