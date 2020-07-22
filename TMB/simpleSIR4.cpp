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


    PARAMETER(logsigma_logP);          // SIR process error
    PARAMETER(logsigma_logbeta);       // beta random walk sd
    PARAMETER(logsigma_logmu);         // mu randomwalk sd
    PARAMETER(logsigma_logC);          // cases observation error
    PARAMETER(logsigma_logD);          // deaths observation error

    PARAMETER_VECTOR(logbeta);      // infection rate time series
    PARAMETER_VECTOR(logmu);        // mortality rate of infection population

    /*
    vector <Type> beta(ntime+1);
    vector <Type> mu(ntime+1);
    for (int t = 0; t <=  ntime; t++)
    {
     // Type u = a + (b - a)*invlogit(stlogit_u);
     // beta[t] = beta_a + (beta_b - beta_a)*invlogit(logitbeta[t]);
     // mu[t] =   mu_a + (mu_b - mu_a)*invlogit(logitmu[t]);
        beta[t] = exp(logbeta[t]);
        mu[t] = exp(logmu[t]);
    }
    */

    // state variables
    vector <Type> logEye(ntime+1);    // number of infections
    vector <Type> logD(ntime+1);      // number of deaths from infected population

    vector <Type> gamma(ntime+1);

    Type sigma_logbeta = exp(logsigma_logbeta); 
    Type sigma_logmu = exp(logsigma_logmu); 
    Type sigma_logP = exp(logsigma_logP);
    Type sigma_logC = exp(logsigma_logC);
    Type sigma_logD = exp(logsigma_logD);

    Type var_logbeta = square(sigma_logbeta);
    Type var_logmu = square(sigma_logmu);
    Type var_logP = square(sigma_logP);
    Type var_logC = square(sigma_logC);
    Type var_logD = square(sigma_logD);

    Type f = 0.0;
    Type betanll = 0.0;
    Type munll = 0.0;
    Type Pnll = 0.0;
    Type cnll = 0.0;
    Type dnll = 0.0;

    //  loop over time
//  logEye(0) = log_obs_cases(0);
//  logD(0) = log_obs_deaths(0);
    gamma[0] = 1e-8;
    for (int t = 1; t <= ntime; t++)
    {
         // infection rate random walk
         betanll += isNaN(NLerr(logbeta(t-1),logbeta(t),var_logbeta),__LINE__);

         // mortality rate random walk
         munll += isNaN(NLerr(logmu(t-1),logmu(t),var_logmu),__LINE__);

         // cases process error
         Type prevEye = exp(logEye(t-1));
         logEye(t) = log(prevEye*(1.0 + (exp(logbeta(t-1)) - gamma(t-1) - 
                         exp(logmu(t-1))))+eps);
         Pnll += isNaN(NLerr(logEye(t-1), logEye(t),var_logP),__LINE__);

         gamma(t) = exp(logbeta(t-1)) - exp(logmu(t-1)) - exp(logEye(t))/prevEye + 1.0;
     //  gamma(t) = 1e-8;

         // deaths process error
         Type prevD = exp(logD(t-1));
         logD(t) = log(prevD + exp(logmu(t-1))*exp(logEye(t-1))+eps);
         Pnll += isNaN(ZILNerr(logD(t-1), logD(t), var_logP, prop_zero_deaths),__LINE__);

     }
 
     // compute observation likelihoods
     for (int t = 0; t <= ntime; t++)
     {   
         cnll += isNaN(  NLerr(log_obs_cases(t),logEye(t),var_logC),__LINE__);

         dnll += isNaN(ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths),__LINE__);
     }

     // total likelihood
     f += isNaN((betanll + munll + Pnll + cnll + dnll),__LINE__);

     REPORT(logEye)
     REPORT(logD)
     REPORT(logbeta)
     REPORT(logmu)

     REPORT(sigma_logP);
     REPORT(sigma_logbeta);
     REPORT(sigma_logmu);
     REPORT(gamma);

     REPORT(f);
     REPORT(betanll);
     REPORT(munll);
     REPORT(Pnll);
     REPORT(cnll);
     REPORT(dnll);

     return isNaN(f,__LINE__);
}
