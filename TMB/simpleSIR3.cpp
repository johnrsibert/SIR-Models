#include <TMB.hpp>
//#include "convenience.hpp"
#include <math.h>
#include <fenv.h> 
const double TWO_M_PI = 2.0*M_PI;
//const double LOG_M_PI = log(M_PI);
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

template <class Type>
Type NLerr(Type logobs, Type logpred, Type var)
{
    /*
    Type resid = (logobs - logpred) / sd;
    Type nll = -log(sqrt(TWO_M_PI));
    bool ztest = ((resid+0.0) == 0.0); // 1 if resid = -0
    if (!ztest)
    {
        Type lsd = log(sd);
        nll -= log(sd) + Type(0.5) * resid * resid;
    }
    //Type tmp = dnorm(logobs,logpred,sd);
    */
    
    Type nll = 0.5*(log(TWO_M_PI*var) + square(logobs-logpred)/var);
    return nll;
}

// zero-inflated log-normal error
template <class Type>
Type ZILNerr(Type logobs, Type logpred, Type var, Type prop0)
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
    //nll = exp(nll);
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

/*
// from the TMB dox
template<class Type>
Type dnorm(Type x, Type mean, Type sd, int give_log=0)
{
   Type resid = (x - mean) / sd;
   Type logans = -log(sqrt(2*M_PI)) - log(sd) - Type(.5) * resid * resid;
   if(give_log) return logans; else return exp(logans);
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
    DATA_SCALAR(beta_a)
    DATA_SCALAR(beta_b)
    DATA_SCALAR(mu_a)
    DATA_SCALAR(mu_b)


    PARAMETER(logsigma_logP);          // SIR process error
    PARAMETER(logsigma_beta);       // beta random walk sd
    PARAMETER(logmu);         // mu randomwalk sd
    PARAMETER(loggamma);            // recovery rate of infection population
    PARAMETER(logsigma_logC);          // cases observation error
    PARAMETER(logsigma_logD);          // deaths observation error

    PARAMETER_VECTOR(logitbeta);      // infection rate time series

    vector <Type> beta(ntime);
    for (int t = 0; t <  ntime; t++)
    {
     // Type u = a + (b - a)*invlogit(stlogit_u);
        beta[t] = beta_a + (beta_b - beta_a)*invlogit(logitbeta[t]);
    }

    // state variables
    vector <Type> logEye(ntime);    // number of infections
    vector <Type> logD(ntime);      // number of deaths from infected population

    Type mu = exp(logmu);
    Type gamma = exp(loggamma);

    Type sigma_beta = exp(logsigma_beta); 
    Type sigma_logP = exp(logsigma_logP);
    Type sigma_logC = exp(logsigma_logC);
    Type sigma_logD = exp(logsigma_logD);

    Type var_beta = square(sigma_beta);
    Type var_logP = square(sigma_logP);
    Type var_logC = square(sigma_logC);
    Type var_logD = square(sigma_logD);

    Type f = 0.0;
    Type betanll = 0.0;
    Type Pnll = 0.0;
    Type cnll = 0.0;
    Type dnll = 0.0;

    //  loop over time
    //logEye(0) = log_obs_cases(0);
    //logD(0) = log_obs_deaths(0);
    for (int t = 1; t <  ntime; t++)
    {
         // infection rate random walk
         betanll += isNaN(NLerr(beta(t-1),beta(t),var_beta),__LINE__);

         // cases process error
         Type prevEye = exp(logEye(t-1));
         logEye(t) = log(prevEye*(1.0 + (beta(t-1) - gamma - mu))+eps);
         Pnll += isNaN(NLerr(logEye(t-1), logEye(t),var_logP),__LINE__);

         // deaths process error
         Type prevD = exp(logD(t-1));
         logD(t) = log(prevD + mu*exp(logEye(t-1))+eps);
         Pnll += isNaN(ZILNerr(logD(t-1), logD(t), var_logP, prop_zero_deaths),__LINE__);

     }
 
     // compute observation likelihoods
     for (int t = 0; t < ntime; t++)
     {   
         cnll += isNaN(  NLerr(log_obs_cases(t),logEye(t),var_logC),__LINE__);

         dnll += isNaN(ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths),__LINE__);
     }

     f += isNaN((betanll + Pnll + cnll + dnll),__LINE__);

     REPORT(logEye)
     REPORT(logD)
     REPORT(beta)
     REPORT(mu)

     REPORT(sigma_logP);
     REPORT(sigma_beta);
     REPORT(loggamma);
     REPORT(gamma);

     REPORT(f);
     REPORT(betanll);
     REPORT(Pnll);
     REPORT(cnll);
     REPORT(dnll);

     return isNaN(f,__LINE__);
}
