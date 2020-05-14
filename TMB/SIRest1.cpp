#include <TMB.hpp>
#include <math.h>
#include "trace.h"
const double TWO_M_PI = 2.0*M_PI;
const double LOG_M_PI = log(M_PI);
//const double logZeroCatch = 0.0;  // log(1.0)
const double eps = 1e-8;

template < class Type > Type square(Type x)
{
    return x * x;
}

// zero-inflated log-normal error
template <class Type>
Type ZILNerr(Type obs, Type pred, Type var, const double prop0 = 0.15)
{
    Type tmp;

    if (obs > 0.0)
    {
        tmp = (1.0-prop0)*0.5*(log(TWO_M_PI*var) + square(log(obs) - log(pred+eps))/var);
    }
    else
    {
        tmp = prop0*0.5*(log(TWO_M_PI*var));
    }


    return tmp;
}

// log-normal error
template <class Type>
Type NLerr(Type obs, Type pred, Type var)
{
    Type nll = 0.5*(log(TWO_M_PI*var) + square(log(obs)-log(pred))/var);
    return nll;
}


template<class Type> 
Type objective_function <Type>::operator()()
{
    DATA_SCALAR(N0)
    DATA_INTEGER(ntime)
    DATA_VECTOR(obs_cases)
    DATA_VECTOR(obs_deaths)
    DATA_SCALAR(prop_zero_deaths)

    PARAMETER(logsigma_P);          // SIR process error
    PARAMETER(logsigma_beta);       // beta random walk sd
    PARAMETER(logmu);               // mortality rate of infection population
    PARAMETER(loggamma);            // recovery rate of infection population
    PARAMETER(logsigma_C);          // cases observation error
    PARAMETER(logsigma_D);          // deaths observation error
    PARAMETER_VECTOR(logbeta);      // infection rate time series

    // state variables
    vector <Type> N(ntime);         // total population
    vector <Type> Eye(ntime);       // number of infections
    vector <Type> S(ntime);         // susceptible population
    vector <Type> R(ntime);         // number of recovered
    vector <Type> D(ntime);         // number of deaths from infected population

    Type mu = exp(logmu);
    Type gamma = exp(loggamma);
    Type sigma_P = exp(logsigma_P);
    Type sigma_C = exp(logsigma_C);
    Type sigma_D = exp(logsigma_D);
    Type sigma_beta = exp(logsigma_beta);

    Type varlogbeta = square(log(sigma_beta));
    Type varlogPop = square(log(sigma_P));
    Type varlogcases = square(log(sigma_C));
    Type varlogdeaths = square(log(sigma_D));

    //  initialize variables and parameters
    N(0) = N0;
    Eye(0) = 1.0;
    S(0) = N(0) - 1.0;

    Type f = 0.0;

    //  loop over time
    for (int t = 1; t <  ntime; t++)
    {
         // infection rate random walk

         Type betanll = 0.0;
         betanll += 0.5*(log(TWO_M_PI*varlogbeta) + square(logbeta(t-1) -logbeta(t))/varlogbeta);
     
         Type Pnll = 0.0;
         N(t) = S(t-1) + Eye(t-1) + R(t-1);

         Type bison = exp(logbeta(t-1))*Eye(t-1)/(N(t-1)+eps);
         Type gEye = gamma*Eye(t-1);
         Type mEye = mu*Eye(t-1);

         Eye(t) = Eye(t-1) + ((exp(logbeta(t-1))*Eye(t-1)*S(t-1)/N(t-1)) - mu*Eye(t-1));
         Pnll += NLerr(Eye(t-1)+eps, Eye(t)+eps,varlogPop);

         S(t) = S(t-1) + exp(logbeta(t-1))*Eye(t-1)*S(t-1)/N(t-1);
         Pnll += NLerr(S(t-1)+eps, S(t)+eps,varlogPop);

         R(t) = R(t-1) + gEye;
         Pnll += NLerr(R(t-1)+eps, R(t)+eps,varlogPop);

         D(t) = mEye;

         f += (betanll + Pnll);
     }
 
     // compute observation likelihoods
     Type cnll = 0.0;
     Type dnll = 0.0;
     for (int t = 0; t < ntime; t++)
     {   
         dnll += ZILNerr(obs_deaths(t),D(t),varlogdeaths);

         cnll += NLerr(obs_cases(t),Eye(t)+eps,varlogcases);
     }

     f += (cnll + dnll);

     REPORT(S)
     REPORT(Eye)
     REPORT(R)
     REPORT(D)
     REPORT(N)
     REPORT(logbeta)
     return f;
}
