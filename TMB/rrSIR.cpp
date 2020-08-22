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
    DATA_VECTOR(obs_cases)
    DATA_VECTOR(obs_deaths)
    DATA_SCALAR(prop_zero_deaths)
    DATA_VECTOR(log_obs_cases)
    DATA_VECTOR(log_obs_deaths)


    PARAMETER(logsigma_logP);          // SIR process error

    PARAMETER(logsigma_logbeta);       // beta random walk sd
    PARAMETER(logsigma_loggamma);      // gamma randomwalk sd
    PARAMETER(logsigma_logmu);         // mu randomwalk sd

    PARAMETER(logbias_logbeta);        // beta random walk bias
    PARAMETER(logbias_loggamma);       // gamma randomwalk bias
    PARAMETER(logbias_logmu);          // mu randomwalk bias

    PARAMETER(logprop_immune);         // proprtion of infected recovered who are immune

    PARAMETER(logsigma_logC);          // cases observation error
    PARAMETER(logsigma_logD);          // deaths observation error

    PARAMETER_VECTOR(logbeta);         // infection rate time series
    PARAMETER_VECTOR(loggamma);        // recovery rate of infection population
    PARAMETER_VECTOR(logmu);           // mortality rate of infection population

    // state variables
    vector <Type> logS(ntime+1);      // number of Suscectibles
    vector <Type> logEye(ntime+1);    // number of Infections
    vector <Type> logR(ntime+1);      // number of Recovered
    vector <Type> logD(ntime+1);      // number of Deaths from infected population

    Type sigma_logbeta = exp(logsigma_logbeta); 
    Type sigma_logmu = exp(logsigma_logmu); 
    Type sigma_loggamma = exp(logsigma_loggamma); 

    Type bias_logbeta = exp(logbias_logbeta); 
    Type bias_logmu = exp(logbias_logmu); 
//  TTRACE(bias_logbeta,bias_logmu)
    Type bias_loggamma = exp(logbias_loggamma); 
    Type prop_immune = exp(prop_immune);
//  TTRACE(bias_loggamma,prop_immune)

    Type sigma_logP = exp(logsigma_logP);
    Type sigma_logC = exp(logsigma_logC);
    Type sigma_logD = exp(logsigma_logD);

    Type var_logbeta = square(sigma_logbeta);
    Type var_logmu = square(sigma_logmu);
    Type var_loggamma = square(sigma_loggamma);
    Type var_logP = square(sigma_logP);

    Type var_logC = square(sigma_logC);
    Type var_logD = square(sigma_logD);

    Type f = 0.0;
    Type betanll = 0.0;
    Type munll = 0.0;
    Type gammanll = 0.0;
    Type Pnll = 0.0;
    Type cnll = 0.0;
    Type dnll = 0.0;

    //  loop over time
    logS[0] = log(N0);
    logEye[0] = eps;
    logD[0] = eps;
    logR[0] = eps;
    for (int t = 1; t <= ntime; t++)
    {
         // infection rate random walk
         betanll += isNaN(NLerr(logbeta(t-1)+bias_logbeta,logbeta(t),var_logbeta),__LINE__);

         // mortality rate random walk
         munll += isNaN(NLerr(logmu(t-1)+bias_logmu,logmu(t),var_logmu),__LINE__);

         // mortality rate random walk
         gammanll += isNaN(NLerr(loggamma(t-1)+bias_loggamma,loggamma(t),var_loggamma),__LINE__);

         // compute process error likelihood
         Type beta = exp(logbeta(t-1));
         Type gamma = exp(loggamma(t-1));
         Type mu = exp(logmu(t-1));
         Type S   = exp(logS(t-1));
         Type Eye = exp(logEye(t-1));
         Type R   = exp(logR(t-1));
         Type N   = S + Eye + R;
         Type D   = exp(logD(t-1));
         Type bison = beta * Eye * S/N;

         // susceptible process error
         Type deltaS = -bison + (1.0-prop_immune)*gamma*Eye;
         logS(t) = log(S + deltaS);
         Pnll += isNaN(NLerr(logS(t-1), logS(t), var_logP),__LINE__);

         // cases process error
         Type deltaEye = bison - mu*Eye - gamma*Eye;
         logEye(t) = log(Eye + deltaEye);
         Pnll += isNaN(NLerr(logEye(t-1), logEye(t),var_logP),__LINE__);

         // recovered process error
         Type deltaR = prop_immune*gamma*Eye;
         logR(t) = log(R+deltaR);
         Pnll += isNaN(NLerr(logR(t-1), logR(t),var_logP),__LINE__);

         // deaths process error
         Type deltaD = mu*Eye;
//       TTRACE(D,deltaD)
         logD(t) = log(D + deltaD);
         Pnll += isNaN(NLerr(logD(t-1), logD(t), var_logP),__LINE__);

     }
 
     // compute observation likelihoods
     for (int t = 0; t <= ntime; t++)
     {   
         cnll += isNaN(  NLerr(log_obs_cases(t),logEye(t),var_logC),__LINE__);

     //  dnll += isNaN(ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths),__LINE__);
     //  temp = -(tobs*log(tpred) - tpred - FACT.log_factorial((int)tobs));
         dnll += -isNaN(obs_deaths(t)*logD(t) - exp(logD(t)) - lfactorial(obs_deaths(t)),__LINE__);
     }

     // total likelihood
     f += isNaN((betanll + munll + gammanll + Pnll + cnll + dnll),__LINE__);

     REPORT(logS)
     REPORT(logEye)
     REPORT(logR)
     REPORT(logD)
     REPORT(logbeta)
     REPORT(loggamma)
     REPORT(logmu)

     REPORT(sigma_logP);
     REPORT(sigma_logbeta);
     REPORT(sigma_loggamma);
     REPORT(sigma_logmu);
     REPORT(bias_logbeta);
     REPORT(bias_loggamma);
     REPORT(bias_logmu);

     REPORT(f);
     REPORT(betanll);
     REPORT(gammanll);
     REPORT(munll);
     REPORT(Pnll);
     REPORT(cnll);
     REPORT(dnll);

     return isNaN(f,__LINE__);
}
