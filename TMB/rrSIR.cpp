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
Type LNerr(Type logobs, Type logpred, Type var)
{
    Type nll = 0.5*(log(TWO_M_PI*var) + square(logobs-logpred)/var);
    // identical:
    //Type tnll = -dnorm(logobs,logpred,sqrt(var),true);
    //TTRACE(nll,tnll)
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
         TRACE(x)
         std::cerr << "NaN at line " << line << std::endl;
       //exit(1);
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
//  DATA_VECTOR(log_obs_CFR)
                  
    PARAMETER(logsigma_logCP);          // SIR process error
//  PARAMETER(logsigma_logZP);          // SIR process error
    PARAMETER(logsigma_logDP);          // SIR process error

    PARAMETER(logsigma_logbeta);       // beta random walk sd
    PARAMETER(logsigma_logZ);      // Z randomwalk sd
    PARAMETER(logsigma_logmu);         // mu randomwalk sd

    PARAMETER(logsigma_logC);          // cases observation error
    PARAMETER(logsigma_logD);          // deaths observation error
 
    PARAMETER_VECTOR(logbeta);         // infection rate time series
    PARAMETER_VECTOR(logZ);            // infected Infections 'attrition' rate
    PARAMETER_VECTOR(logmu);           // mortality rate of infection population

    // state variables
    vector <Type> logS(ntime+1);      // number of Suscectibles
    vector <Type> logEye(ntime+1);    // number of Infections
    vector <Type> logR(ntime+1);      // number of Recovered
    vector <Type> logD(ntime+1);      // number of Deaths from infected populationtime+1);
    vector <Type> loggamma(ntime+1);  // recovery rate of infection population
    vector <Type> brn(ntime+1);       // basic reproduction ratio

    Type sigma_logbeta = exp(logsigma_logbeta); 
    Type sigma_logZ = exp(logsigma_logZ); 
    Type sigma_logmu = exp(logsigma_logmu); 

    Type sigma_logCP = exp(logsigma_logCP);
//  Type sigma_logZP = exp(logsigma_logZP);
    Type sigma_logDP = exp(logsigma_logDP);

    Type sigma_logC = exp(logsigma_logC);
    Type sigma_logD = exp(logsigma_logD);

    Type var_logbeta = square(sigma_logbeta);
    Type var_logZ = square(sigma_logZ);
    Type var_logmu = square(sigma_logmu);
    Type var_logCP = square(sigma_logCP);
//  Type var_logZP = square(sigma_logZP);
    Type var_logDP = square(sigma_logDP);

    Type var_logC = square(sigma_logC);
    Type var_logD = square(sigma_logD);

    Type f = 0.0;
    Type betanll = 0.0;
    Type Znll = 0.0;
    Type munll = 0.0;
    Type Pnll = 0.0;
    Type cnll = 0.0;
    Type dnll = 0.0;
//  Type CFRnll = 0.0;

    //  loop over time
    logS[0] = log(N0);
    logEye[0] = log_obs_cases[0];
    logR[0] = 0.0;
    logD[0] = log_obs_deaths[0];
    for (int t = 1; t <= ntime; t++)
    {
       //TRACE(t)

         // infection rate random walk
       //`TRACE(t,var_logbeta)
       //TTRACE(logbeta(t-1),logbeta(t))
         betanll += isNaN(LNerr(logbeta(t-1),logbeta(t),var_logbeta),__LINE__);

         // recovery rate random walk
         Znll += isNaN(LNerr(logZ(t-1),logZ(t),var_logZ),__LINE__);
 
         // mortality rate random walk
         munll += isNaN(LNerr(logmu(t-1),logmu(t),var_logmu),__LINE__);

         // compute process error likelihood
         Type beta = exp(logbeta(t-1));
         Type mu   = exp(logmu(t-1));
         Type gamma = exp(logZ(t-1)) - mu; 
         loggamma(t-1) = log(gamma);

        
//       log_pred_CFR(t-1) = log(cfr);
         

         Type S   = exp(logS(t-1));
         Type Eye = exp(logEye(t-1));
         Type R   = exp(logR(t-1));
         Type D   = exp(logD(t-1));
         Type Z   = exp(logZ(t-1));
         Type N   = S + Eye + R;
         Type bison = beta * Eye * S/N;
       //TTRACE(t,N)
       //TTRACE(Eye,D)
       //TTRACE(S,R)
       //TRACE(bison)
         
         Type deltaS = -bison;
         Type nextS = S + deltaS;
         logS(t) = log(nextS);
         // susceptible process error
         /*
         Type deltaS = -bison
         Type nextS = S + deltaS;
         if (nextS > 0.0)
         {
             logS(t) = log(nextS);
             Pnll += isNaN(LNerr(logS(t-1), logS(t), var_logP),__LINE__);
         }
         else
         {
             Pnll += square(deltaS);
             TTRACE(nextS,deltaS)
         }
         */

         // cases process error
         Type deltaEye = bison - Z*Eye;
         Type nextEye = Eye + deltaEye;
         if (nextEye > 0.0)
         {
             logEye(t) = log(nextEye);
             Pnll += isNaN(LNerr(logEye(t-1), logEye(t),var_logCP),__LINE__);
         }
         else
         {
             Pnll += square(deltaEye);
             TTRACE(nextEye,deltaEye)
         }

         Type deltaR = gamma*Eye;
         Type nextR = R + deltaR;
         if (nextR > 0.0)
             logR(t) = log(nextR);
         else
         {
             Znll += square(deltaR);
             TTRACE(nextR,deltaR)
         }

         // deaths process error
         Type deltaD = mu*Eye;
         Type nextD = D + deltaD;
         if (nextD > 0.0)
         {
             logD(t) = log(nextD);
             Pnll += isNaN(LNerr(logD(t-1), logD(t),var_logDP),__LINE__);
         }
         else
         {
             Pnll += square(deltaD);
             TTRACE(nextD,deltaD)
         }

     }
     loggamma(ntime) = loggamma(ntime-1);


     // compute observation likelihoods
     for (int t = 0; t <= ntime; t++)
     {   
       //TRACE(t)
         cnll += isNaN(  LNerr(log_obs_cases(t),logEye(t),var_logC),__LINE__);
       //TTRACE(cnll,logEye(t))


     //  Zero inflated log normal
         dnll += isNaN(ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths),__LINE__);
     //  dnll += isNaN(  LNerr(log_obs_deaths(t),logD(t),var_logD),__LINE__);

     //  Poisson error
     //  dnll += -isNaN(obs_deaths(t)*logD(t) - exp(logD(t)) - lfactorial(obs_deaths(t)),__LINE__);

       //TTRACE(dnll,logD(t))

     //  TTRACE(cnll,dnll)
     }
//   TTRACE(ntime,CFRnll)
//   TTRACE(cnll,dnll)


     // total likelihood
     f += isNaN((betanll + Znll + munll + Pnll + cnll + dnll),__LINE__);

     brn = exp(logbeta-loggamma);

     REPORT(logS)
     REPORT(logEye)
     REPORT(logR)
     REPORT(logD)

     REPORT(logbeta)
     REPORT(logZ)
     REPORT(logmu)
     REPORT(loggamma)
     REPORT(brn)

     REPORT(sigma_logCP);
//   REPORT(sigma_logZP);
     REPORT(sigma_logDP);

     REPORT(sigma_logbeta);
     REPORT(sigma_logmu);

     REPORT(f);
     REPORT(betanll);
     REPORT(Znll);
     REPORT(munll);
     REPORT(Pnll);
     REPORT(cnll);
     REPORT(dnll);
/*
     if (1)
     {
     //   TRACE(logbeta)
     //   TRACE(loggamma)
     //   TRACE(logmu)
          TRACE(betanll);
          TRACE(gammanll);
          TTRACE(cnll,dnll)
          TTRACE(f,CFRnll)
     //   return(Type(-1.0));
     }
*/
   //TRACE(f)
     return isNaN(f,__LINE__);
}
