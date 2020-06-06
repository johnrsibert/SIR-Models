GLOBALS_SECTION;
  #include <math.h>
  #include <adstring.hpp>
  #include "trace.h"
  #include <admodel.h>
  #include <fvar.hpp>

  #undef PINOUT
  #define PINOUT(object) pin << "# " << #object ":\n" << setprecision(5) << object << endl;
  #undef REPORT
  #define REPORT(object) report << "# " << #object ":\n" << setprecision(5) << object << endl;

  ofstream Clogf;
  const double TWO_M_PI = 2.0*M_PI;
  //const double LOG_M_PI = log(M_PI);
  const double eps = 1e-8;
  const double logeps = log(eps);

   //  simpleSIR3 -noinit -iprint 1 -shess &> issams6.out&
   //  simpleSIR3 -noinit -mcmc2 500000 -mcsave 20 -shess &> issams6.out

  template <typename SCALAR> SCALAR logit(const SCALAR& p)
  {
     SCALAR a = log(p/(1.0-p));
     return a;
  }
  template double logit<double>(const double& p);
  template dvariable logit<dvariable>(const dvariable& p);

  template <typename SCALAR> 
  SCALAR alogit(const SCALAR& a)
  {
     SCALAR p = 1.0/(1.0+(mfexp(-a))+1e-20);
     return p;
  }
  template double alogit<double>(const double& a);
  template dvariable alogit<dvariable>(const dvariable& a);


  template <typename SCALAR> 
  SCALAR NLerr(const SCALAR& logobs, const SCALAR& logpred, const SCALAR& var)
  {
      SCALAR nll = 0.5*(log(TWO_M_PI*var) + square(logobs-logpred)/var);
      return nll;
  }
  template dvariable NLerr(const dvariable& logobs, const dvariable& logpred, 
                                      const dvariable& var);

  // zero-inflated log-normal error
  template <class SCALAR>
  SCALAR ZILNerr(SCALAR logobs, SCALAR logpred, SCALAR var, SCALAR prop0 = 0.15)
  {
      SCALAR nll;

      if (logobs > logeps)  //log zero deaths
      {
          nll = (1.0-prop0)*0.5*(log(TWO_M_PI*var) + square(logobs - logpred)/var);
      }
      else
      {
          nll = prop0*0.5*(log(TWO_M_PI*var));
      }
      nll = exp(nll);
      return nll;
  }

  template < class SCALAR > SCALAR isNaN(SCALAR x, const int line)
  {
      if (x != x)
      {
           std::cerr << "NaN at line " << line << std::endl;
           exit(1);
      }
      return x;
  }


DATA_SECTION
  init_adstring county;
  init_adstring updated;
  init_int N0;
  init_adstring Date0;
  init_int ntime;
  init_matrix obs(1,2,0,ntime);
  number beta_a;
  number beta_b;
  !!  TRACE(obs) 
      
  //  DATA_SCALAR(N0)
  //  DATA_INTEGER(ntime)
  //  DATA_VECTOR(log_obs_cases)
  //  DATA_VECTOR(log_obs_deaths)
  //  DATA_SCALAR(prop_zero_deaths)
  //  DATA_SCALAR(beta_a)
  //  DATA_SCALAR(beta_b)
  //  DATA_SCALAR(mu_a)
  //  DATA_SCALAR(mu_b)
     

PARAMETER_SECTION
  init_number logsigma_logP;          // SIR process error
  init_number logsigma_beta;       // beta random walk sd
  init_number logmu;         // mu randomwalk sd
  init_number loggamma;            // recovery rate of infection population
  init_number logsigma_logC;          // cases observation error
  init_number logsigma_logD;          // deaths observation error

  random_effects_vector logitbeta(0,ntime);      // infection rate time series
  vector beta(0,ntime);

  // state variables
  vector logEye(0,ntime);    // number of infections
  vector logD(0,ntime);      // number of deaths from infected population

  number mu;
  number gamma;

  number sigma_beta;
  number sigma_logP;
  number sigma_logC;
  number sigma_logD;

  number var_beta;
  number var_logP;
  number var_logC;
  number var_logD;

  number betanll;
  number Pnll;
  number cnll;
  number dnll;
    
  objective_function_value f;


PRELIMINARY_CALCS_SECTION
  beta_a = eps;
  beta_b = 2.0;
  for (int t = 0; t <  ntime; t++)
  {
       beta(t) = beta_a + (beta_b - beta_a)*invlogit(logitbeta(t));
  }
 
  

PROCEDURE_SECTION
  //  loop over time
  //logEye(0) = log_obs_cases(0);
  //logD(0) = log_obs_deaths(0);
  for (int t = 1; t <  ntime; t++)
  {
       step(t, beta(t-1), beta(t), var_beta, var_logP);
  }

  
       // compute observation likelihoods
  for (int t = 0; t < ntime; t++)
  {   
       obs(t); 
  }



SEPARABLE_FUNCTION void step(const int t, const dvariable& pbeta, const dvariable& betat, const dvariable& varb, const dvariable& varP)
           // infection rate random walk
        // betanll += isNaN(NLerr(beta(t-1),beta(t),var_beta),__LINE__);
           betanll += isNaN(NLerr(pbeta,betat,varb),__LINE__);

           // cases process error
        // Type prevEye = exp(logEye(t-1));
           dvariable prevEye = exp(logEye(t-1));
        // logEye(t) = log(prevEye*(1.0 + (beta(t-1) - gamma - mu))+eps);
           logEye(t) = log(prevEye*(1.0 + (pbeta - gamma - mu))+eps);
        // Pnll += isNaN(NLerr(logEye(t-1), logEye(t),var_logP),__LINE__);
           Pnll += isNaN(NLerr(logEye(t-1), logEye(t),varP),__LINE__);

           // deaths process error
           Type prevD = exp(logD(t-1));
           logD(t) = log(prevD + mu*exp(logEye(t-1))+eps);
           Pnll += isNaN(ZILNerr(logD(t-1), logD(t), varP),__LINE__);

           f += isNaN((betanll + Pnll),__LINE__);
 
SEPARABLE_FUNCTION void obs(const int t)
           cnll += isNaN(  NLerr(log_obs_cases(t),logEye(t),var_logC),__LINE__);

           dnll += isNaN(ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths),__LINE__);

           f += isNaN((cnll + dnll),__LINE__);

REPORT_SECTION
  for (int t = 0; t <= ntime; t++) 
  {
      report << obs_cases(t) << ","
             << obs_deaths (t) << ","
             << log_obs_cases(t) << ","
             << log_obs_deaths (t) << ","
             << log_pred_cases(t) << ","
             << log_pred_deaths(t) << ","
             << beta(t) << ","
             << mu << endl;
  }
       /*
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
       */
