   //  simpleSIR3 -noinit -iprint 1 &> simpleSIR3.out&
   //  simpleSIR3 -noinit -mcmc2 500000 -mcsave 20 -shess &> simpleSIR3.out
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


  template <typename SCALAR> SCALAR logit(const SCALAR& p)
  {
     SCALAR a = log(p/(1.0-p));
     return a;
  }
  template double logit<double>(const double& p);
  template dvariable logit<dvariable>(const dvariable& p);
  //template dvariable logit<dvariable>(const double& p);

  template <typename SCALAR> 
  SCALAR alogit(const SCALAR& a)
  {
     SCALAR p = 1.0/(1.0+(mfexp(-a))+1e-20);
     return p;
  }
  template double alogit<double>(const double& a);
  template dvariable alogit<dvariable>(const dvariable& a);

  //  // zero-inflated log-normal error
  //  template <typename T1, typename T2, typename T3> 
  //  T2 ZILNerr(const T1& logobs, const T2& logpred, const T3& var, const double& prop0)
  //  {
  //      dvariable nll;
  //
  //      if (logobs > logeps)  //log zero deaths
  //      {
  //          nll = (1.0-prop0)*0.5*(log(TWO_M_PI*var) + square(logobs - logpred)/var);
  //      }
  //      else
  //      {
  //          nll = prop0*0.5*(log(TWO_M_PI*var));
  //      }
  //      nll = exp(nll);
  //      return nll;
  //  }
  //  dvariable ZINLerr(const dvariable& logobs, const dvariable& logpred, const dvariable& var, double& prop0);

    // log-normal error
    template <typename SCALAR>
    SCALAR NLerr(const SCALAR& logobs, const SCALAR& logpred, const SCALAR& var)
    {
        SCALAR nll = 0.5*(log(TWO_M_PI*var) + square(logobs-logpred)/var);
        return nll;
    }
    template dvariable NLerr(const dvariable& logobs, const dvariable& logpred, const dvariable& var);

  //template <typename SCALAR> 
  //SCALAR isNaN(const SCALAR& x, const int line)
  //{
  //    if (x != x)
  //    {
  //         std::cerr << "NaN at line " << line << std::endl;
  //         exit(1);
  //    }
  //    return x;
  //}
  //template dvariable isNaN(const dvariable& x, const int line);

TOP_OF_MAIN_SECTION
  arrmblsize = 50000000;
  gradient_structure::set_CMPDIF_BUFFER_SIZE(  150000000L);
  gradient_structure::set_GRADSTACK_BUFFER_SIZE(12550000L);
  gradient_structure::set_MAX_NVAR_OFFSET(3000000);

  adstring logname(adstring(argv[0])+"_program.log");
  Clogf.open(logname);
  if ( !Clogf ) {
    cerr << "Cannot open program log " << logname << endl;
    ad_exit(1);
  }
  cout << "Opened program log: " << logname << endl;
  Clogf << "Opened program log: " << logname << endl;
  //pad();

DATA_SECTION
  init_int phase_logsigma_logP;
  init_number init_sigma_logP;
  !!  TTRACE(phase_logsigma_logP,init_sigma_logP)

  init_int phase_logsigma_beta;
  init_number init_sigma_beta;
  !!  TTRACE(phase_logsigma_beta,init_sigma_beta)

  init_int phase_mu;
  init_number init_mu;
  !!  TTRACE(phase_mu,init_mu)

  init_int phase_gamma;
  init_number init_gamma;
  !!  TTRACE(phase_gamma,init_gamma)

  init_int phase_sigma_logC;
  init_number init_sigma_logC;
  !!  TTRACE(phase_sigma_logC,init_sigma_logC)

  init_int phase_sigma_logD;
  init_number init_sigma_logD;
  !!  TTRACE(phase_sigma_logD,init_sigma_logD)

  init_number init_beta;
  !!  TRACE(init_beta)
  init_number beta_a;
  init_number beta_b;
  !!  TTRACE(beta_a,beta_b)

  init_adstring dat_file_path;
  !!  TRACE(dat_file_path)

  !! ad_comm::change_datafile_name(dat_file_path);
  init_adstring county;
  //!! TRACE(county)
  init_adstring updated;
  //!! TRACE(updated) 
  init_int N0;
  //!! TRACE(N0) 
  init_adstring Date0;
  //!! TRACE(Date0) 
  init_int ntime;
  !! TRACE(ntime) 
  init_matrix obs(0,ntime,1,2);
  //!!  TRACE(obs)
  //!!  TRACE(trans(obs))
  matrix tobs;
  !! tobs = trans(obs);
  //!! TRACE(tobs)    
  vector y;
  vector obs_cases;
  !!  obs_cases = tobs(1);
  //!!  TRACE(obs_cases)
  vector obs_deaths;
  !!  obs_deaths = tobs(2);
  //!!  TRACE(obs_deaths)
  //!!  if(1)  ad_exit(1);

  vector log_obs_cases(0,ntime);
  vector log_obs_deaths(0,ntime);
  number prop_zero_deaths;

PARAMETER_SECTION
  init_number logsigma_logP(phase_logsigma_logP); // SIR process error
  init_number logsigma_beta(phase_logsigma_beta); // beta random walk sd
  init_number logmu(phase_mu);                    // mortality rate
  init_number loggamma(phase_gamma);              // recovery rate of infection population
  init_number logsigma_logC(phase_sigma_logC);    // cases observation error
  init_number logsigma_logD(phase_sigma_logD);    // deaths observation error

  random_effects_vector logitbeta(0,ntime);       // infection rate time series
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

  //number betanll;
  //number Pnll;
  //number cnll;
  //number dnll;
    
  objective_function_value f;


PRELIMINARY_CALCS_SECTION
  logsigma_logP = log(init_sigma_logP);
  logsigma_beta = log(init_sigma_beta);
  logmu = log(init_mu);
  loggamma = log(init_gamma);
  TTRACE(loggamma,logmu)
  logsigma_logC = log(init_sigma_logC);
  logsigma_logD = log(init_sigma_logD);
  TTRACE(sigma_beta,sigma_logP)
  TTRACE(sigma_logC,sigma_logD)

  double init_logit_beta = beta_a + (beta_b-beta_a)*logit(double(init_beta));
  int zero_count = 0;
  for (int t = 0; t <=  ntime; t++)
  {
       logitbeta(t) = init_logit_beta;
     //TTRACE(t,logitbeta(t))
       log_obs_cases(t) = log(obs_cases(t)+eps);
     //TTRACE(t,obs_cases(t))
       if (obs_deaths(t) < 1.0)
            zero_count ++;
       log_obs_deaths(t) = log(obs_deaths(t)+eps);
     //TTRACE(t,obs_deaths(t))
  }
  prop_zero_deaths = double(zero_count)/double(ntime);
  TRACE(prop_zero_deaths)
  //  if(1)  ad_exit(1);
 
RUNTIME_SECTION
  // derivatives get inaccurate at gradients < 1e-4
  // with current data
  convergence_criteria .001

PROCEDURE_SECTION

  //f = 0.0;

  TRACE(prop_zero_deaths)
  TTRACE(loggamma,logmu)
  gamma = mfexp(loggamma);
  mu = mfexp(logmu);
  TTRACE(gamma,mu)

  sigma_beta = mfexp(logsigma_beta);
  sigma_logP = mfexp(logsigma_logP);
  sigma_logC = mfexp(logsigma_logC);
  sigma_logD = mfexp(logsigma_logD);
  TTRACE(sigma_beta,sigma_logP)
  TTRACE(sigma_logC,sigma_logD)

  var_beta = square(sigma_beta);
  var_logP = square(sigma_logP);
  var_logC = square(sigma_logC);
  var_logD = square(sigma_logD);

  for (int t = 0; t <=  ntime; t++)
  {
     beta(t) = beta_a + (beta_b - beta_a)*invlogit(logitbeta(t));
  }


  //  loop over time
  logEye(0) = log_obs_cases(0);
  logD(0) = log_obs_deaths(0);
  for (int t = 1; t <=  ntime; t++)
  {
       TRACE(t)
       step(t, beta(t-1), beta(t), var_beta, logEye(t-1), logEye(t), logD(t-1), logD(t), var_logP, gamma, mu);
       TRACE(t)
  }
  //TRACE(f)
  
       // compute observation likelihoods
  for (int t = 0; t <= ntime; t++)
  {   
       obs(t, log_obs_cases(t), logEye(t), var_logC, log_obs_deaths(t), logD(t),var_logD); 
  }
  TRACE(f)
  TTRACE(sigma_beta,sigma_logP)
  TTRACE(sigma_logC,sigma_logD)




SEPARABLE_FUNCTION void step(const int t, const dvariable& pbeta, const dvariable& betat, const dvariable& varb, const dvariable& plogEye, const dvariable& logEyet, const dvariable& plogD, const dvariable& logDt, const dvariable& varP, const dvariable& gamma, const dvariable& mu)
           TRACE(t)
           dvariable betanll = 0.0;
           // infection rate random walk
        // betanll += isNaN(NLerr(beta(t-1),beta(t),var_beta),__LINE__);
        // betanll += 0.5*(log(TWO_M_PI*varb) + square(pbeta-betat)/varb);
           betanll += NLerr(pbeta, betat, varb);
           // cases process error
        // Type prevEye = exp(logEye(t-1));
           dvariable prevEye = mfexp(plogEye);
        // logEye(t) = log(prevEye*(1.0 + (beta(t-1) - gamma - mu))+eps);
           dvariable nextlogEye = log(prevEye*(1.0 + (pbeta - gamma - mu))+eps);

           dvariable Pnll = 0.0;
        // Pnll += isNaN(NLerr(logEye(t-1), logEye(t),var_logP),__LINE__);
           Pnll += 0.5*(log(TWO_M_PI*varP) + square(plogEye-nextlogEye)/varb);
           TRACE(t)

           // deaths process error
           dvariable prevd = mfexp(plogD);
        // logd(t) = log(prevd + mu*exp(logeye(t-1))+eps);
           dvariable nextlogD = log(prevd + mu*mfexp(plogEye)+eps);
           //Pnll += zilnerr(plogD, logdt, varp, prop_zero_deaths);
           if (value(plogD) > logeps)  //log zero deaths
           {
              Pnll += (1.0-prop_zero_deaths)*0.5*(log(TWO_M_PI*varP) + square(plogD- nextlogD)/varP);
           }
           else
           {
              Pnll += prop_zero_deaths*0.5*(log(TWO_M_PI*varP));
           }

           f += (betanll + Pnll);
 
SEPARABLE_FUNCTION void obs(const int t, const double& lobsC, const dvariable& lEye, const dvariable& varC, double& lobsD, const dvariable& lpredD, const dvariable& varD)
           dvariable cnll = 0.0;
           dvariable dnll = 0.0;
           cnll+= 0.5*(log(TWO_M_PI*varC) + square(lobsC-lEye)/varC);

           //dvariable dnll = ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths);
           if (lobsD > logeps)  //log zero deaths
           {
              dnll += (1.0-prop_zero_deaths)*0.5*(log(TWO_M_PI*varD) + square(lobsD - lpredD)/varD);
           }
           else
           {
              dnll += prop_zero_deaths*0.5*(log(TWO_M_PI*varD));
           }

           f += (cnll + dnll);

REPORT_SECTION
  for (int t = 0; t <= ntime; t++) 
  {
      report << obs_cases(t) << ","
             << obs_deaths (t) << ","
             << log_obs_cases(t) << ","
             << log_obs_deaths (t) << ","
             << logEye(t) << ","
             << logD(t) << ","
             << beta(t) << endl; // ","
             //<< mu << endl;
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
