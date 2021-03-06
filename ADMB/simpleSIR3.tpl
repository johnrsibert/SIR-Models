   //  simpleSIR3 -noinit -est -shess
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
  const double logone = log(1.0-eps);


  template <typename SCALAR> SCALAR rlogit(const SCALAR& p)
  {
     SCALAR a = log(p/(1.0-p));
     return a;
  }
  template double rlogit<double>(const double& p);
  template dvariable rlogit<dvariable>(const dvariable& p);
  //template dvariable logit<dvariable>(const double& p);

  template <typename SCALAR> 
  SCALAR arlogit(const SCALAR& a)
  {
     SCALAR p = 1.0/(1.0+mfexp(-a));
     return p;
  }
  template double arlogit<double>(const double& a);
  template dvariable arlogit<dvariable>(const dvariable& a);

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
  //template <typename SCALAR>
  //SCALAR NLerr(const SCALAR& logobs, const SCALAR& logpred, const SCALAR& var)
  //{
  //    SCALAR nll = 0.5*(log(TWO_M_PI*var) + square(logobs-logpred)/var);
  //    return nll;
  //}
  //template dvariable NLerr(const dvariable& logobs, const dvariable& logpred, const dvariable& var);

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

  init_int phase_logitbeta; 
  init_number init_beta;
  !!  TTRACE(phase_logitbeta,init_beta)
  init_number beta_a;
  init_number beta_b;
  !!  TTRACE(beta_a,beta_b)

  init_adstring dat_file_path;
  !!  TRACE(dat_file_path)

  !! ad_comm::change_datafile_name(dat_file_path);
  init_adstring county;
  !! TRACE(county)
  init_adstring updated;
  !! TRACE(updated) 
  init_int N0;
  !! TRACE(N0) 
  init_adstring Date0;
  !! TRACE(Date0) 
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
  !!  TRACE(obs_cases)
  vector obs_deaths;
  !!  obs_deaths = tobs(2);
  !!  TRACE(obs_deaths)
  //!!  if(1)  ad_exit(1);

  vector log_obs_cases(0,ntime);
  vector log_obs_deaths(0,ntime);
  number prop_zero_deaths;

PARAMETER_SECTION
  init_number logsigma_logP(phase_logsigma_logP); // SIR process error
  init_number logsigma_beta(phase_logsigma_beta); // beta random walk sd
  init_bounded_number logmu(logeps,logone,phase_mu);  // mortality rate
  init_bounded_number loggamma(logeps,logone,phase_gamma);              // recovery rate of infection population
  init_number logsigma_logC(phase_sigma_logC);    // cases observation error
  init_number logsigma_logD(phase_sigma_logD);    // deaths observation error

  // state variables
  random_effects_vector logEye(0,ntime,phase_logitbeta);    // number of infections
  random_effects_vector logD(0,ntime,phase_logitbeta);      // number of deaths from infected population

  random_effects_vector logitbeta(0,ntime,phase_logitbeta);       // infection rate time series
  //vector beta(0,ntime);

  objective_function_value nll;


PRELIMINARY_CALCS_SECTION
  
  logsigma_logP = log(init_sigma_logP);
  logsigma_beta = log(init_sigma_beta);
  logmu = log(init_mu);
  TTRACE(init_mu,logmu)
  loggamma = log(init_gamma);
  TTRACE(init_gamma,loggamma)
  logsigma_logC = log(init_sigma_logC);
  logsigma_logD = log(init_sigma_logD);

  double init_logit_beta = beta_a + (beta_b-beta_a)*rlogit(double(init_beta));
  TTRACE(beta_a,beta_b)
  TTRACE(init_beta,init_logit_beta)

  int zero_count = 0;
  for (int t = 0; t <=  ntime; t++)
  {
       logitbeta(t) = init_logit_beta;
       log_obs_cases(t) = log(obs_cases(t)+eps);
     //TTRACE(t,obs_cases(t))
       if (obs_deaths(t) < 1.0)
            zero_count ++;
       log_obs_deaths(t) = log(obs_deaths(t)+eps);
     //TTRACE(t,obs_deaths(t))
       logEye(t) = eps;
       logD(t) = eps;
  }
  TRACE(logitbeta)
  TTRACE(zero_count,ntime)
  prop_zero_deaths = double(zero_count)/double(ntime+1);
  TRACE(prop_zero_deaths)
  //  if(1)  ad_exit(1);
 
RUNTIME_SECTION
  convergence_criteria .001

PROCEDURE_SECTION

  //f = 0.0;

  //  loop over time
  //logEye(0) = log_obs_cases(0);
  //logD(0) = log_obs_deaths(0);
  for (int t = 1; t <=  ntime; t++)
  {
  //   step(t, logitbeta, logsigma_beta, logEye, logD, logsigma_logP, loggamma, logmu);
       step(t, logitbeta(t-1),logitbeta(t), logsigma_beta, logEye(t-1), logEye(t), logD(t-1), logD(t), logsigma_logP, loggamma, logmu);
  }
  
       // compute observation likelihoods
  for (int t = 0; t <= ntime; t++)
  {   
  //   obs(t, logEye, logsigma_logC, logD, logsigma_logD);
       obs(t, logEye(t), logsigma_logC, logD(t), logsigma_logD);
  }

          //step(t, logitbeta, logsigma_beta, logEye, logD, logsigma_logP, loggamma, logmu);
  //SEPARABLE_FUNCTION void step(const int t, const dvar_vector& logitbeta, const dvariable& logsigma_beta, const dvar_vector& logEye, const dvar_vector& logD, const dvariable& logsigma_logP, const dvariable& loggamma, const dvariable& logmu)
          //step(t, logitbeta(t-1),logitbeta(t), logsigma_beta, logEye(t-1), logEye(t), logD(t-1), logD(t), logsigma_logP, loggamma, logmu);
SEPARABLE_FUNCTION void step(const int t, const dvariable& lb1, const dvariable& lbt, const dvariable& logsigma_beta, const dvariable& logEye1, const dvariable& logEyet, const dvariable& logD1, const dvariable& logDt, const dvariable& logsigma_logP, const dvariable& loggamma, const dvariable& logmu)

           dvariable betanll = 0.0;
           // infection rate random walk
         //TTRACE(lb1,lbt)
           dvariable pbeta = beta_a + (beta_b - beta_a)*arlogit(lb1+1e-20);
         //TTRACE(lb1,pbeta)
           dvariable nbeta = beta_a + (beta_b - beta_a)*arlogit(lbt+1e-20);
         //TTRACE(lb1,nbeta)
           dvariable varb = square(mfexp(logsigma_beta));
           betanll += 0.5*(log(TWO_M_PI*varb) + square(pbeta-nbeta)/varb);

           dvariable Pnll = 0.0;
           // cases process error
         //dvariable pEye = mfexp(logEye1);
           dvariable gamma = mfexp(loggamma);
         //TTRACE(loggamma,gamma)
           dvariable mu = mfexp(logmu);
         //TTRACE(logmu,mu)
         //dvariable nlogEye = log(pEye*(1.0 + (pbeta - gamma - mu))+eps);
         //int nrep = 10;
         //double dt = 1.0/double(nrep);
         //dvariable tmp = 0.0;
         //for (int n = 1; n <= nrep; n++)
         //    tmp = tmp - log(1.0 - dt*nbeta) + log(1.0 - dt*(gamma+mu));
         //dvariable nlogEye = logEye1 + tmp;

           dvariable nlogEye = logEye1 - log(1.0 - nbeta) + log(1.0 - (gamma+mu));

           dvariable varP = square(mfexp(logsigma_logP));
           Pnll += 0.5*(log(TWO_M_PI*varP) + square(logEye1-nlogEye)/varP);
           if (isnan(value(Pnll)))
           {
                TRACE(t)
                TTRACE(Pnll,varP)
                TTRACE(logEye1,nlogEye)
                TTRACE(pbeta,nbeta)
                TTRACE(gamma,mu)
                ad_exit(1);
           }

           // deaths process error
           dvariable prevD = mfexp(logD1);
           dvariable nlogD = log(prevD + mu*mfexp(logEye1)+eps);
         //dvariable nlogD = log(prevD + mu*mfexp(nlogEye)+eps);

         //Pnll += 0.5*(log(TWO_M_PI*varP) + square(logD1 - nlogD)/varP);
         //if (log_obs_deaths(t) > logeps)
           if (obs_deaths(t) > 1.0)
           {
              Pnll += (1.0-prop_zero_deaths)*0.5*(log(TWO_M_PI*varP) + square(logD1 - nlogD)/varP);
           }
           else
           {
              Pnll += prop_zero_deaths*0.5*(log(TWO_M_PI*varP));
           }

         //TTRACE(betanll,Pnll);
           nll += (betanll + Pnll);
 
     //obs(t, logEye, logsigma_logC, logD, logsigma_logD);
     //SEPARABLE_FUNCTION void obs(const int t, const dvar_vector& logEye, const dvariable& logsigma_logC, const dvar_vector& logD, const dvariable& logsigma_logD)
     //obs(t, logEye(t), logsigma_logC, logD(t), logsigma_logD);
SEPARABLE_FUNCTION void obs(const int t, const dvariable& logEyet, const dvariable& logsigma_logC, const dvariable& logDt, const dvariable& logsigma_logD)
           dvariable cnll = 0.0;
           dvariable dnll = 0.0;
           dvariable varC = square(mfexp(logsigma_logC));
           cnll+= 0.5*(log(TWO_M_PI*varC) + square(log_obs_cases(t)-logEyet)/varC);

           //dvariable dnll = ZILNerr(log_obs_deaths(t),logD(t),var_logD, prop_zero_deaths);
           dvariable varD = square(mfexp(logsigma_logD));
         //if (log_obs_deaths(t) > logeps)  //log zero deaths
           if (obs_deaths(t) > 1.0)  //log zero deaths
           {
              dnll += (1.0-prop_zero_deaths)*0.5*(log(TWO_M_PI*varD) + square(log_obs_deaths(t) - logDt)/varD);
           }
           else
           {
              dnll += prop_zero_deaths*0.5*(log(TWO_M_PI*varD));
           }

         //TTRACE(cnll , dnll);
           nll += (cnll + dnll);

REPORT_SECTION
  int save_precision = report.precision();
  report.precision(15);
  report << "# meta:" << endl;
  report << "names,data" << endl;
  report << "county," << county << endl;
  report << "update_stamp," << updated << endl;
  report << "N0," << N0 << endl;
  report << "Date0," << Date0 << endl;
  report << "ntime," << ntime << endl;
  report << "prop_zero_deaths," << prop_zero_deaths << endl;
  report << "fn," << nll << endl;
  report << "convergence," << nll.gmax << endl;
  report << "model,"  << argv[0] << endl;

  report << "# diag:" <<endl;
  report << "obs_cases,obs_deaths,log_obs_cases,log_obs_deaths," <<
            "log_pred_cases,log_pred_deaths,beta,mu" << endl;
  for (int t = 0; t <= ntime; t++) 
  {
      report << obs_cases(t) << ","
             << obs_deaths (t) << ","
             << log_obs_cases(t) << ","
             << log_obs_deaths (t) << ","
             << logEye(t) << ","
             << logD(t) << ","
             << (beta_a + (beta_b - beta_a)*arlogit(logitbeta(t))) << ","
             << (mfexp(logmu))
             << endl; // ","
             //<< mu << endl;
  }

  report << "# ests:" << endl;
  report << "names,init,est" << endl;
  report << "logsigma_logP," << log(init_sigma_logP) << "," << logsigma_logP << endl;
  report << "logsigma_beta," << log(init_sigma_beta) << "," << logsigma_beta << endl;
  report << "logmu," << log(init_mu) << "," << logmu << endl;
  report << "loggamma," << log(init_gamma) << "," << loggamma << endl;
  report << "logsigma_logC," << log(init_sigma_logC) << "," << logsigma_logC << endl;
  report << "logsigma_logD," << log(init_sigma_logD) << "," << logsigma_logD << endl;
  report.precision(save_precision);
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
