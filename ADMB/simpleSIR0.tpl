   //  simpleSIR3 -noinit -est -shess

GLOBALS_SECTION;
  #include <math.h>
  #include <adstring.hpp>
  #include "trace.h"
  #include <admodel.h>
  #include <fvar.hpp>
  //#include "rlogit.hpp"

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

  init_int phase_Eye0;
  init_number init_Eye0;
  !!  TTRACE(phase_Eye0,init_Eye0)

  init_int phase_logbeta; 
  init_number init_beta;
  !!  TTRACE(phase_logbeta,init_beta)
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
  init_number logEye0(phase_Eye0);

  // state variables
  //init_vector logEye(0,ntime);    // number of infections
  //init_vector logD(0,ntime);      // number of deaths from infected population
  random_effects_vector logEye(0,ntime);    // number of infections
  random_effects_vector logD(0,ntime);      // number of deaths from infected population

  //init_vector logbeta(0,ntime,phase_logitbeta);       // infection rate time series
  random_effects_vector logbeta(0,ntime,phase_logbeta);       // infection rate time series


  sdreport_number sigma_logP;
  sdreport_number sigma_beta;
  //sdreport_number sigma_logC;
  //sdreport_number sigma_logD;
  //sdreport_number Eye0;


  objective_function_value nll;

PRELIMINARY_CALCS_SECTION
  logsigma_logP = log(init_sigma_logP);
  TTRACE(logsigma_logP,active(logsigma_logP))
  logsigma_beta = log(init_sigma_beta);
  TTRACE(logsigma_beta,active(logsigma_beta))
  logmu = log(init_mu);
  TTRACE(logmu,active(logmu))
  loggamma = log(init_gamma);
  TTRACE(loggamma,active(loggamma))
  logsigma_logC = log(init_sigma_logC);
  TTRACE(logsigma_logC,active(logsigma_logC))
  logsigma_logD = log(init_sigma_logD);
  TTRACE(logsigma_logD,active(logsigma_logD))

  double init_logbeta = log(init_beta); //rlogit(beta_a,beta_b,double(init_beta));
  TRACE(active(logbeta))
  TTRACE(beta_a,beta_b)
  TTRACE(init_beta,init_logbeta)

  logEye0 = log(init_Eye0);
  TTRACE(logEye0,active(logEye0))

  int zero_count = 0;
  for (int t = 0; t <=  ntime; t++)
  {
       logbeta(t) = init_logbeta;
       log_obs_cases(t) = log(obs_cases(t)+eps);
     //TTRACE(t,obs_cases(t))
       if (obs_deaths(t) < 1.0)
            zero_count ++;
       log_obs_deaths(t) = log(obs_deaths(t)+eps);
       TTRACE(obs_deaths(t),log_obs_deaths(t))
       logEye(t) = 1.0;
       logD(t) = eps;
  }
  TRACE(logbeta)
  TRACE(logEye)
  TRACE(logD)
  TRACE(log_obs_deaths)
  TTRACE(zero_count,ntime)
  prop_zero_deaths = double(zero_count)/double(ntime+1);
  TRACE(prop_zero_deaths)
  //  if(1)  ad_exit(1);
  HERE
 
RUNTIME_SECTION
  convergence_criteria .01

PROCEDURE_SECTION
  //f = 0.0;
  
  // compute process errors
  for (int t = 1; t <=  ntime; t++)
  {
       step(t, logbeta(t-1),logbeta(t), logsigma_beta, logEye(t-1), logEye(t), logD(t-1), logD(t), logsigma_logP, loggamma, logmu);
  }

  // compute observation likelihoods
  for (int t = 0; t <= ntime; t++)
  {   
       obs(t, logEye(t), logsigma_logC, logD(t), logsigma_logD);
  }

  if (sd_phase())
  {
      sigma_logP = mfexp(logsigma_logP);
      sigma_beta = mfexp(logsigma_beta);
      //sigma_logC = mfexp(logsigma_logC);
      //sigma_logD = mfexp(logsigma_logD);
      //Eye0 = mfexp(logEye(0));
   }

SEPARABLE_FUNCTION void step(const int t, const dvariable& lb1, const dvariable& lbt, const dvariable& logsigma_beta, const dvariable& logEye1, const dvariable& logEyet, const dvariable& logD1, const dvariable& logDt, const dvariable& logsigma_logP, const dvariable& loggamma, const dvariable& logmu)
  // call:     step(t, logbeta(t-1),logbeta(t), logsigma_beta, logEye(t-1), logEye(t), logD(t-1), logD(t), logsigma_logP, loggamma, logmu);

           dvariable betanll = 0.0;
           // infection rate random walk
           dvariable pbeta = mfexp(lb1); //arlogit(beta_a,beta_b,lb1+1e-20);
           dvariable nbeta = mfexp(lbt); //arlogit(beta_a,beta_b,lbt+1e-20);
           dvariable varb = square(mfexp(logsigma_beta));
           betanll += 0.5*(log(TWO_M_PI*varb) + square(pbeta-nbeta)/varb);
 
           dvariable gamma = mfexp(loggamma);
           dvariable mu = mfexp(logmu);
           dvariable prevlogEye = logEye1;
           // mixed explicit-implicit approximation:
           //dvariable nextlogEye = prevlogEye - log(1.0 - pbeta) + log(1.0 - (gamma+mu));
           // explicit approximation:
           dvariable nextlogEye = prevlogEye + log(1.0 + pbeta - gamma - mu);

           // cases process error
           dvariable Pnll = 0.0;
           dvariable varP = square(mfexp(logsigma_logP));
           Pnll += 0.5*(log(TWO_M_PI*varP) + square(logEye1-nextlogEye)/varP);
           if (isnan(value(Pnll)))
           {
                TRACE(t)
                TTRACE(Pnll,varP)
                TTRACE(logEye1,logEyet)
                TTRACE(prevlogEye,nextlogEye)
                TTRACE(pbeta,nbeta)
                TTRACE(gamma,mu)
                ad_exit(1);
           }

           // deaths process error
           dvariable prevD = mfexp(logD1);
           dvariable nextlogD = log(prevD + mu*mfexp(prevlogEye)+eps);
           TTRACE(prevD,mfexp(nextlogD))
           
           if(logD(t-1) > logeps)
           {
              Pnll += (1.0-prop_zero_deaths)*0.5*(log(TWO_M_PI*varP) + square(logD1 - nextlogD)/varP);
           }
           else
           {
              Pnll += prop_zero_deaths*0.5*(log(TWO_M_PI*varP));
           }

           nll += (betanll + Pnll);
           TTRACE(betanll,Pnll)
           TTRACE(t,nll)


SEPARABLE_FUNCTION void obs(const int t, const dvariable& logEyet, const dvariable& logsigma_logC, const dvariable& logDt, const dvariable& logsigma_logD)
  // call: obs(t, logEye(t), logsigma_logC, logD(t), logsigma_logD);
           dvariable cnll = 0.0;
           dvariable varC = square(mfexp(logsigma_logC));
           cnll+= 0.5*(log(TWO_M_PI*varC) + square(log_obs_cases(t)-logEyet)/varC);

           dvariable dnll = 0.0;

           dvariable varD = square(mfexp(logsigma_logD));
           dnll += 0.5*(log(TWO_M_PI*varD) + square(log_obs_deaths(t) - logDt)/varD);
         //TTRACE(log_obs_deaths(t),logDt)
          
         //TTRACE(t,mfexp(log_obs_deaths(t)))
           if (log_obs_deaths(t) > logeps)  //log zero deaths
           {
              dnll += (1.0-prop_zero_deaths)*0.5*(log(TWO_M_PI*varD) + square(log_obs_deaths(t) - logDt)/varD);
           }
           else
           {
              dnll += prop_zero_deaths*0.5*(log(TWO_M_PI*varD));
           }
         
           nll += (cnll + dnll);
           TTRACE(cnll,dnll)
           TTRACE(t,nll)

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
             << mfexp(logbeta(t)) << ","
       //    << arlogit(beta_a,beta_b,logitbeta(t)) << ","
       //    << (beta_a + (beta_b - beta_a)*arlogit(logitbeta(t))) << ","
             << (mfexp(logmu))
             << endl; // ","
             //<< mu << endl;
  }

  report << "# ests:" << endl;
  report << "names,init,est,active" << endl;
  report << "logsigma_logP," << log(init_sigma_logP) << "," << logsigma_logP << ","
         << active(logsigma_logP) << endl;
  report << "logsigma_beta," << log(init_sigma_beta) << "," << logsigma_beta << ","
         << active(logsigma_beta) << endl;
  report << "logmu," << log(init_mu) << "," << logmu << "," << active(logmu) << endl;
  report << "loggamma," << log(init_gamma) << "," << loggamma << "," << active(loggamma) << endl;
  report << "logsigma_logC," << log(init_sigma_logC) << "," << logsigma_logC << ","
         << active(logsigma_logC) << endl;
  report << "logsigma_logD," << log(init_sigma_logD) << "," << logsigma_logD << ","
         << active(logsigma_logD) << endl;
  report << "logEye0," << 0.0 << "," << logEye0 << "," << active(logEye0) << endl;
  report.precision(save_precision);
  /*
  REPORT(logsigma_logP)
  REPORT(logsigma_beta)
  REPORT(logmu)
  REPORT(loggamma)
  REPORT(logsigma_logC)
  REPORT(logsigma_logD)
  REPORT(logEye0)
  */
