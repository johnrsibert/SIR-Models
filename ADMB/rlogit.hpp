#ifndef __RLOGIT__
#define __RLOGIT__

  template <typename Type> 
  Type rlogit(const double& a, const double&b, const Type& p)
  {
     Type y = a + (b-a)*log(p/(1.0-p));
     return y;
  }
  template double rlogit<double>(const double& a, const double& b, const double& p);
  template dvariable rlogit<dvariable>(const double& a, const double& b, const dvariable& p);

  template <typename Type> 
  Type arlogit(const double& a, const double& b,const Type& y)
  {
     Type x = 1.0/(1.0+mfexp(-y/(a+(b-a))));
     return x;
  }
  template double arlogit<double>(const double& a, const double& b, const double& y);
  template dvariable arlogit<dvariable>(const double& a, const double& b, const dvariable& y);

#endif
