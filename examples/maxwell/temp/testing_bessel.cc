#include <cmath>
#include <iostream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_message.h>
#include <gsl/gsl_sf_bessel.h>

int main()
{
    double result;
    double nu;
    double x;
    int i;
    int n;
    x=1;
    nu=1./3.;
    std::cout << nu << std::endl;
    n=3;
//    result=1.1;
    for (i=0;i<11;i++)
    {
        x=0.1*i;
//        result= gsl_sf_bessel_Jn(n,x);
        result = gsl_sf_bessel_Jnu(nu,x);
    std::cout << result << " ";
    }
    std::cout << std::endl;
    return 0;
};