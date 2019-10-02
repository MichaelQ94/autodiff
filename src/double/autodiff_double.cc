#include "src/double/autodiff_double.h"

#include <cmath>

namespace autodiff {
namespace dbl {
namespace {

double (*std_exp)(double) = ::exp;
double (*std_ln)(double) = ::log;

double (*std_sin)(double) = ::sin;
double (*std_cos)(double) = ::cos;

double (*std_sinh)(double) = ::sinh;
double (*std_cosh)(double) = ::cosh;


} // namespace;

const DualFunc<double> exp(dual_func<double>(std_exp, std_exp));
const DualFunc<double> ln(dual_func<double>(std_ln, [](double t) { return 1.0 / t; }));

const DualFunc<double> sin(dual_func<double>(std_sin, std_cos));
const DualFunc<double> cos(dual_func<double>(std_cos, [](double t) { return -std_sin(t); }));

const DualFunc<double> sinh(dual_func<double>(std_sinh, std_cosh));
const DualFunc<double> cosh(dual_func<double>(std_cosh, std_sinh));

/*
Dual<double> exp(Dual<double> t) {
  return Dual(::exp(t.real()), ::exp(t.dual()));
}

Dual<double> ln(Dual<double> t) {
  return Dual(::log(t.real()), static_cast<double>(1) / t.dual());
}

Dual<double> sin(Dual<double> t) {
  return Dual(::sin(t.real()), ::cos(t.dual()));
}

Dual<double> cos(Dual<double> t) {
  return Dual(::cos(t.real()), -(::sin(t.dual())));
}*/

} // namespace double
} // namespace autodiff