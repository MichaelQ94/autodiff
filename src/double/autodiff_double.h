#pragma once

#include "src/dual/dual_func.h"

namespace autodiff {
namespace dbl {

extern const DualFunc<double> exp;
extern const DualFunc<double> ln;

extern const DualFunc<double> sin;
extern const DualFunc<double> cos;

extern const DualFunc<double> sinh;
extern const DualFunc<double> cosh;

} // namespace double
} // namespace autodiff
