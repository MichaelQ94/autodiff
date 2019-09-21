#ifndef AUTODIFF_AUTODIFF_H
#define AUTODIFF_AUTODIFF_H

#include <functional>

#include "src/dual/dual.h"
#include "src/dual/dual_func.h"

namespace autodiff {

template<typename T>
std::function<T(T)> d(const DualFunc<T>& dual_func) {
  return [dual_func](const T& t) {
    return dual_func(var(t)).dual();
  };
}

} // namespace autodiff

#endif
