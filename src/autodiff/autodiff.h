#ifndef AUTODIFF_AUTODIFF_H
#define AUTODIFF_AUTODIFF_H

#include <functional>
#include <vector>

#include "src/dual/dual.h"
#include "src/dual/dual_func.h"

namespace autodiff {

template<typename T>
std::function<T(const T&)> d(const DualFunc<T>& dual_func) {
  return [dual_func](const T& t) {
    return dual_func(var(t)).dual();
  };
}

template<typename T>
std::function<T(const std::vector<T>&)> p_d(
    const MultiVarDualFunc<T>& multi_var_dual_func, size_t index) {
  return [multi_var_dual_func, index](const std::vector<T>& args) {
    std::vector<Dual<T>> dual_args;
    dual_args.reserve(args.size());

    for (int i = 0; i < args.size(); ++i) {
      dual_args.push_back(i == index ? var(args[i]) : con(args[i]));
    }

    return multi_var_dual_func(dual_args).dual();
  };
}

} // namespace autodiff

#endif
