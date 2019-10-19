#ifndef AUTODIFF_AUTODIFF_H
#define AUTODIFF_AUTODIFF_H

#include <functional>
#include <vector>

#include "src/dual/dual.h"
#include "src/dual/dual_func.h"

namespace autodiff {

template<typename T>
std::function<T(const T&)> derivative(const DualFunc<T>& dual_func) {
  return [dual_func](const T& t) {
    return dual_func(variable(t)).dual();
  };
}

template<typename T>
std::function<T(const std::vector<T>&)> partial_derivative(
    const MultiVarDualFunc<T>& func, size_t index) {
  return [func, index](const std::vector<T>& args) {
    std::vector<Dual<T>> dual_args;
    dual_args.reserve(args.size());

    for (int i = 0; i < args.size(); ++i) {
      dual_args.push_back(i == index ? variable(args[i]) : constant(args[i]));
    }

    return func(dual_args).dual();
  };
}

template<typename T>
std::function<T(const std::vector<T>&, const std::vector<T>&)> directional_derivative(
    const MultiVarDualFunc<T>& func) {
  return [func](const std::vector<T>& position,
      const std::vector<T>& velocity) {
    std::vector<Dual<T>> dual_args;
    dual_args.reserve(position.size());

    for (int i = 0; i < position.size(); ++i) {
      dual_args.push_back(Dual<T>(position[i], velocity[i]));
    }

    return func(dual_args).dual();
  };
}

} // namespace autodiff

#endif
