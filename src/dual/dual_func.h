#pragma once

#include <functional>
#include <vector>

#include "src/dual/dual.h"
#include "src/util/identity.h"

namespace autodiff {

template<typename T, typename Id = util::Identity<T>>
Dual<T> constant(const T& c) { return Dual<T>(c, Id::zero()); }

template<typename T, typename Id = util::Identity<T>>
Dual<T> variable(const T& t) { return Dual<T>(t, Id::one()); }

template <typename T>
using DualFunc = std::function<Dual<T>(const Dual<T>&)>;

template <typename T>
using MultiVarDualFunc = std::function<Dual<T>(const std::vector<Dual<T>>&)>;

template <typename T>
DualFunc<T> dual_func(const std::function<T(const T&)>& function,
                      const std::function<T(const T&)>& derivative) {
  return [function, derivative](const Dual<T> &t) {
    return Dual<T>(function(t.real()), derivative(t.real()) * t.dual()); 
  };
}

} // namespace autodiff
