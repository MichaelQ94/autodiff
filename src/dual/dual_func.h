#ifndef AUTODIFF_DUAL_FUNC_H
#define AUTODIFF_DUAL_FUNC_H

#include <functional>

#include "src/dual/dual.h"
#include "src/util/composable_func.h"
#include "src/util/identity.h"

namespace autodiff {

template<typename T, typename Id = util::Identity<T>>
Dual<T> con(const T& c) { return Dual<T>(c, Id::zero()); }

template<typename T, typename Id = util::Identity<T>>
Dual<T> var(const T& t) { return Dual<T>(t, Id::one()); }

template<typename T>
using DualFunc = util::ComposableFunc<Dual<T>, Dual<T>>;

template<typename T>
DualFunc<T> dual_func(const std::function<T(T)>& function,
                      const std::function<T(T)>& derivative) {
  return DualFunc<T>([function, derivative](Dual<T> t) {
    return Dual<T>(function(t.real()),
                   derivative(t.real()) * t.dual());
  });
}

template<typename T>
using MultiVarDualFunc = util::ComposableFunc<Dual<T>, std::vector<Dual<T>>>;

} // namespace autodiff

#endif
