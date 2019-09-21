#ifndef AUTODIFF_DUAL_DUAL_FUNC_H
#define AUTODIFF_DUAL_DUAL_FUNC_H

#include <functional>

#include "src/util/func_util.h"
#include "src/util/identity.h"

namespace autodiff {

template<typename T, typename Id = util::Identity<T>>
Dual<T> con(const T& c) { return Dual<T>(c, Id::zero()); }

template<typename T, typename Id = util::Identity<T>>
Dual<T> var(const T& t) { return Dual<T>(t, Id::one()); }

template<typename T>
using DualFunc = util::EndoFunc<Dual<T>>;

template<typename T>
DualFunc<T> dual_func(const std::function<Dual<T>(Dual<T>)>& dual_func) {
  return DualFunc<T>(dual_func);
}

template<typename T>
DualFunc<T> dual_func(const std::function<T(T)>& function,
                      const std::function<T(T)>& derivative) {
  return dual_func([function, derivative](Dual<T> t) {
    return Dual<T>(function(t.real()),
                derivative(t.real()) * t.dual());
  });
}

} // namespace autodiff

#endif
