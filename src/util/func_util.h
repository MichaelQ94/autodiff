#ifndef AUTODIFF_UTIL_FUNC_UTIL_H
#define AUTODIFF_UTIL_FUNC_UTIL_H

#include <functional>

namespace autodiff {
namespace util {

template<typename T, typename U, typename V>
std::function<T(V)> compose(const std::function<T(U)>& outer,
                            const std::function<U(V)>& inner) {
  return [outer, inner](V v) { return outer(inner(v)); };
}

template<typename Ret, typename Arg>
class ComposableFunc {
 private:
  std::function<Ret(Arg)> func_;

 public:
  explicit ComposableFunc(const std::function<Ret(Arg)>& func) : func_(func) {}

  const std::function<Ret(Arg)>& func() const { return func_; }

  Ret operator()(const Arg& arg) const { return func()(arg); }

  template<typename InnerArg>
  ComposableFunc<Ret, InnerArg> operator<<(
      const ComposableFunc<Arg, InnerArg>& inner) const {
    return ComposableFunc(compose(func(), inner.func()));
  }

  template<typename OuterRet>
  ComposableFunc<OuterRet, Arg> operator>>(
      const ComposableFunc<OuterRet, Ret>& outer) const {
    return ComposableFunc(compose(outer.func(), func()));
  }
};

template<typename T>
using EndoFunc = ComposableFunc<T, T>;

} // namespace util
} // namespace autodiff

#endif
