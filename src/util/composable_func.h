#ifndef AUTODIFF_COMPOSABLE_FUNC_H
#define AUTODIFF_COMPOSABLE_FUNC_H

#include <functional>
#include <vector>

namespace autodiff {
namespace util {

template<typename T, typename U, typename V>
std::function<T(V)> compose(const std::function<T(const U&)>& outer,
                            const std::function<U(const V&)>& inner) {
  return [outer, inner](const V& v) { return outer(inner(v)); };
}

template<typename Ret, typename Arg>
class ComposableFunc {
 private:
  std::function<Ret(const Arg&)> func_;

 public:
  explicit ComposableFunc(const std::function<Ret(const Arg&)>& func) : func_(func) {}

  const std::function<Ret(const Arg&)>& func() const { return func_; }

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

} // namespace util
} // namespace autodiff

#endif
