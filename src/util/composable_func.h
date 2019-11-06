#ifndef AUTODIFF_COMPOSABLE_FUNC_H
#define AUTODIFF_COMPOSABLE_FUNC_H

#include <functional>
#include <vector>

namespace autodiff {
namespace util {

template<typename Ret, typename Arg>
class ComposableFunc {
 private:
  std::function<Ret(const Arg&)> func_;

 public:
  template<typename Func>
  ComposableFunc(const Func& func) : func_(func) {}

  const std::function<Ret(const Arg&)>& func() const { return func_; }

  Ret operator()(const Arg& arg) const { return func()(arg); }

  /*
  Non-member operator overloads:

  template<typename T, typename U, typename V>
  ComposableFunc<T, V> operator<<(
      const ComposableFunc<T, U>& lhs,
      const ComposableFunc<U, V>& rhs);
  
  template<typename T, typename U, typename V>
  ComposableFunc<T, V> operator>>(
      const ComposableFunc<T, U>& lhs,
      const ComposableFunc<U, V>& rhs);
  */
};

template<typename T, typename U, typename V>
std::function<T(V)> compose(const std::function<T(const U&)>& outer,
                            const std::function<U(const V&)>& inner) {
  return [outer, inner](const V& v) { return outer(inner(v)); };
}

template<typename T, typename U, typename V>
ComposableFunc<T, V> operator<<(
    const ComposableFunc<T, U>& lhs,
    const ComposableFunc<U, V>& rhs) {
  return ComposableFunc<T, V>(compose(lhs.func(), rhs.func()));
}

template<typename T, typename U, typename V>
ComposableFunc<T, V> operator>>(
    const ComposableFunc<U, V>& lhs,
    const ComposableFunc<T, U>& rhs) {
  return ComposableFunc<T, V>(compose(rhs.func(), lhs.func()));
}

} // namespace util
} // namespace autodiff

#endif
