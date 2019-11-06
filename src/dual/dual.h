#ifndef AUTODIFF_DUAL_H
#define AUTODIFF_DUAL_H

#include <functional>

namespace autodiff {

template<typename T>
class Dual {
 private:
  T real_, dual_;

 public:
  explicit Dual(const T& real, const T& dual);

  const T& real() const;
  const T& dual() const;

  Dual<T> operator-() const;

  /*
  Non-member operator overloads:

  Dual<T> operator+(const Dual<T>& lhs, const Dual<T>& rhs);
  Dual<T> operator-(const Dual<T>& lhs, const Dual<T>& rhs);
  Dual<T> operator*(const Dual<T>& lhs, const Dual<T>& rhs);
  Dual<T> operator/(const Dual<T>& lhs, const Dual<T>& rhs);
  */
};

template<typename T>
Dual<T>::Dual(const T& real, const T& dual) : real_(real), dual_(dual) {}

template<typename T>
const T& Dual<T>::real() const { return real_; }

template<typename T>
const T& Dual<T>::dual() const { return dual_; }

template<typename T>
Dual<T> Dual<T>::operator-() const {
  return Dual<T>(-real(), -dual());
}

template<typename T>
Dual<T> operator+(const Dual<T>& lhs, const Dual<T>& rhs) {
  return Dual<T>(lhs.real() + rhs.real(), lhs.dual() + rhs.dual());
}

template<typename T>
Dual<T> operator-(const Dual<T>& lhs, const Dual<T>& rhs) {
  return Dual<T>(lhs.real() - rhs.real(), lhs.dual() - rhs.dual());
}

template<typename T>
Dual<T> operator*(const Dual<T>& lhs, const Dual<T>& rhs) {
  return Dual<T>(lhs.real() * rhs.real(),
              (lhs.dual() * rhs.real()) + (lhs.real() * rhs.dual()));
}

template<typename T>
Dual<T> operator/(const Dual<T>& lhs, const Dual<T>& rhs) {
  return Dual<T>(
      lhs.real() / rhs.real(),
      (lhs.dual() / rhs.real()) - ((lhs.real() * rhs.dual()) / (rhs.real() * rhs.real())));
}

} // namespace autodiff

#endif
