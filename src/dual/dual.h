#ifndef AUTODIFF_DUAL_H
#define AUTODIFF_DUAL_H

#include <functional>

namespace autodiff {

template<typename T>
class Dual {
 private:
  T real_, dual_;

 public:
  explicit Dual(const T& real, const T& dual) : real_(real), dual_(dual) {}

  const T& real() const { return real_; }
  const T& dual() const { return dual_; }

  Dual<T> operator+(const Dual<T>& rhs) const {
    return Dual(real() + rhs.real(), dual() + rhs.dual());
  }

  Dual<T> operator-(const Dual<T>& rhs) const {
    return Dual(real() - rhs.real(), dual() - rhs.dual());
  }

  Dual<T> operator*(const Dual<T>& rhs) const {
    return Dual(real() * rhs.real(),
                (dual() * rhs.real()) + (real() * rhs.dual()));
  }

  Dual<T> operator/(const Dual<T>& rhs) const {
    return Dual(
        real() / rhs.real(),
        (dual() / rhs.real()) - ((real() * rhs.dual()) / (rhs.real() * rhs.real())));
  }

  Dual<T> operator-() const {
    return Dual(-real(), -dual());
  }
};

} // namespace autodiff

#endif
