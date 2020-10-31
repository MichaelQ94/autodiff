#pragma once

#include <memory>

#include "src/util/identity.h"

namespace autodiff {
namespace {

template<typename T> std::unique_ptr<T> wrap_unique(T* ptr) {
  return std::unique_ptr<T>(ptr);
}

/** Base class for smooth functions. Smooth functions should be evaluatable and differentiable. */
template<typename T, typename Identity>
class SmoothFnBase {
public:
  using Ptr = std::unique_ptr<SmoothFnBase<T, Identity>>;

  virtual T operator()(const T&) const = 0;
  virtual Ptr derivative() const = 0;
  virtual Ptr copy() const = 0;
};

template<typename T, typename Identity>
class ConstFn : public SmoothFnBase<T, Identity> {
private:
  const T value_;

  ConstFn(const T& value) : value_(value) {}

public:
  static typename SmoothFnBase<T, Identity>::Ptr make(const T& value) {
    return wrap_unique(new ConstFn<T, Identity>(value));
  }

  T operator()(const T& t) const {
    return value_;
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return make(Identity::zero());
  }

  typename SmoothFnBase<T, Identity>::Ptr copy() const {
    return make(value_);
  }
};

template<typename T, typename Identity>
class IdentityFn : public SmoothFnBase<T, Identity> {
private:
  IdentityFn() {}

public:
  static typename SmoothFnBase<T, Identity>::Ptr make() {
    return wrap_unique(new IdentityFn());
  }

  T operator()(const T& t) const {
    return t;
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return ConstFn<T, Identity>::make(Identity::one());
  }

  typename SmoothFnBase<T, Identity>::Ptr copy() const {
    return make();
  }
};

template<typename T, typename Identity>
class SumFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

  SumFn(typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs)
    : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

public:
  static typename SmoothFnBase<T, Identity>::Ptr make(
    typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs) {
    return wrap_unique(new SumFn(std::move(lhs), std::move(rhs)));
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) + (*rhs_)(t);
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return make(lhs_->derivative(), rhs_->derivative());
  }

  typename SmoothFnBase<T, Identity>::Ptr copy() const {
    return make(lhs_->copy(), rhs_->copy());
  }
};

template<typename T, typename Identity>
class DifferenceFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

  DifferenceFn(typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs)
    : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

public:
  static typename SmoothFnBase<T, Identity>::Ptr make(
    typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs) {
    return wrap_unique(new DifferenceFn(std::move(lhs), std::move(rhs)));
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) - (*rhs_)(t);
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return make(lhs_->derivative(), rhs_->derivative());
  }

  typename SmoothFnBase<T, Identity>::Ptr copy() const {
    return make(lhs_->copy(), rhs_->copy());
  }
};

template<typename T, typename Identity>
class ProductFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

  ProductFn(
    typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs)
    : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

public:
  static typename SmoothFnBase<T, Identity>::Ptr make(
    typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs) {
    return wrap_unique(new ProductFn(std::move(lhs), std::move(rhs)));
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) * (*rhs_)(t);
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return SumFn<T, Identity>::make(
      ProductFn<T, Identity>::make(lhs_->derivative(), rhs_->copy()),
      ProductFn<T, Identity>::make(lhs_->copy(), rhs_->derivative()));
  }

  typename SmoothFnBase<T, Identity>::Ptr copy() const {
    return make(lhs_->copy(), rhs_->copy());
  }
};

template<typename T, typename Identity>
class QuotientFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

  QuotientFn(typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs)
    : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

public:
  static typename SmoothFnBase<T, Identity>::Ptr make(
    typename SmoothFnBase<T, Identity>::Ptr&& lhs, typename SmoothFnBase<T, Identity>::Ptr&& rhs) {
    return wrap_unique(new QuotientFn(std::move(lhs), std::move(rhs)));
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) / (*rhs_)(t);
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return QuotientFn<T, Identity>::make(
      DifferenceFn<T, Identity>::make(
        ProductFn<T, Identity>::make(lhs_->derivative(), rhs_->copy()),
        ProductFn<T, Identity>::make(lhs_->copy(), rhs_->derivative())),
      ProductFn<T, Identity>::make(rhs_->copy(), rhs_->copy()));
  }

  typename SmoothFnBase<T, Identity>::Ptr copy() const {
    return make(lhs_->copy(), rhs_->copy());
  }
};

template<typename T, typename Identity>
class CompositeFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr outer_, inner_;

  CompositeFn(
    typename SmoothFnBase<T, Identity>::Ptr&& outer,
    typename SmoothFnBase<T, Identity>::Ptr&& inner)
    : outer_(std::move(outer)), inner_(std::move(inner)) {}

public:
  static typename SmoothFnBase<T, Identity>::Ptr make(
      typename SmoothFnBase<T, Identity>::Ptr&& outer,
      typename SmoothFnBase<T, Identity>::Ptr&& inner) {
    return wrap_unique(new CompositeFn(std::move(outer), std::move(inner)));
  }

  T operator()(const T& t) const {
    return (*outer_)((*inner_)(t));
  }

  // [f o g]' = [f' o g] * g'
  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return ProductFn<T, Identity>::make(
      CompositeFn<T, Identity>::make(outer_->derivative(), inner_->copy()),
      inner_->derivative());
  }

  typename SmoothFnBase<T, Identity>::Ptr copy() const {
    return make(inner_->copy(), outer_->copy());
  }
};

} // namespace

template<typename T, typename Identity = util::Identity<T>>
class SmoothFn {
private:
  const typename SmoothFnBase<T, Identity>::Ptr delegate_;
  
  SmoothFn(typename SmoothFnBase<T, Identity>::Ptr&& delegate) : delegate_(std::move(delegate)) {}

public:
  SmoothFn(const SmoothFn<T, Identity>& other) : delegate_(other.delegate_->copy()) {}
  SmoothFn(SmoothFn<T, Identity>&&) = default;
  SmoothFn& operator=(const SmoothFn<T, Identity>& other) {
    delegate_ = other.delegate_->copy();
    return *this;
  }
  SmoothFn& operator=(SmoothFn<T, Identity>&&) = default;
  ~SmoothFn() = default;

  static SmoothFn<T, Identity> identity() {
    return SmoothFn<T, Identity>(IdentityFn<T, Identity>::make());
  }


  T operator()(const T& t) const {
    return (*delegate_)(t);
  }

  SmoothFn<T, Identity> derivative() const {
    return SmoothFn<T, Identity>(delegate_->derivative());
  }

  SmoothFn<T, Identity> derivative(int n) const {
    return n == 0
      ? SmoothFn(delegate_->copy())
      : derivative(n - 1).derivative();
  }

  template<typename U, typename I>
  friend SmoothFn<U, I> operator+(const SmoothFn<U, I>& lhs, const SmoothFn<U, I>& rhs);

  template<typename U, typename I>
  friend SmoothFn<U, I> operator-(const SmoothFn<U, I>& lhs, const SmoothFn<U, I>& rhs);

  template<typename U, typename I>
  friend SmoothFn<U, I> operator*(const SmoothFn<U, I>& lhs, const SmoothFn<U, I>& rhs);

  template<typename U, typename I>
  friend SmoothFn<U, I> operator/(const SmoothFn<U, I>& lhs, const SmoothFn<U, I>& rhs);

  template<typename U, typename I>
  friend SmoothFn<U, I> operator<<(const SmoothFn<U, I>& lhs, const SmoothFn<U, I>& rhs);

  template<typename U, typename I>
  friend SmoothFn<U, I> operator>>(const SmoothFn<U, I>& lhs, const SmoothFn<U, I>& rhs);
};

template<typename T, typename Identity>
SmoothFn<T, Identity> operator+(
  const SmoothFn<T, Identity>& lhs, const SmoothFn<T, Identity>& rhs) {
  return SmoothFn<T, Identity>(
    SumFn<T, Identity>::make(
      lhs.delegate_->copy(),
      rhs.delegate_->copy()));
}

template<typename T, typename Identity>
SmoothFn<T, Identity> operator-(
  const SmoothFn<T, Identity>& lhs, const SmoothFn<T, Identity>& rhs) {
  return SmoothFn<T, Identity>(
    DifferenceFn<T, Identity>::make(
      lhs.delegate_->copy(),
      rhs.delegate_->copy()));
}

template<typename T, typename Identity>
SmoothFn<T, Identity> operator*(
  const SmoothFn<T, Identity>& lhs, const SmoothFn<T, Identity>& rhs) {
  return SmoothFn<T, Identity>(
    ProductFn<T, Identity>::make(
      lhs.delegate_->copy(),
      rhs.delegate_->copy()));
}

template<typename T, typename Identity>
SmoothFn<T, Identity> operator/(
  const SmoothFn<T, Identity>& lhs, const SmoothFn<T, Identity>& rhs) {
  return SmoothFn<T, Identity>(
    QuotientFn<T, Identity>::make(
      lhs.delegate_->copy(),
      rhs.delegate_->copy()));
}

template<typename T, typename Identity>
SmoothFn<T, Identity> operator<<(
  const SmoothFn<T, Identity>& lhs, const SmoothFn<T, Identity>& rhs) {
  return SmoothFn<T, Identity>(
    CompositeFn<T, Identity>::make(
      lhs.delegate_->copy(),
      rhs.delegate_->copy()));
}

template<typename T, typename Identity>
SmoothFn<T, Identity> operator>>(
  const SmoothFn<T, Identity>& lhs, const SmoothFn<T, Identity>& rhs) {
  return SmoothFn<T, Identity>(
    CompositeFn<T, Identity>::make(
      rhs.delegate_->copy(),
      lhs.delegate_->copy()));
}

} // namespace autodiff
