#pragma once

#include <memory>

#include "src/util/identity.h"

namespace autodiff {
namespace {

/** Base class for smooth functions. Smooth functions should be evaluatable and differentiable. */
template<typename T, typename Identity>
class SmoothFnBase {
public:
  using Ptr = std::shared_ptr<const SmoothFnBase<T, Identity>>;

  virtual T operator()(const T&) const = 0;
  virtual Ptr derivative() const = 0;
};

template<typename T, typename Identity>
class ConstFn : public SmoothFnBase<T, Identity> {
private:
  const T value_;

public:
  ConstFn(const T& value) : value_(value) {}

  static const typename SmoothFnBase<T, Identity>::Ptr ZERO;
  static const typename SmoothFnBase<T, Identity>::Ptr ONE;

  static typename SmoothFnBase<T, Identity>::Ptr make(const T& value) {
    return std::make_shared<ConstFn<T, Identity>>(value);
  }

  T operator()(const T& t) const {
    return value_;
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return ZERO;
  }
};

template<typename T, typename Identity>
const typename SmoothFnBase<T, Identity>::Ptr ConstFn<T, Identity>::ZERO
  = ConstFn<T, Identity>::make(Identity::zero());

template<typename T, typename Identity>
const typename SmoothFnBase<T, Identity>::Ptr ConstFn<T, Identity>::ONE
  = ConstFn<T, Identity>::make(Identity::one());

template<typename T, typename Identity>
class IdentityFn : public SmoothFnBase<T, Identity> {
  static const typename SmoothFnBase<T, Identity>::Ptr INSTANCE;

public:

  static typename SmoothFnBase<T, Identity>::Ptr make() {
    return INSTANCE;
  }

  T operator()(const T& t) const {
    return t;
  }

  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return ConstFn<T, Identity>::ONE;
  }
};

template<typename T, typename Identity>
const typename SmoothFnBase<T, Identity>::Ptr IdentityFn<T, Identity>::INSTANCE
  = std::make_shared<IdentityFn>();

template<typename T, typename Identity>
class SumFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

public:
  SumFn(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs)
    : lhs_(lhs), rhs_(rhs) {}

  static typename SmoothFnBase<T, Identity>::Ptr make(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs) {
    return std::make_shared<SumFn<T, Identity>>(lhs, rhs);
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) + (*rhs_)(t);
  }

  // (f + g)' = f' + g'
  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return make(lhs_->derivative(), rhs_->derivative());
  }
};

template<typename T, typename Identity>
class DifferenceFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

public:
  DifferenceFn(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs)
    : lhs_(lhs), rhs_(rhs) {}

  static typename SmoothFnBase<T, Identity>::Ptr make(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs) {
    return std::make_shared<DifferenceFn<T, Identity>>(lhs, rhs);
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) - (*rhs_)(t);
  }

  // (f - g)' = f' - g'
  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return make(lhs_->derivative(), rhs_->derivative());
  }
};

template<typename T, typename Identity>
class ProductFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

public:
  ProductFn(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs)
    : lhs_(lhs), rhs_(rhs) {}

  static typename SmoothFnBase<T, Identity>::Ptr make(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs) {
    return std::make_shared<ProductFn<T, Identity>>(lhs, rhs);
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) * (*rhs_)(t);
  }

  // (f * g)' = (f' * g) + (f * g')
  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return SumFn<T, Identity>::make(
      ProductFn<T, Identity>::make(lhs_->derivative(), rhs_),
      ProductFn<T, Identity>::make(lhs_, rhs_->derivative()));
  }
};

template<typename T, typename Identity>
class QuotientFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr lhs_, rhs_;

public:
  QuotientFn(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs)
    : lhs_(lhs), rhs_(rhs) {}

  static typename SmoothFnBase<T, Identity>::Ptr make(
      const typename SmoothFnBase<T, Identity>::Ptr& lhs,
      const typename SmoothFnBase<T, Identity>::Ptr& rhs) {
    return std::make_shared<QuotientFn<T, Identity>>(lhs, rhs);
  }

  T operator()(const T& t) const {
    return (*lhs_)(t) / (*rhs_)(t);
  }

  // (f / g)' = ((f' * g) - (f * g')) / (g * g)
  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return QuotientFn<T, Identity>::make(
      DifferenceFn<T, Identity>::make(
        ProductFn<T, Identity>::make(lhs_->derivative(), rhs_),
        ProductFn<T, Identity>::make(lhs_, rhs_->derivative())),
      ProductFn<T, Identity>::make(rhs_, rhs_));
  }
};

template<typename T, typename Identity>
class CompositeFn : public SmoothFnBase<T, Identity> {
private:
  const typename SmoothFnBase<T, Identity>::Ptr outer_, inner_;

public:
  CompositeFn(
      const typename SmoothFnBase<T, Identity>::Ptr& outer,
      const typename SmoothFnBase<T, Identity>::Ptr& inner)
    : outer_(outer), inner_(inner) {}

  static typename SmoothFnBase<T, Identity>::Ptr make(
      const typename SmoothFnBase<T, Identity>::Ptr& outer,
      const typename SmoothFnBase<T, Identity>::Ptr& inner) {
    return std::make_shared<CompositeFn<T, Identity>>(outer, inner);
  }

  T operator()(const T& t) const {
    return (*outer_)((*inner_)(t));
  }

  // (f o g)' = (f' o g) * g'
  typename SmoothFnBase<T, Identity>::Ptr derivative() const {
    return ProductFn<T, Identity>::make(
      CompositeFn<T, Identity>::make(outer_->derivative(), inner_),
      inner_->derivative());
  }
};

} // namespace

template<typename T, typename Identity = util::Identity<T>>
class SmoothFn {
private:
  typename SmoothFnBase<T, Identity>::Ptr delegate_;
  
  SmoothFn(const typename SmoothFnBase<T, Identity>::Ptr& delegate) : delegate_(delegate) {}

public:
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
    return n == 0 ? *this: derivative(n - 1).derivative();
  }

  SmoothFn<T, Identity> operator+(const SmoothFn<T, Identity>& rhs) const {
    return SmoothFn<T, Identity>(SumFn<T, Identity>::make(delegate_, rhs.delegate_));
  }

  SmoothFn<T, Identity> operator-(const SmoothFn<T, Identity>& rhs) const {
    return SmoothFn<T, Identity>(DifferenceFn<T, Identity>::make(delegate_, rhs.delegate_));
  }

  SmoothFn<T, Identity> operator*(const SmoothFn<T, Identity>& rhs) const {
    return SmoothFn<T, Identity>(ProductFn<T, Identity>::make(delegate_, rhs.delegate_));
  }

  SmoothFn<T, Identity> operator/(const SmoothFn<T, Identity>& rhs) const {
    return SmoothFn<T, Identity>(QuotientFn<T, Identity>::make(delegate_, rhs.delegate_));
  }

  SmoothFn<T, Identity> operator<<(const SmoothFn<T, Identity>& rhs) const {
    return SmoothFn<T, Identity>(CompositeFn<T, Identity>::make(delegate_, rhs.delegate_));
  }

  SmoothFn<T, Identity> operator>>(const SmoothFn<T, Identity>& rhs) const {
    return SmoothFn<T, Identity>(CompositeFn<T, Identity>::make(rhs.delegate_, delegate_));
  }
};

} // namespace autodiff
