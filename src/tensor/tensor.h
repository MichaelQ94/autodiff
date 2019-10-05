#ifndef AUTODIFF_TENSOR_H
#define AUTODIFF_TENSOR_H

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace autodiff {

template<typename T, size_t Order>
class Tensor;

namespace tensor {

template<size_t Order>
using Shape = std::array<size_t, Order>;

template<typename T, size_t Order>
struct Data {
  Shape<Order> shape_;
  std::vector<Tensor<T, Order - 1>> sub_tensors_;

  Data(const Shape<Order>& shape,
      const std::vector<Tensor<T, Order - 1>>& sub_tensors)
    : shape_(shape), sub_tensors_(sub_tensors) {}
};

} // namespace tensor

template<typename T, size_t Order>
class Tensor {
 private:
  std::shared_ptr<const tensor::Data<T, Order>> data_;

 public:
  Tensor(const std::shared_ptr<tensor::Data<T, Order>>& data) : data_(data) {}
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor& operator=(Tensor&&) = default;

  size_t order() const {
    return Order;
  }

  const Tensor<T, Order - 1>& operator[](size_t index) const {
    return data_->sub_tensors_[index];
  }

  const tensor::Shape<Order>& shape() const {
    return data_->shape_;
  }
};

template<>
template<typename T>
class Tensor<T, 0> {
 private:
  static const tensor::Shape<0> SCALAR_SHAPE;

  T value_;

 public:
  Tensor(const T& value) : value_(value) {}
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor& operator=(Tensor&&) = default;

  operator T() const {
    return value_;
  }

  constexpr size_t order() const {
    return 0;
  }

  const tensor::Shape<0>& shape() const {
    return SCALAR_SHAPE;
  }

  Tensor operator+(const Tensor& rhs) const {
    return Tensor(value_ + rhs.value_);
  }

  Tensor operator-(const Tensor& rhs) const {
    return Tensor(value_ - rhs.value_);
  }

  Tensor operator*(const Tensor& rhs) const {
    return Tensor(value_ * rhs.value_);
  }

  Tensor operator/(const Tensor& rhs) const {
    return Tensor(value_ / rhs.value_);
  }

  Tensor operator-() const {
    return Tensor(-value_);
  }
};

template<typename T>
using Scalar = Tensor<T, 0>;

template<typename T>
using Vector = Tensor<T, 1>;

template<typename T>
using Matrix = Tensor<T, 2>;

template<typename T, size_t Order>
using DTensor = Tensor<Dual<T>, Order>;

template<typename T>
using DScalar = DTensor<T, 0>;

template<typename T>
using DVector = DTensor<T, 1>;

template<typename T>
using DMatrix = DTensor<T, 2>;

template<typename T>
Vector<T> make_vector(const std::vector<T>& scalars) {
  std::vector<Scalar<T>> sub_tensors;
  sub_tensors.reserve(scalars.size());
  for (const T& scalar : scalars) {
    sub_tensors.push_back(scalar);
  }

  return Vector<T>(
    std::make_shared<tensor::Data<T, 1>>(
      std::array<size_t, 1>({sub_tensors.size()}), sub_tensors));
}

template<typename T, size_t Order>
Tensor<T, Order> make_tensor(const std::vector<Tensor<T, Order - 1>>& sub_tensors) {
  const tensor::Shape<Order - 1>& sub_shape = sub_tensors[0].shape();
  for (int i = 1; i < sub_tensors.size(); ++i) {
    if (sub_tensors[i].shape() != sub_shape) {
      throw std::invalid_argument("Tensors must be constructed out of sub-tensors with identical shapes.");
    }
  }

  tensor::Shape<Order> shape;
  for (int i = 0; i < Order - 1; ++i) {
    shape[i] = sub_shape[i];
  }
  shape[Order - 1] = sub_tensors.size();

  return Tensor<T, Order>(
    std::make_shared<tensor::Data<T, Order>>(
      shape, sub_tensors));
}

} // namespace autodiff

#endif
