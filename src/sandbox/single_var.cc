#include <cmath>
#include <functional>
#include <iostream>

#include "src/autodiff/autodiff.h"
#include "src/double/autodiff_double.h"

template<typename T>
void test(
  const autodiff::DualFunc<T>& func,
  const std::function<T(T)> expected_derivative,
  const std::vector<T>& inputs) {
  auto derivative = autodiff::derivative(func);
  for (auto& t : inputs) {
    autodiff::Dual<T> output = func(autodiff::variable(t));
    std::cout << "f(" << t << ") = " << output.real()
        << ", f'(" << t << ") = " << output.dual()
        << " (expected " << expected_derivative(t) << ")" << std::endl;
  }
}

autodiff::Dual<double> half_square(autodiff::Dual<double> t) {
  return autodiff::constant(1.0 / 2) * t * t;
}

// sin((1/2)t^2)
// expect derivative to be: cos((1/2)t^2) * t
autodiff::Dual<double> func(autodiff::Dual<double> t) {
  return autodiff::dbl::sin(half_square(t));
}

double expected_derivative(double t) {
  return std::cos((1.0 / 2) * t * t) * t;
}

int main() {
  test<double>(func, expected_derivative, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  return 0;
}
