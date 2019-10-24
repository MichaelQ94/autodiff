#include <iostream>

#include <functional>

#include "src/autodiff/autodiff.h"
#include "src/double/autodiff_double.h"

template<typename T>
void test(const autodiff::MultiVarDualFunc<T>& func,
          size_t partial_derivative_index,
          const std::vector<std::vector<T>>& inputs) {
  auto partial_derivative = autodiff::partial_derivative(func, partial_derivative_index);

  for (const auto& vector : inputs) {
    std::cout << "f[" << partial_derivative_index << "](";

    for (const auto& component : vector) {
      std::cout << component << ", ";
    }
    
    std::cout << ") = " << partial_derivative(vector) << std::endl;
  }
}

std::vector<std::vector<double>> generate_inputs(
    double lower_bound, double upper_bound, double step_size,
    std::function<std::vector<double>(double)> curve) {
  std::vector<std::vector<double>> inputs;
  for (double t = lower_bound; t < upper_bound; t += step_size) {
    inputs.push_back(curve(t));
  }
  return inputs;
}

int main() {
  autodiff::MultiVarDualFunc<double> f([](const std::vector<autodiff::Dual<double>>& args) {
    return args[0] * autodiff::dbl::ln(args[1]);
  });

  double lower_bound = 0;
  double upper_bound = 1;
  double step_size = 0.1;

  size_t partial_derivative_index = 1;

  test(f,
       partial_derivative_index,
       generate_inputs(lower_bound, upper_bound, step_size,
                       [](double t) { return std::vector<double>({0, t}); }));
  test(f,
       partial_derivative_index,
       generate_inputs(lower_bound, upper_bound, step_size,
                       [](double t) { return std::vector<double>({t, 0}); }));
  test(f,
       partial_derivative_index,
       generate_inputs(lower_bound, upper_bound, step_size,
                       [](double t) { return std::vector<double>({t, t}); }));

  return 0;
}
