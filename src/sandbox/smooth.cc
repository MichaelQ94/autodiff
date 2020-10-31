#include <iostream>
#include <vector>

#include "src/smooth/smooth.h"

template<typename T>
void test(const autodiff::SmoothFn<T>& func, const std::vector<T>& inputs) {
  autodiff::SmoothFn<T> derivative = func.derivative();
  for (auto& t : inputs) {
    std::cout << "f(" << t << ") = " << func(t)
        << ", f'(" << t << ") = " << derivative(t) << std::endl;
  }
}

int main() {
  autodiff::SmoothFn<double> f =
    autodiff::SmoothFn<double>::identity()
      * autodiff::SmoothFn<double>::identity()
      * autodiff::SmoothFn<double>::identity();

  std::vector<double> inputs;

  for (double i = 0; i < 3.2; i += 0.1) {
    inputs.push_back(i);
  }

  test(f, inputs);

  return 0;
}