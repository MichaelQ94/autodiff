#include <iostream>

#include "src/autodiff/autodiff.h"
#include "src/double/autodiff_double.h"

template<typename T>
void test(const autodiff::DualFunc<T>& func, const std::vector<T>& inputs) {
  auto derivative = autodiff::derivative(func);
  for (auto& t : inputs) {
    std::cout << "f(" << t << ") = " << func(autodiff::variable(t)).real()
        << ", f'(" << t << ") = " << derivative(t) << std::endl;
  }
}

int main() {
  autodiff::DualFunc<double> f([](autodiff::Dual<double> t) {
    return t * t;
  });

  autodiff::DualFunc<double> g([](autodiff::Dual<double> t) {
    return autodiff::constant<double>(0.5) * t;
  });

  autodiff::DualFunc<double> h([=](autodiff::Dual<double> t) {
    return g(f(t));
  });

  std::vector<double> inputs;

  for (double i = 0; i < 3.2; i += 0.1) {
    inputs.push_back(i);
  }

  test(h, inputs);

  return 0;
}
