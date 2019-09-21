#include <iostream>
#include <vector>

#include "src/autodiff/autodiff.h"

template<typename T>
void test(const autodiff::DualFunc<T>& func, const std::vector<T>& inputs) {
  for (auto& t : inputs) {
    std::cout << "f(" << t << ") = " << func(autodiff::var(t)).real() << std::endl;
  }

  auto derivative = autodiff::d(func);
  for (auto& t : inputs) {
    std::cout << "f'(" << t << ") = " << derivative(t) << std::endl;
  }
}

int main() {
  auto f1 = autodiff::dual_func<double>([](autodiff::Dual<double> t) {
    return t * t;
  });

  auto f2 = autodiff::dual_func<double>([](autodiff::Dual<double> t) {
    return t * autodiff::con(1.0/2);
  });

  auto func = f1 >> f2;

  test(func, {0, 0.1, 0.2, 0.3, 0.00004, 0.00005, 6, 7, 8, 0.0009});

  return 0;
}
