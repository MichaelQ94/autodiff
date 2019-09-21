#include <iostream>
#include <vector>

#include "src/dual/dual.h"
#include "src/dual/dual_func.h"

template<typename T>
void test(const autodiff::DualFunc<T>& func, const std::vector<T>& inputs) {
  for (auto& t : inputs) {
    std::cout << "f(" << t << ") = " << func(autodiff::var(t)).real() << std::endl;
  }

  for (auto& t : inputs) {
    std::cout << "f'(" << t << ") = " << func(autodiff::var(t)).dual() << std::endl;
  }
}

int main() {
  auto func = autodiff::dual_func<double>([](autodiff::Dual<double> t) {
    return (t * t * autodiff::con<double>(1.0 / 2)) + t;
  });

  test(func, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  return 0;
}