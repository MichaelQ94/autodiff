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
  autodiff::DualFunc<double> func([](autodiff::Dual<double> t) {
    return t * t;
  });

  test(func, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  return 0;
}
