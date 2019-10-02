#include <iostream>
#include <vector>

#include "src/autodiff/autodiff.h"
#include "src/double/autodiff_double.h"

template<typename T>
void test_single(const autodiff::DualFunc<T>& func, const std::vector<T>& inputs) {
  for (auto& t : inputs) {
    std::cout << "f(" << t << ") = " << func(autodiff::var(t)).real() << std::endl;
  }

  auto derivative = autodiff::derivative(func);
  for (auto& t : inputs) {
    std::cout << "f'(" << t << ") = " << derivative(t) << std::endl;
  }
}

template<typename T>
void printvec(const std::vector<T>& vec) {
  for (auto& v : vec) {
    std::cout << v << ", ";
  }
}

template<typename T>
std::vector<autodiff::Dual<T>> map_to_duals(const std::vector<T>& values) {
  std::vector<autodiff::Dual<T>> duals;
  for (auto& value : values) {
    duals.push_back(autodiff::var(value));
  }
  return duals;
}

template<typename T>
void test_partial(const autodiff::MultiVarDualFunc<T>& func, const std::vector<std::vector<T>>& arglists) {
  int arity = arglists[0].size();
  for (auto& arglist : arglists) {
    std::cout << "f(";
    printvec(arglist);
    std::cout << ") = " << func(map_to_duals(arglist)).real() << std::endl;
  }

  for (int i = 0; i < arity; ++i) {
    auto partial_derivative = autodiff::partial_derivative(func, i);

    for (auto& arglist : arglists) {
      std::cout << "f_" << i << "(";
      printvec(arglist);
      std::cout << ") = " << partial_derivative(arglist) << std::endl;
    }
  }
}

template<typename T>
std::vector<T> unit_vector(size_t dimension, size_t index) {
  std::vector<T> v;
  for (int i = 0; i < dimension; ++i) {
    v.push_back(i == index ? 1 : 0);
  }
  return v;
}

template<typename T>
void test_directional(const autodiff::MultiVarDualFunc<T>& func, const std::vector<std::vector<T>>& arglists) {
  int arity = arglists[0].size();
  for (auto& arglist : arglists) {
    std::cout << "f(";
    printvec(arglist);
    std::cout << ") = " << func(map_to_duals(arglist)).real() << std::endl;
  }

  auto directional_derivative = autodiff::directional_derivative(func);

  for (int i = 0; i < arity; ++i) {
    for (auto& arglist : arglists) {
      std::cout << "f_" << i << "(";
      printvec(arglist);
      std::cout << ") = " << directional_derivative(arglist, unit_vector<T>(arity, i)) << std::endl;
    }
  }

}

int main() {
  std::vector<double> inputs;
  for (double t = 0; t < 3.2; t += 0.1) {
    inputs.push_back(t);
  }

  test_single(autodiff::dbl::exp, inputs);

  return 0;
}
