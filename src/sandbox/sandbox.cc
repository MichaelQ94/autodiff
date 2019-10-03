#include <iostream>
#include <vector>

#include "src/autodiff/autodiff.h"
#include "src/double/autodiff_double.h"
#include "src/tensor/tensor.h"

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

template<typename T>
void test_tensor(const std::vector<std::vector<T>>& mat) {
  std::vector<autodiff::Vector<T>> vectors;
  for (const std::vector<T> vec : mat) {
    vectors.push_back(autodiff::make_vector(vec));
  }
  autodiff::Matrix<T> matrix = autodiff::make_tensor<T, 2>(vectors);

  for (int i = 0; i < matrix.shape()[0]; ++i) {
    for (int j = 0; j < matrix.shape()[1]; ++j) {
      std::cout << matrix[i][j] << ", ";
    }
    std::cout << std::endl;
  }
}

int main() {
  std::vector<std::vector<double>> mat;

  mat.push_back({1, 2, 3});
  mat.push_back({4, 5, 6});
  mat.push_back({7, 8, 9});

  test_tensor(mat);

  return 0;
}
