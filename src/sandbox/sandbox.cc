#include <iostream>
#include <vector>

#include "src/autodiff/autodiff.h"

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
  /*
  autodiff::DualFunc<double> f1([](autodiff::Dual<double> t) {
    return t * t;
  });

  autodiff::DualFunc<double> f2([](const autodiff::Dual<double>& t) {
    return t * autodiff::con(1.0/2);
  });
  auto func = f1 >> f2;

  test_single(func, {0, 0.1, 0.2, 0.3, 0.00004, 0.00005, 6, 7, 8, 0.0009});
  */

  autodiff::MultiVarDualFunc<double> f([](const std::vector<autodiff::Dual<double>>& args) {
    return args[0] + (args[1] * args[1]);
  });

  std::vector<std::vector<double>> test_args;

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      test_args.push_back({(double) i, (double) j});
    }
  }

  test_directional(f, test_args);

  return 0;
}
