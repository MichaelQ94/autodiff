#ifndef AUTODIFF_UTIL_IDENTITY_H
#define AUTODIFF_UTIL_IDENTITY_H

namespace autodiff {
namespace util {

template<typename T>
struct Identity {
  static T zero() { return T(0); }
  static T one() { return T(1); }
};

} // namespace util
} // namespace autodiff

#endif
