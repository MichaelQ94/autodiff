package(
  default_visibility = ["//visibility:public"]
)

cc_library(
  name = "dual",
  hdrs = ["dual.h"],
)

cc_library(
  name = "dual_func",
  hdrs = ["dual_func.h"],
  deps = [
    "dual",
    "//src/tensor:tensor",
    "//src/util:composable_func",
    "//src/util:identity"
  ]
)
