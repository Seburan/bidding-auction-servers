# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "inference_sidecar_onnxruntime_v1_20_0",
    srcs = ["@inference_common//:inference_sidecar_main.cc"],
    deps = [
        ":libonnxruntime",
        ":onnxruntime",
        "@inference_common//:grpc_sidecar",
    ],
)

cc_library(
    name = "onnxruntime",
    srcs = ["onnxruntime.cc"],
    deps = [
        ":libonnxruntime",
        ":onnxruntime_parser",
        "@com_google_absl//absl/status:statusor",
        "@inference_common//model:model_store",
        "@inference_common//modules:module_interface",
        "@inference_common//proto:inference_sidecar_cc_proto",
        "@inference_common//utils:error",
        "@inference_common//utils:inference_error_code",
        "@inference_common//utils:inference_metric_util",
        "@inference_common//utils:request_parser",
        "@libonnxruntime_v1_20_0//:onnxruntime_hdrs",
    ],
)

genrule(
    name = "test_pcvr_model_target",
    srcs = ["benchmark_models/pcvr_model"],
    outs = ["pcvr_model"],
    cmd = "cp $< $@",
)

genrule(
    name = "test_pctr_model_target",
    srcs = ["benchmark_models/pctr_model"],
    outs = ["pctr_model"],
    cmd = "cp $< $@",
)

cc_test(
    name = "onnxruntime_test",
    size = "small",
    srcs = ["onnxruntime_test.cc"],
    data = [
        ":pctr_model",
        ":pcvr_model",
    ],
    deps = [
        ":libonnxruntime",
        ":onnxruntime",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@inference_common//proto:inference_sidecar_cc_proto",
        "@inference_common//utils:file_util",
        "@inference_common//utils:inference_metric_util",
        "@inference_common//utils:test_util",
    ],
)

# Bazel environment requires `.so.1` suffix.
genrule(
    name = "generate_libonnxruntime",
    srcs = ["//docker:libonnxruntime.so.1.20.0"],
    outs = ["libonnxruntime.so.1"],
    cmd = "cp $(location //docker:libonnxruntime.so.1.20.0) $(location libonnxruntime.so.1)",
)

cc_import(
    name = "libonnxruntime",
    shared_library = ":generate_libonnxruntime",
)

cc_library(
    name = "onnxruntime_parser",
    srcs = ["onnxruntime_parser.cc"],
    hdrs = ["onnxruntime_parser.h"],
    deps = [
        ":libonnxruntime",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@inference_common//utils:error",
        "@inference_common//utils:request_parser",
        "@libonnxruntime_v1_20_0//:onnxruntime_hdrs",
    ],
)

cc_test(
    name = "onnxruntime_parser_test",
    size = "small",
    srcs = ["onnxruntime_parser_test.cc"],
    deps = [
        ":libonnxruntime",
        ":onnxruntime_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# Generate inference sidecar as "inference_sidecar" during test time.
genrule(
    name = "inference_sidecar_test_target",
    srcs = [":inference_sidecar_onnxruntime_v1_20_0"],
    outs = ["inference_sidecar"],
    cmd = "cp $< $@",
)

cc_library(
    name = "test_constants",
    testonly = True,
    hdrs = ["test_constants.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

genrule(
    name = "test_model_target",
    srcs = ["benchmark_models/simple_model.onnx"],
    outs = ["test_model"],
    cmd = "cp $< $@",
)

cc_test(
    name = "inference_sidecar_test",
    size = "medium",
    timeout = "short",
    srcs = ["@inference_common//:inference_sidecar_test.cc"],
    data = [
        ":inference_sidecar_test_target",
        ":test_model_target",
    ],
    flaky = True,
    tags = ["notsan"],
    deps = [
        ":test_constants",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:reflection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@inference_common//proto:inference_sidecar_cc_grpc_proto",
        "@inference_common//proto:inference_sidecar_cc_proto",
        "@inference_common//sandbox:sandbox_executor",
        "@inference_common//utils:file_util",
    ],
)

genrule(
    name = "collect-logs",
    outs = ["collect_logs.bin"],
    cmd_bash = """cat << EOF > '$@'
tools/collect-logs "\\$$@"
EOF""",
    executable = True,
    local = True,
    message = "copy bazel build and test logs",
)
