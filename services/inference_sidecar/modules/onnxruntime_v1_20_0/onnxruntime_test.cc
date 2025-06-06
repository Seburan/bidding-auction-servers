// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <gmock/gmock-matchers.h>

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "modules/module_interface.h"
#include "proto/inference_sidecar.pb.h"
#include "utils/file_util.h"
#include "utils/test_util.h"

namespace privacy_sandbox::bidding_auction_servers::inference {
namespace {

using ::testing::HasSubstr;
using ::testing::StartsWith;

constexpr absl::string_view kPcvrModelPath = "pcvr_model";
constexpr absl::string_view kPctrModelPath = "pctr_model";

constexpr char kPcvrJsonRequest[] = R"json({
  "request" : [{
    "model_path" : "pcvr_model",
    "tensors" : [
    {
      "tensor_name": "args_0",
      "data_type": "DOUBLE",
      "tensor_shape": [
        1, 10
      ],
      "tensor_content": ["0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11"]
    },
    {
      "tensor_name": "args_0_1",
      "data_type": "DOUBLE",
      "tensor_shape": [
        1, 10
      ],
      "tensor_content": ["0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11"]
    },
    {
      "tensor_name": "args_0_2",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_3",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_4",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_5",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_6",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    }
  ]
}]
    })json";

constexpr absl::string_view kPcvrJsonResponse =
    "{\"response\":[{\"model_path\":\"pcvr_model\",\"tensors\":[{\"tensor_"
    "name\":\"output_0\",\"tensor_shape\":[1,1],\"data_type\":\"FLOAT\","
    "\"tensor_content\":[0.011748075485229493]}]}]}";

TEST(OnnxModuleRegisterTest, Success_RegisterModel) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  EXPECT_TRUE(ort_module->RegisterModel(register_request).ok());
}

TEST(OnnxModuleRegisterTest, Failure_RegisterModelWithEmptyPath) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  register_request.mutable_model_spec()->set_model_path("");

  absl::StatusOr<RegisterModelResponse> response =
      ort_module->RegisterModel(register_request);
  ASSERT_FALSE(response.ok());

  EXPECT_EQ(response.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(response.status().message(),
            "Empty model path during registration");
}

TEST(OnnxModuleRegisterTest, Failure_RegisterModelAlreadyRegistered) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  // First model registration
  RegisterModelRequest register_request_1;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request_1).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request_1).ok());

  // Second registration attempt with the same model path
  RegisterModelRequest register_request_2;
  // Same model path as the first attempt
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request_2).ok());
  absl::StatusOr<RegisterModelResponse> response_2 =
      ort_module->RegisterModel(register_request_2);

  ASSERT_FALSE(response_2.ok());
  EXPECT_EQ(response_2.status().code(), absl::StatusCode::kAlreadyExists);
}

TEST(OnnxModuleRegisterTest, Failure_RegisterModelIncorrectFileCount) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  // Setting the model path without providing model content
  RegisterModelRequest register_request;
  register_request.mutable_model_spec()->set_model_path(kPcvrModelPath);
  absl::StatusOr<RegisterModelResponse> response =
      ort_module->RegisterModel(register_request);

  ASSERT_FALSE(response.ok());
  EXPECT_EQ(response.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(response.status().message(),
            "The number of model files should be exactly 1, but got 0");
}

TEST(OnnxModuleRegisterTest, Failure_RegisterModelInvalidModel) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  // Setting the model path and providing invalid model content.
  RegisterModelRequest register_request;
  register_request.mutable_model_spec()->set_model_path(kPcvrModelPath);
  (*register_request.mutable_model_files())[kPcvrModelPath] = " ";
  absl::StatusOr<RegisterModelResponse> response =
      ort_module->RegisterModel(register_request);

  ASSERT_FALSE(response.ok());
  EXPECT_EQ(response.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(response.status().message(),
              StartsWith("Exception thrown during Onnxruntime model loading"));
}

TEST(OnnxModuleRegisterTest, Success_RegisterModelWithWarmUpRequest) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  register_request.set_warm_up_batch_request_json(kPcvrJsonRequest);
  EXPECT_TRUE(ort_module->RegisterModel(register_request).ok());
}

TEST(OnnxModuleRegisterTest,
     Success_RegisterModelDefaultReturnNoConstructMetric) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  auto register_response = ort_module->RegisterModel(register_request);
  ASSERT_TRUE(register_response.ok());
  EXPECT_EQ(register_response.value().metrics_list_size(), 0);
}

TEST(OnnxModuleRegisterTest,
     Success_RegisterModelWithWarmUpDataReturnConstructMetric) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  register_request.set_warm_up_batch_request_json(kPcvrJsonRequest);
  auto register_response = ort_module->RegisterModel(register_request);
  ASSERT_TRUE(register_response.ok());
  EXPECT_EQ(register_response.value().metrics_list_size(), 1);
  ASSERT_TRUE(register_response.value().metrics_list().find(
                  "kInferenceRegisterModelResponseModelWarmUpDuration") !=
              register_response.value().metrics_list().end());
}

TEST(OnnxModulePredictTest, Success_Predict) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequest);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);

  ASSERT_TRUE(predict_response.ok());
  ASSERT_EQ(predict_response->output(), kPcvrJsonResponse);
  ASSERT_FALSE(predict_response->metrics_list().empty());
  EXPECT_EQ(predict_response->metrics_list().size(), 7);
}

TEST(OnnxModulePredictTest, JsonError_PredictInvalidJson) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input("");
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);

  ASSERT_TRUE(predict_response.ok());
  EXPECT_THAT(
      predict_response->output(),
      StartsWith(
          R"({"response":[{"error":{"error_type":"INPUT_PARSING","description")"));
}

TEST(OnnxModulePredictTest, JsonError_PredictModelNotRegistered) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequest);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);

  ASSERT_TRUE(predict_response.ok());
  EXPECT_THAT(
      predict_response->output(),
      StartsWith(
          R"({"response":[{"model_path":"pcvr_model","error":{"error_type":"MODEL_NOT_FOUND","description")"));
}

constexpr char kPcvrJsonRequestMissingTensorName[] = R"json({
  "request" : [{
    "model_path" : "pcvr_model",
    "tensors" : [
    {
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    }
  ]
}]
    })json";

TEST(OnnxruntimeModuleTest, JsonError_PredictMissingTensorName) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequestMissingTensorName);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);

  ASSERT_TRUE(predict_response.ok());
  EXPECT_THAT(
      predict_response->output(),
      StartsWith(
          R"({"response":[{"model_path":"pcvr_model","error":{"error_type":"INPUT_PARSING","description")"));
}

constexpr char kPcvrJsonRequestInvalidTensor[] = R"json({
  "request" : [{
    "model_path" : "pcvr_model",
    "tensors" : [
    {
      "tensor_name": "args_0",
      "data_type": "FLOAT",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["seven"]
    }
  ]
}]
    })json";

TEST(OnnxruntimeModuleTest, JsonError_PredictInvalidTensor) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequestInvalidTensor);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);

  ASSERT_TRUE(predict_response.ok());
  EXPECT_THAT(
      predict_response->output(),
      StartsWith(
          R"({"response":[{"model_path":"pcvr_model","error":{"error_type":"INPUT_PARSING","description")"));
}

constexpr char kPcvrJsonRequestWrongInputTensor[] = R"json({
  "request" : [{
    "model_path" : "pcvr_model",
    "tensors" : [
    {
      "tensor_name": "wrong_tensor",
      "data_type": "FLOAT",
      "tensor_shape": [
        2, 2
      ],
      "tensor_content": ["1.1", "1.1", "1.1", "1.1"]
    }
  ]
}]
    })json";

TEST(OnnxruntimeModuleTest, JsonError_PredictModelExecutionError) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequestWrongInputTensor);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);

  ASSERT_TRUE(predict_response.ok());
  EXPECT_THAT(
      predict_response->output(),
      StartsWith(
          R"({"response":[{"model_path":"pcvr_model","error":{"error_type":"MODEL_EXECUTION","description")"));
}

TEST(OnnxModulePredictTest, Success_PredictValidateMetrics) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequest);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);
  ASSERT_TRUE(predict_response.ok());
  ASSERT_FALSE(predict_response->output().empty());
  ASSERT_EQ(predict_response->output(), kPcvrJsonResponse);
  ASSERT_FALSE(predict_response->metrics_list().empty());
  EXPECT_EQ(predict_response->metrics_list().size(), 7);
  CheckMetricList(predict_response->metrics_list(), "kInferenceRequestCount", 0,
                  1);
  CheckMetricList(predict_response->metrics_list(), "kInferenceRequestSize", 0,
                  1286);
  CheckMetricList(predict_response->metrics_list(),
                  "kInferenceRequestBatchCountByModel", 0, 1);

  auto it = predict_response->metrics_list().find("kInferenceRequestDuration");
  ASSERT_NE(it, predict_response->metrics_list().end())
      << "kInferenceRequestDuration metric is missing.";
}

TEST(OnnxModulePredictTest, Success_PredictWithConsentedRequest) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequest);
  predict_request.set_is_consented(true);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);
  ASSERT_TRUE(predict_response.ok());
  ASSERT_FALSE(predict_response->output().empty());
  ASSERT_EQ(predict_response->output(), kPcvrJsonResponse);
  ASSERT_FALSE(predict_response->metrics_list().empty());
  EXPECT_EQ(predict_response->metrics_list().size(), 7);
}

constexpr char kPcvrJsonRequestBatchSize2[] = R"json({
  "request" : [{
    "model_path" : "pcvr_model",
    "tensors" : [
    {
      "tensor_name": "args_0",
      "data_type": "DOUBLE",
      "tensor_shape": [
        2, 10
      ],
      "tensor_content": ["0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11"]
    },
    {
      "tensor_name": "args_0_1",
      "data_type": "DOUBLE",
      "tensor_shape": [
        2, 10
      ],
      "tensor_content": ["0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11"]
    },
    {
      "tensor_name": "args_0_2",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["7", "3"]
    },
    {
      "tensor_name": "args_0_3",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["7", "3"]
    },
    {
      "tensor_name": "args_0_4",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["7", "3"]
    },
    {
      "tensor_name": "args_0_5",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["7", "3"]
    },
    {
      "tensor_name": "args_0_6",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["7", "3"]
    }
  ]
}]
    })json";

TEST(OnnxModulePredictTest, Success_PredictBatchSize2) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequestBatchSize2);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);
  ASSERT_TRUE(predict_response.ok());
  ASSERT_FALSE(predict_response->output().empty());
  ASSERT_EQ(
      predict_response->output(),
      "{\"response\":[{\"model_path\":\"pcvr_model\",\"tensors\":[{\"tensor_"
      "name\":\"output_0\",\"tensor_shape\":[2,1],\"data_type\":\"FLOAT\","
      "\"tensor_content\":[0.011748075485229493,0.10649198293685913]}]}]}");
  ASSERT_FALSE(predict_response->metrics_list().empty());
  EXPECT_EQ(predict_response->metrics_list().size(), 7);
}

constexpr char kPcvrJsonRequestWith2Model[] = R"json({
  "request" : [{
    "model_path" : "pcvr_model",
    "tensors" : [
    {
      "tensor_name": "args_0",
      "data_type": "DOUBLE",
      "tensor_shape": [
        1, 10
      ],
      "tensor_content": ["0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11"]
    },
    {
      "tensor_name": "args_0_1",
      "data_type": "DOUBLE",
      "tensor_shape": [
        1, 10
      ],
      "tensor_content": ["0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.32", "0.12", "0.98", "0.11"]
    },
    {
      "tensor_name": "args_0_2",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_3",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_4",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_5",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    },
    {
      "tensor_name": "args_0_6",
      "data_type": "INT64",
      "tensor_shape": [
        1, 1
      ],
      "tensor_content": ["7"]
    }
  ]
},
{
    "model_path" : "pctr_model",
    "tensors" : [
    {
      "tensor_name": "args_0",
      "data_type": "DOUBLE",
      "tensor_shape": [
        2, 10
      ],
      "tensor_content": ["0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.12", "0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.12"]
    },
    {
      "tensor_name": "args_0_1",
      "data_type": "DOUBLE",
      "tensor_shape": [
        2, 10
      ],
      "tensor_content": ["0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.12", "0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.33", "0.13", "0.97", "0.12"]
    },
    {
      "tensor_name": "args_0_2",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["8", "4"]
    },
    {
      "tensor_name": "args_0_3",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["8", "4"]
    },
    {
      "tensor_name": "args_0_4",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["8", "4"]
    },
    {
      "tensor_name": "args_0_5",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["8", "4"]
    },
    {
      "tensor_name": "args_0_6",
      "data_type": "INT64",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["8", "4"]
    }
  ]
}]
    })json";

TEST(OnnxModulePredictTest, Success_PredictWith2Models) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request_1;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request_1).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request_1).ok());

  RegisterModelRequest register_request_2;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPctrModelPath, register_request_2).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request_2).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequestWith2Model);
  absl::StatusOr<PredictResponse> predict_response =
      ort_module->Predict(predict_request);
  ASSERT_TRUE(predict_response.ok());
  ASSERT_FALSE(predict_response->output().empty());
  EXPECT_EQ(predict_response->output(),
            "{\"response\":[{\"model_path\":\"pcvr_model\",\"tensors\":[{"
            "\"tensor_name\":\"output_0\",\"tensor_shape\":[1,1],\"data_type\":"
            "\"FLOAT\",\"tensor_content\":[0.011748075485229493]}]},{\"model_"
            "path\":\"pctr_model\",\"tensors\":[{\"tensor_name\":\"output_0\","
            "\"tensor_shape\":[2,1],\"data_type\":\"FLOAT\",\"tensor_content\":"
            "[0.8405165076255798,0.7376009225845337]}]}]}");
  ASSERT_FALSE(predict_response->metrics_list().empty());
  EXPECT_EQ(predict_response->metrics_list().size(), 7);
}

TEST(OnnxModulePredictTest, CanReturnPartialBatchOutput) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequestWith2Model);
  absl::StatusOr predict_output = ort_module->Predict(predict_request);
  ASSERT_TRUE(predict_output.ok());
  EXPECT_THAT(predict_output->output(),
              AllOf(HasSubstr("MODEL_NOT_FOUND"), HasSubstr("\"tensors\":")));
}

constexpr char kTwoInvalidInputsJsonRequest[] = R"json({
  "request" : [{
    "model_path" : "pcvr_model",
    "tensors" : [
    {
      "tensor_name": "wrong_tensor",
      "data_type": "FLOAT",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["7", "3"]
    }
  ]
},
{
    "model_path" : "pctr_model",
    "tensors" : [
    {
      "tensor_name": "wrong_tensor",
      "data_type": "FLOAT",
      "tensor_shape": [
        2, 1
      ],
      "tensor_content": ["7", "3"]
    }
  ]
}]
    })json";

TEST(OnnxModulePredictTest, CanReturnBatchOutputWithAllErrors) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request_1;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request_1).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request_1).ok());

  RegisterModelRequest register_request_2;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPctrModelPath, register_request_2).ok());
  ASSERT_TRUE(ort_module->RegisterModel(register_request_2).ok());

  PredictRequest predict_request;
  predict_request.set_input(kTwoInvalidInputsJsonRequest);
  absl::StatusOr predict_output = ort_module->Predict(predict_request);
  ASSERT_TRUE(predict_output.ok());
  EXPECT_THAT(predict_output->output(), AllOf(HasSubstr("MODEL_EXECUTION"),
                                              HasSubstr("MODEL_EXECUTION")));
}

TEST(OnnxModuleTest, Failure_DeleteModel_NotFound) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  DeleteModelRequest delete_request;
  delete_request.mutable_model_spec()->set_model_path(
      std::string(kPcvrModelPath));
  auto status = ort_module->DeleteModel(delete_request);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kNotFound);
}

TEST(OnnxModuleTest, Success_DeleteModel) {
  InferenceSidecarRuntimeConfig config;
  std::unique_ptr<ModuleInterface> ort_module = ModuleInterface::Create(config);
  RegisterModelRequest register_request;
  ASSERT_TRUE(
      PopulateRegisterModelRequest(kPcvrModelPath, register_request).ok());
  absl::StatusOr<RegisterModelResponse> status_or =
      ort_module->RegisterModel(register_request);
  ASSERT_TRUE(status_or.ok()) << status_or.status();

  DeleteModelRequest delete_request;
  delete_request.mutable_model_spec()->set_model_path(
      std::string(kPcvrModelPath));
  EXPECT_TRUE(ort_module->DeleteModel(delete_request).ok());

  PredictRequest predict_request;
  predict_request.set_input(kPcvrJsonRequest);
  auto predict_response = ort_module->Predict(predict_request);
  ASSERT_TRUE(predict_response.ok());
  EXPECT_THAT(
      predict_response->output(),
      StartsWith(
          R"({"response":[{"model_path":"pcvr_model","error":{"error_type":"MODEL_NOT_FOUND","description")"));
}

}  // namespace
}  // namespace privacy_sandbox::bidding_auction_servers::inference
