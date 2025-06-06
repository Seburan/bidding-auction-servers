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

#include "onnxruntime_parser.h"

#include <algorithm>
#include <utility>

#include "absl/status/statusor.h"
#include "googletest/include/gtest/gtest.h"
#include "src/util/status_macro/status_macros.h"

#include "onnxruntime_cxx_api.h"

namespace privacy_sandbox::bidding_auction_servers::inference {
namespace {

TEST(OnnxruntimeParserTest, TestConversion) {
  Tensor tensor;
  tensor.data_type = DataType::kFloat;
  tensor.tensor_content = {"1.2", "-2", "3", "4", "5", "6"};
  tensor.tensor_shape = {2, 3};

  const absl::StatusOr<Ort::Value> ort_tensor =
      ConvertFlatArrayToTensor(tensor);
  EXPECT_TRUE(ort_tensor.ok());

  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetDimensionsCount(), 2);
  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetShape()[0], 2);
  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetShape()[1], 3);

  for (int i = 0; i < tensor.tensor_content.size(); i++) {
    EXPECT_FLOAT_EQ(ort_tensor->GetTensorData<float>()[i],
                    std::stof(tensor.tensor_content[i]));
  }
}

TEST(OnnxruntimeParserTest, TestConversionInt8) {
  Tensor tensor;
  tensor.data_type = DataType::kInt8;
  tensor.tensor_content = {"5"};
  tensor.tensor_shape = {1, 1};

  const absl::StatusOr<Ort::Value> ort_tensor =
      ConvertFlatArrayToTensor(tensor);
  EXPECT_TRUE(ort_tensor.ok());

  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetElementType(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);

  EXPECT_EQ(ort_tensor->GetTensorData<int8_t>()[0], 5);
}

TEST(OnnxruntimeParserTest, TestConversionInt16) {
  Tensor tensor;
  tensor.data_type = DataType::kInt16;
  tensor.tensor_content = {"5"};
  tensor.tensor_shape = {1, 1};

  const absl::StatusOr<Ort::Value> ort_tensor =
      ConvertFlatArrayToTensor(tensor);
  EXPECT_TRUE(ort_tensor.ok());

  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetElementType(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);

  EXPECT_EQ(ort_tensor->GetTensorData<int16_t>()[0], 5);
}

TEST(OnnxruntimeParserTest, TestConversionInt32) {
  Tensor tensor;
  tensor.data_type = DataType::kInt32;
  tensor.tensor_content = {"5"};
  tensor.tensor_shape = {1, 1};

  const absl::StatusOr<Ort::Value> ort_tensor =
      ConvertFlatArrayToTensor(tensor);
  EXPECT_TRUE(ort_tensor.ok());

  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetElementType(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

  EXPECT_EQ(ort_tensor->GetTensorData<int>()[0], 5);
}

TEST(OnnxruntimeParserTest, TestConversion_BatchSizeOne) {
  Tensor tensor;
  tensor.data_type = DataType::kInt64;
  tensor.tensor_content = {"7"};
  tensor.tensor_shape = {1, 1};

  const absl::StatusOr<Ort::Value> ort_tensor =
      ConvertFlatArrayToTensor(tensor);
  EXPECT_TRUE(ort_tensor.ok());

  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetDimensionsCount(), 2);
  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetShape()[0], 1);
  EXPECT_EQ(ort_tensor->GetTensorTypeAndShapeInfo().GetShape()[1], 1);

  EXPECT_EQ(ort_tensor->GetTensorData<int64_t>()[0], 7);
}

TEST(OnnxruntimeParserTest, TestConversion_WrongType) {
  Tensor tensor;
  tensor.data_type = DataType::kInt64;
  tensor.tensor_content = {"seven"};  // Incompatible type.
  tensor.tensor_shape = {1, 1};

  const absl::StatusOr result = ConvertFlatArrayToTensor(tensor);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(result.status().message(), "Error in int64 conversion");
}

TEST(OnnxruntimeParserTest, ConvertTensorsOrErrorToJson) {
  const std::vector<int64_t> tensor_shape({2, 3});  // 2x3 tensor
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::Value ort_tensor = Ort::Value::CreateTensor<float>(
      alloc, tensor_shape.data(), tensor_shape.size());

  // Access and modify tensor elements.
  auto tensor_data = ort_tensor.GetTensorMutableData<float>();
  for (int i = 0; i < 6; ++i) {
    tensor_data[i] = i * 2.0f;  // Set values (i * 2)
  }
  std::vector<TensorWithName> tensors;
  tensors.push_back({.tensor_name = "output", .tensor = std::move(ort_tensor)});
  std::vector<TensorsOrError> batch_tensors;
  batch_tensors.push_back({.model_path = "my_bucket/models/pcvr_models/1/",
                           .tensors = std::move(tensors)});

  auto output = ConvertBatchOutputsToJson(batch_tensors);
  EXPECT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(
      output.value(),
      R"({"response":[{"model_path":"my_bucket/models/pcvr_models/1/","tensors":[{"tensor_name":"output","tensor_shape":[2,3],"data_type":"FLOAT","tensor_content":[0.0,2.0,4.0,6.0,8.0,10.0]}]}]})");
}

TEST(OnnxruntimeParserTest, ConvertTensorsOrErrorToJson_UnsupportedType) {
  const std::vector<int64_t> tensor_shape({2, 3});  // 2x3 tensor
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::Value ort_tensor = Ort::Value::CreateTensor<bool>(
      alloc, tensor_shape.data(), tensor_shape.size());

  std::vector<TensorWithName> tensors;
  tensors.push_back({.tensor_name = "output", .tensor = std::move(ort_tensor)});
  std::vector<TensorsOrError> batch_tensors;
  batch_tensors.push_back({.model_path = "my_bucket/models/pcvr_models/1/",
                           .tensors = std::move(tensors)});

  auto output = ConvertBatchOutputsToJson(batch_tensors);
  EXPECT_FALSE(output.ok());
  EXPECT_EQ(output.status().message(), "Unsupported tensor data type: 9");
}

TEST(OnnxruntimeParserTest, ConvertTensorsOrErrorToJson_int8) {
  const std::vector<int64_t> tensor_shape({2});
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::Value ort_tensor = Ort::Value::CreateTensor<int8_t>(
      alloc, tensor_shape.data(), tensor_shape.size());

  auto tensor_data = ort_tensor.GetTensorMutableData<int8_t>();
  const std::vector<int8_t> int8_data = {10, -5};
  std::copy(int8_data.begin(), int8_data.end(), tensor_data);

  std::vector<TensorWithName> tensors;
  tensors.push_back({.tensor_name = "output", .tensor = std::move(ort_tensor)});
  std::vector<TensorsOrError> batch_tensors;
  batch_tensors.push_back({.model_path = "my_bucket/models/pcvr_models/1/",
                           .tensors = std::move(tensors)});

  auto output = ConvertBatchOutputsToJson(batch_tensors);
  EXPECT_TRUE(output.ok());
  EXPECT_EQ(
      output.value(),
      R"({"response":[{"model_path":"my_bucket/models/pcvr_models/1/","tensors":[{"tensor_name":"output","tensor_shape":[2],"data_type":"INT8","tensor_content":[10,-5]}]}]})");
}

TEST(OnnxruntimeParserTest, ConvertTensorsOrErrorToJson_int16) {
  const std::vector<int64_t> tensor_shape({2});
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::Value ort_tensor = Ort::Value::CreateTensor<int16_t>(
      alloc, tensor_shape.data(), tensor_shape.size());

  auto tensor_data = ort_tensor.GetTensorMutableData<int16_t>();
  const std::vector<int16_t> int16_data = {10, -5};
  std::copy(int16_data.begin(), int16_data.end(), tensor_data);

  std::vector<TensorWithName> tensors;
  tensors.push_back({.tensor_name = "output", .tensor = std::move(ort_tensor)});
  std::vector<TensorsOrError> batch_tensors;
  batch_tensors.push_back({.model_path = "my_bucket/models/pcvr_models/1/",
                           .tensors = std::move(tensors)});

  auto output = ConvertBatchOutputsToJson(batch_tensors);
  EXPECT_TRUE(output.ok());
  EXPECT_EQ(
      output.value(),
      R"({"response":[{"model_path":"my_bucket/models/pcvr_models/1/","tensors":[{"tensor_name":"output","tensor_shape":[2],"data_type":"INT16","tensor_content":[10,-5]}]}]})");
}

TEST(OnnxruntimeParserTest, TestConvertTensorsOrErrorToJson_MultipleTensors) {
  Ort::AllocatorWithDefaultOptions alloc;
  // Double tensor.
  const std::vector<int64_t> tensor_shape_1({1, 1});
  Ort::Value ort_tensor_1 = Ort::Value::CreateTensor<double>(
      alloc, tensor_shape_1.data(), tensor_shape_1.size());
  *ort_tensor_1.GetTensorMutableData<double>() = 3.14;

  // Int64 tensor.
  const std::vector<int64_t> tensor_shape_2({1, 1});
  Ort::Value ort_tensor_2 = Ort::Value::CreateTensor<int64_t>(
      alloc, tensor_shape_2.data(), tensor_shape_2.size());
  *ort_tensor_2.GetTensorMutableData<int64_t>() = 1000;

  std::vector<TensorWithName> tensors;
  tensors.push_back(
      {.tensor_name = "output1", .tensor = std::move(ort_tensor_1)});
  tensors.push_back(
      {.tensor_name = "output2", .tensor = std::move(ort_tensor_2)});

  std::vector<TensorsOrError> batch_tensors;
  batch_tensors.push_back({.model_path = "my_bucket/models/pcvr_models/1/",
                           .tensors = std::move(tensors)});

  auto output = ConvertBatchOutputsToJson(batch_tensors);
  EXPECT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(
      output.value(),
      R"({"response":[{"model_path":"my_bucket/models/pcvr_models/1/","tensors":[{"tensor_name":"output1","tensor_shape":[1,1],"data_type":"DOUBLE","tensor_content":[3.14]},{"tensor_name":"output2","tensor_shape":[1,1],"data_type":"INT64","tensor_content":[1000]}]}]})");
}

TEST(OnnxruntimeParserTest, TestConvertTensorsOrErrorToJson_BatchOfModels) {
  Ort::AllocatorWithDefaultOptions alloc;
  // Double tensor.
  const std::vector<int64_t> tensor_shape_1({1, 1});
  Ort::Value ort_tensor_1 = Ort::Value::CreateTensor<double>(
      alloc, tensor_shape_1.data(), tensor_shape_1.size());
  *ort_tensor_1.GetTensorMutableData<double>() = 3.14;

  // Int64 tensor.
  const std::vector<int64_t> tensor_shape_2({1, 1});
  Ort::Value ort_tensor_2 = Ort::Value::CreateTensor<int64_t>(
      alloc, tensor_shape_2.data(), tensor_shape_2.size());
  *ort_tensor_2.GetTensorMutableData<int64_t>() = 1000;

  std::vector<TensorWithName> tensors_1;
  tensors_1.push_back(
      {.tensor_name = "output1", .tensor = std::move(ort_tensor_1)});
  std::vector<TensorWithName> tensors_2;
  tensors_2.push_back(
      {.tensor_name = "output2", .tensor = std::move(ort_tensor_2)});
  std::vector<TensorsOrError> batch_tensors;
  batch_tensors.push_back({.model_path = "my_bucket/models/pcvr_models/1/",
                           .tensors = std::move(tensors_1)});
  batch_tensors.push_back({.model_path = "my_bucket/models/pctr_models/1/",
                           .tensors = std::move(tensors_2)});

  auto output = ConvertBatchOutputsToJson(batch_tensors);
  EXPECT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(
      output.value(),
      R"({"response":[{"model_path":"my_bucket/models/pcvr_models/1/","tensors":[{"tensor_name":"output1","tensor_shape":[1,1],"data_type":"DOUBLE","tensor_content":[3.14]}]},{"model_path":"my_bucket/models/pctr_models/1/","tensors":[{"tensor_name":"output2","tensor_shape":[1,1],"data_type":"INT64","tensor_content":[1000]}]}]})");
}

TEST(OnnxruntimeParserTest, TestConvertTensorsOrErrorToJson_PartialResult) {
  const std::vector<int64_t> tensor_shape({1, 1});
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::Value ort_tensor = Ort::Value::CreateTensor<double>(
      alloc, tensor_shape.data(), tensor_shape.size());
  *ort_tensor.GetTensorMutableData<double>() = 3.14;

  std::vector<TensorWithName> tensors;
  tensors.push_back(
      {.tensor_name = "output1", .tensor = std::move(ort_tensor)});
  std::vector<TensorsOrError> batch_tensors;
  batch_tensors.push_back({.model_path = "my_bucket/models/pcvr_models/1/",
                           .tensors = std::move(tensors)});
  batch_tensors.push_back({.model_path = "my_bucket/models/pctr_models/1/",
                           .error = Error{.error_type = Error::MODEL_NOT_FOUND,
                                          .description = "Model not found."}});

  auto output = ConvertBatchOutputsToJson(batch_tensors);
  EXPECT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(
      output.value(),
      R"({"response":[{"model_path":"my_bucket/models/pcvr_models/1/","tensors":[{"tensor_name":"output1","tensor_shape":[1,1],"data_type":"DOUBLE","tensor_content":[3.14]}]},{"model_path":"my_bucket/models/pctr_models/1/","error":{"error_type":"MODEL_NOT_FOUND","description":"Model not found."}}]})");
}

}  // namespace
}  // namespace privacy_sandbox::bidding_auction_servers::inference
