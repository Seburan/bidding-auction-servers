//  Copyright 2024 Google LLC
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "onnxruntime_parser.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "rapidjson/document.h"
#include "src/util/status_macro/status_macros.h"
#include "utils/error.h"
#include "utils/json_util.h"
#include "utils/request_parser.h"

#include "onnxruntime_cxx_api.h"

namespace privacy_sandbox::bidding_auction_servers::inference {
namespace {

template <typename T>
absl::StatusOr<Ort::Value> ConvertFlatArrayToTensorInternal(
    const Tensor& tensor) {
  try {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::Value ort_tensor = Ort::Value::CreateTensor<T>(
        alloc, tensor.tensor_shape.data(), tensor.tensor_shape.size());

    // The invariant that `tensor.tensor_content` is equal to the product of the
    // `tensor.tensor_shape` array is already checked in request_parser.cc.
    for (size_t i = 0;
         i < ort_tensor.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
      PS_ASSIGN_OR_RETURN(T result, Convert<T>(tensor.tensor_content[i]));
      ort_tensor.GetTensorMutableData<T>()[i] = result;
    }

    return ort_tensor;
  } catch (const Ort::Exception& e) {
    return absl::InternalError(absl::StrCat(
        "Exception thrown during conversion to ONNX tensor: ", e.what()));
  } catch (...) {
    return absl::InternalError(
        "Unknown exception occurred during conversion to ONNX tensor.");
  }
}

// Converts an Onnxruntime tensor to JSON.
absl::StatusOr<rapidjson::Value> TensorToJsonValue(
    const std::string& tensor_name, const Ort::Value& tensor,
    rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value json_tensor(rapidjson::kObjectType);
  rapidjson::Value tensor_name_value;
  tensor_name_value.SetString(tensor_name.c_str(), allocator);
  json_tensor.AddMember("tensor_name", tensor_name_value, allocator);

  std::vector<int64_t> tensor_shape =
      tensor.GetTensorTypeAndShapeInfo().GetShape();
  rapidjson::Value tensor_shape_json(rapidjson::kArrayType);
  for (size_t i = 0; i < tensor_shape.size(); ++i) {
    tensor_shape_json.PushBack(tensor_shape[i], allocator);
  }
  json_tensor.AddMember("tensor_shape", tensor_shape_json, allocator);

  // Flatten the tensor.
  ONNXTensorElementDataType data_type =
      tensor.GetTensorTypeAndShapeInfo().GetElementType();
  size_t tensor_count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  rapidjson::Value tensor_content(rapidjson::kArrayType);

  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      json_tensor.AddMember("data_type", "FLOAT", allocator);
      const float* data = tensor.GetTensorData<float>();
      for (size_t i = 0; i < tensor_count; ++i) {
        tensor_content.PushBack(data[i], allocator);
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      json_tensor.AddMember("data_type", "DOUBLE", allocator);
      const double* data = tensor.GetTensorData<double>();
      for (size_t i = 0; i < tensor_count; ++i) {
        tensor_content.PushBack(data[i], allocator);
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      json_tensor.AddMember("data_type", "INT8", allocator);
      const int8_t* data = tensor.GetTensorData<int8_t>();
      for (size_t i = 0; i < tensor_count; ++i) {
        tensor_content.PushBack(data[i], allocator);
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      json_tensor.AddMember("data_type", "INT16", allocator);
      const int16_t* data = tensor.GetTensorData<int16_t>();
      for (size_t i = 0; i < tensor_count; ++i) {
        tensor_content.PushBack(data[i], allocator);
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      json_tensor.AddMember("data_type", "INT32", allocator);
      const int32_t* data = tensor.GetTensorData<int32_t>();
      for (size_t i = 0; i < tensor_count; ++i) {
        tensor_content.PushBack(data[i], allocator);
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      json_tensor.AddMember("data_type", "INT64", allocator);
      const int64_t* data = tensor.GetTensorData<int64_t>();
      for (size_t i = 0; i < tensor_count; ++i) {
        tensor_content.PushBack(data[i], allocator);
      }
      break;
    }
    default:
      return absl::InternalError(
          absl::StrCat("Unsupported tensor data type: ", data_type));
  }

  json_tensor.AddMember("tensor_content", tensor_content, allocator);
  return json_tensor;
}

}  // namespace

absl::StatusOr<Ort::Value> ConvertFlatArrayToTensor(const Tensor& tensor) {
  switch (tensor.data_type) {
    case DataType::kFloat: {
      return ConvertFlatArrayToTensorInternal<float>(tensor);
    }
    case DataType::kDouble: {
      return ConvertFlatArrayToTensorInternal<double>(tensor);
    }
    case DataType::kInt8: {
      return ConvertFlatArrayToTensorInternal<int8_t>(tensor);
    }
    case DataType::kInt16: {
      return ConvertFlatArrayToTensorInternal<int16_t>(tensor);
    }
    case DataType::kInt32: {
      return ConvertFlatArrayToTensorInternal<int>(tensor);
    }
    case DataType::kInt64: {
      return ConvertFlatArrayToTensorInternal<int64_t>(tensor);
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported data type %d", tensor.data_type));
  }
}

absl::StatusOr<std::string> ConvertBatchOutputsToJson(
    const std::vector<TensorsOrError>& batch_outputs) {
  rapidjson::Document document;
  document.SetObject();
  rapidjson::MemoryPoolAllocator<>& allocator = document.GetAllocator();

  rapidjson::Value batch(rapidjson::kArrayType);
  for (const auto& output : batch_outputs) {
    rapidjson::Value nested_object(rapidjson::kObjectType);
    nested_object.AddMember("model_path",
                            rapidjson::Value().SetString(
                                rapidjson::StringRef(output.model_path.data())),
                            allocator);

    if (output.tensors) {
      rapidjson::Value tensors_value(rapidjson::kArrayType);
      for (const auto& [tensor_name, tensor] : output.tensors.value()) {
        absl::StatusOr<rapidjson::Value> json =
            TensorToJsonValue(tensor_name, tensor, allocator);
        if (json.ok()) {
          tensors_value.PushBack(json.value(), allocator);
        } else {
          return json.status();
        }
      }
      nested_object.AddMember("tensors", tensors_value.Move(), allocator);
    } else if (output.error) {
      nested_object.AddMember(
          "error", CreateSingleError(allocator, *output.error), allocator);
    }
    batch.PushBack(nested_object.Move(), allocator);
  }

  document.AddMember("response", batch.Move(), allocator);
  return SerializeJsonDoc(document);
}

}  // namespace privacy_sandbox::bidding_auction_servers::inference
