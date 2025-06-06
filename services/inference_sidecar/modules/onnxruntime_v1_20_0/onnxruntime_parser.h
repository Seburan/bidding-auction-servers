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

#ifndef SERVICES_INFERENCE_SIDECAR_MODULES_ONNXRUNTIME_V1_20_0_ONNXRUNTIME_PARSER_H_
#define SERVICES_INFERENCE_SIDECAR_MODULES_ONNXRUNTIME_V1_20_0_ONNXRUNTIME_PARSER_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "utils/error.h"
#include "utils/request_parser.h"

#include "onnxruntime_cxx_api.h"

namespace privacy_sandbox::bidding_auction_servers::inference {

// Not copy constructible because Ort::Value is not.
struct TensorWithName {
  std::string tensor_name;
  Ort::Value tensor;

  TensorWithName(TensorWithName&&) noexcept = default;
  TensorWithName& operator=(TensorWithName&&) noexcept = default;

  TensorWithName(const TensorWithName&) = delete;
  TensorWithName& operator=(const TensorWithName&) = delete;
};

// Holds either a vector of tensors or an error for inference result.
struct TensorsOrError {
  std::string model_path;
  std::optional<std::vector<TensorWithName>> tensors;
  std::optional<Error> error;

  TensorsOrError() = default;
  TensorsOrError(TensorsOrError&&) noexcept = default;
  TensorsOrError& operator=(TensorsOrError&&) noexcept = default;

  TensorsOrError(const TensorsOrError&) = delete;
  TensorsOrError& operator=(const TensorsOrError&) = delete;
};

// Transforms internal generic dense tensor representation (in the format of
// one-dimensional array) into a Ort tensor and the desired tensor shape.
absl::StatusOr<Ort::Value> ConvertFlatArrayToTensor(const Tensor& tensor);

// Converts inference output corresponding to each model to a JSON string.
absl::StatusOr<std::string> ConvertBatchOutputsToJson(
    const std::vector<TensorsOrError>& batch_outputs);

}  // namespace privacy_sandbox::bidding_auction_servers::inference

#endif  // SERVICES_INFERENCE_SIDECAR_MODULES_ONNXRUNTIME_V1_20_0_ONNXRUNTIME_PARSER_H_
