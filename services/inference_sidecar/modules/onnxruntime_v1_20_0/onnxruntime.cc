/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <future>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "model/model_store.h"
#include "modules/module_interface.h"
#include "proto/inference_sidecar.pb.h"
#include "utils/error.h"
#include "utils/inference_error_code.h"
#include "utils/inference_metric_util.h"
#include "utils/log.h"
#include "utils/request_parser.h"

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_parser.h"

namespace privacy_sandbox::bidding_auction_servers::inference {
namespace {

// Creates the static Env using the threading options.
// This function returns a singleton instance of `Ort::Env` which is only
// initialized during the first call. Subsequent calls of this function returns
// the pre-initialized instance.
Ort::Env& CreateOrGetOrtEnv(const InferenceSidecarRuntimeConfig& config) {
  OrtThreadingOptions* threading_options = nullptr;

  OrtStatusPtr ort_status;
  ort_status = Ort::GetApi().CreateThreadingOptions(&threading_options);
  CHECK(ort_status == nullptr) << Ort::GetApi().GetErrorMessage(ort_status);
  ort_status = Ort::GetApi().SetGlobalIntraOpNumThreads(
      threading_options, config.num_intraop_threads());
  CHECK(ort_status == nullptr) << Ort::GetApi().GetErrorMessage(ort_status);
  ort_status = Ort::GetApi().SetGlobalInterOpNumThreads(
      threading_options, config.num_interop_threads());
  CHECK(ort_status == nullptr) << Ort::GetApi().GetErrorMessage(ort_status);
  // Disable spinning as the B&A inference workflow has high CPU utilization.
  ort_status = Ort::GetApi().SetGlobalSpinControl(threading_options, 0);
  CHECK(ort_status == nullptr) << Ort::GetApi().GetErrorMessage(ort_status);

  // Creates an environment with a global thread pool.
  static Ort::Env* env =
      new Ort::Env(threading_options, ORT_LOGGING_LEVEL_FATAL, "Default");
  Ort::GetApi().ReleaseThreadingOptions(threading_options);
  return *env;
}

class OnnxModule final : public ModuleInterface {
 public:
  explicit OnnxModule(const InferenceSidecarRuntimeConfig& config);

  absl::StatusOr<PredictResponse> Predict(
      const PredictRequest& request,
      const RequestContext& request_context) override;
  absl::StatusOr<RegisterModelResponse> RegisterModel(
      const RegisterModelRequest& request) override;
  absl::StatusOr<DeleteModelResponse> DeleteModel(
      const DeleteModelRequest& request) override;

 private:
  const InferenceSidecarRuntimeConfig runtime_config_;
  std::unique_ptr<ModelStore<Ort::Session>> store_;
};

absl::StatusOr<std::vector<TensorWithName>> PredictPerModel(
    std::shared_ptr<Ort::Session> model,
    const InferenceRequest& inference_request) {
  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_tensors;
  for (const auto& tensor : inference_request.inputs) {
    if (tensor.tensor_name.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          kInferenceTensorInputNameError,
          ". Message: ", "Name is required for each Onnxruntime tensor input"));
    }
    input_names.push_back(tensor.tensor_name.c_str());

    absl::StatusOr<Ort::Value> ort_tensor = ConvertFlatArrayToTensor(tensor);
    if (!ort_tensor.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(kInferenceInputTensorConversionError,
                       ". Message: ", ort_tensor.status().message()));
    }
    input_tensors.push_back(*std::move(ort_tensor));
  }

  try {
    Ort::AllocatorWithDefaultOptions alloc;
    size_t output_size = model->GetOutputCount();

    // The variable `output_names_str` is used to store the content of the
    // unique pointers of output names returned by `GetOutputNameAllocated`.
    // These unique pointers go out of the scope outside the loop.
    std::vector<std::string> output_names_str(output_size);
    for (size_t i = 0; i < output_size; ++i) {
      output_names_str[i] = model->GetOutputNameAllocated(i, alloc).get();
    }

    std::vector<const char*> output_names(output_size);
    for (size_t i = 0; i < output_size; ++i) {
      output_names[i] = output_names_str[i].c_str();
    }

    std::vector<Ort::Value> outputs = model->Run(
        Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
        input_names.size(), output_names.data(), output_size);

    std::vector<TensorWithName> zipped_vector;
    for (size_t i = 0; i < output_size; ++i) {
      zipped_vector.push_back(
          {.tensor_name = output_names[i], .tensor = std::move(outputs[i])});
    }

    return zipped_vector;
  } catch (const Ort::Exception& e) {
    return absl::InternalError(absl::StrCat(
        kInferenceModelExecutionError,
        "Exception thrown during Onnxruntime model inference: ", e.what()));
  } catch (...) {
    return absl::InternalError(
        "Unknown exception occurred during Onnxruntime model inference");
  }
}

absl::StatusOr<std::shared_ptr<Ort::Session>> OrtModelConstructor(
    const InferenceSidecarRuntimeConfig& config,
    const RegisterModelRequest& request,
    ModelConstructMetrics& construct_metrics) {
  std::shared_ptr<Ort::Session> model;
  try {
    Ort::SessionOptions session_options;
    // To enable a process level thread pool, we need to disable the session
    // level thread pool.
    session_options.DisablePerSessionThreads();
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    const std::string& model_payload = request.model_files().begin()->second;
    const char* model_data = model_payload.data();
    size_t model_size = model_payload.size();
    model = std::make_shared<Ort::Session>(
        CreateOrGetOrtEnv(config), model_data, model_size, session_options);
  } catch (const Ort::Exception& e) {
    return absl::InternalError(absl::StrCat(
        "Exception thrown during Onnxruntime model loading: ", e.what()));
  } catch (...) {
    return absl::InternalError(
        "Unknown exception occurred during Onnxruntime model loading");
  }

  // perform warm up if metadata been provided.
  // TODO(b/362338463): Add optional execute mode choice.
  if (!request.warm_up_batch_request_json().empty()) {
    absl::StatusOr<std::vector<ParsedRequestOrError>> parsed_requests =
        ParseJsonInferenceRequest(request.warm_up_batch_request_json());
    if (!parsed_requests.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Encounters warm up batch inference request parsing error: ",
          parsed_requests.status().message()));
    }
    size_t parsed_request_size = 0;
    auto start_pre_warm_time = absl::Now();
    // Process warm up for each inference request.
    for (const ParsedRequestOrError& parsed_request : (*parsed_requests)) {
      if (parsed_request.request) {
        InferenceRequest inference_request = parsed_request.request.value();
        parsed_request_size += 1;
        if (inference_request.model_path != request.model_spec().model_path()) {
          return absl::InvalidArgumentError(
              "Warm up request using different model path.");
        }
        auto inference_response = PredictPerModel(model, inference_request);
        if (!inference_response.ok()) {
          return inference_response.status();
        }
      } else {
        // TODO(b/384551230): parsing error handling when partial failure
        ABSL_LOG(ERROR)
            << "Encounters warm up batch inference request parsing error: "
            << "Model: "
            << absl::StrCat(parsed_request.error.value().model_path)
            << " Description: "
            << absl::StrCat(parsed_request.error.value().description);
      }
    }
    if (parsed_request_size == 0) {
      return absl::InvalidArgumentError(
          "Encounters warm up batch inference request parsing error: All "
          "requests can not be parsed");
    }
    // Set warm up latency metric
    construct_metrics.set_model_pre_warm_latency(
        ToDoubleMicroseconds((absl::Now() - start_pre_warm_time)));
  }
  return model;
}

OnnxModule::OnnxModule(const InferenceSidecarRuntimeConfig& config)
    : runtime_config_(config),
      store_(std::make_unique<ModelStore<Ort::Session>>(config,
                                                        OrtModelConstructor)) {
  CreateOrGetOrtEnv(config);
}

absl::StatusOr<RegisterModelResponse> OnnxModule::RegisterModel(
    const RegisterModelRequest& request) {
  absl::string_view model_key = request.model_spec().model_path();
  if (model_key.empty()) {
    return absl::InvalidArgumentError("Empty model path during registration");
  }
  if (request.model_files().size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The number of model files should be exactly 1, but got ",
                     request.model_files().size()));
  }

  if (store_->GetModel(model_key).ok()) {
    return absl::AlreadyExistsError(
        absl::StrCat("Model ", model_key, " has already been registered"));
  }
  // Set collector for metric during model construct.
  ModelConstructMetrics model_construct_metrics;
  PS_RETURN_IF_ERROR(
      store_->PutModel(model_key, request, model_construct_metrics));

  RegisterModelResponse register_model_response;
  if (!request.warm_up_batch_request_json().empty()) {
    AddMetric(register_model_response,
              "kInferenceRegisterModelResponseModelWarmUpDuration",
              model_construct_metrics.model_pre_warm_latency());
  }
  return register_model_response;
}

absl::StatusOr<DeleteModelResponse> OnnxModule::DeleteModel(
    const DeleteModelRequest& request) {
  PS_RETURN_IF_ERROR(store_->DeleteModel(request.model_spec().model_path()));
  return DeleteModelResponse();
}

absl::StatusOr<PredictResponse> OnnxModule::Predict(
    const PredictRequest& request, const RequestContext& request_context) {
  PredictResponse predict_response;
  absl::Time start_inference_execution_time = absl::Now();
  AddMetric(predict_response, "kInferenceRequestSize", request.ByteSizeLong());
  absl::StatusOr<std::vector<ParsedRequestOrError>> parsed_requests =
      ParseJsonInferenceRequest(request.input());
  if (!parsed_requests.ok()) {
    AddMetric(predict_response, "kInferenceErrorCountByErrorCode", 1,
              std::string(kInferenceUnableToParseRequest));
    INFERENCE_LOG(ERROR, request_context) << parsed_requests.status();
    predict_response.set_output(CreateBatchErrorString(
        Error{.error_type = Error::INPUT_PARSING,
              .description = std::string(parsed_requests.status().message())}));
    return predict_response;
  }

  AddMetric(predict_response, "kInferenceRequestCount", 1);

  std::vector<TensorsOrError> batch_outputs(parsed_requests->size());
  std::vector<std::future<absl::StatusOr<std::vector<TensorWithName>>>> tasks(
      parsed_requests->size());
  for (size_t task_id = 0; task_id < parsed_requests->size(); ++task_id) {
    if ((*parsed_requests)[task_id].request) {
      const InferenceRequest& inference_request =
          (*parsed_requests)[task_id].request.value();
      const std::string& model_path = inference_request.model_path;
      INFERENCE_LOG(INFO, request_context)
          << "Received inference request to model: " << model_path;
      absl::StatusOr<std::shared_ptr<Ort::Session>> model =
          store_->GetModel(model_path, request.is_consented());
      if (!model.ok()) {
        AddMetric(predict_response, "kInferenceErrorCountByErrorCode", 1,
                  std::string(kInferenceModelNotFoundError));
        INFERENCE_LOG(ERROR, request_context)
            << "Fails to get model: " << model_path
            << " Reason: " << model.status();
        batch_outputs[task_id] = TensorsOrError{
            .model_path = model_path,
            .error =
                Error{.error_type = Error::MODEL_NOT_FOUND,
                      .description = std::string(model.status().message())}};
      } else {
        // Only log count by model for available models since there is no metric
        // partition for unregistered models.
        AddMetric(predict_response, "kInferenceRequestCountByModel", 1,
                  model_path);
        int batch_count = inference_request.inputs[0].tensor_shape[0];
        AddMetric(predict_response, "kInferenceRequestBatchCountByModel",
                  batch_count, model_path);
        tasks[task_id] = std::async(std::launch::async, &PredictPerModel,
                                    *model, inference_request);
      }
    } else {
      const Error error = (*parsed_requests)[task_id].error.value();
      batch_outputs[task_id] =
          TensorsOrError{.model_path = error.model_path, .error = error};
    }
  }

  for (size_t task_id = 0; task_id < parsed_requests->size(); ++task_id) {
    if (!batch_outputs[task_id].error) {
      absl::StatusOr<std::vector<TensorWithName>> tensors =
          tasks[task_id].get();
      const std::string& model_path =
          (*parsed_requests)[task_id].request.value().model_path;

      if (!tensors.ok()) {
        AddMetric(predict_response, "kInferenceErrorCountByErrorCode", 1,
                  std::optional(
                      ExtractErrorCodeFromMessage(tensors.status().message())));
        AddMetric(predict_response, "kInferenceRequestFailedCountByModel", 1,
                  model_path);
        INFERENCE_LOG(ERROR, request_context)
            << "Inference fails for model: " << model_path
            << " Reason: " << tensors.status();
        Error::ErrorType error_type =
            tensors.status().code() == absl::StatusCode::kInvalidArgument
                ? Error::INPUT_PARSING
                : Error::MODEL_EXECUTION;

        batch_outputs[task_id] = TensorsOrError{
            .model_path = model_path,
            .error =
                Error{.error_type = error_type,
                      .description = std::string(tensors.status().message())}};
      } else {
        int model_execution_time_ms =
            (absl::Now() - start_inference_execution_time) /
            absl::Milliseconds(1);
        AddMetric(predict_response, "kInferenceRequestDurationByModel",
                  model_execution_time_ms, model_path);
        batch_outputs[task_id] = TensorsOrError{.model_path = model_path,
                                                .tensors = *std::move(tensors)};
      }
    }
  }

  auto output_json = ConvertBatchOutputsToJson(batch_outputs);
  if (!output_json.ok()) {
    AddMetric(predict_response, "kInferenceErrorCountByErrorCode", 1,
              std::string(kInferenceOutputParsingError));

    INFERENCE_LOG(ERROR, request_context) << output_json.status();
    predict_response.set_output(CreateBatchErrorString(
        Error{.error_type = Error::OUTPUT_PARSING,
              .description = "Error during output parsing to json."}));
    return predict_response;
  }
  for (const ParsedRequestOrError& parsed_request : *parsed_requests) {
    if (parsed_request.request) {
      store_->IncrementModelInferenceCount(
          parsed_request.request.value().model_path);
    }
  }

  predict_response.set_output(output_json.value());
  AddMetric(predict_response, "kInferenceResponseSize",
            predict_response.ByteSizeLong());
  int inference_execution_time_ms =
      (absl::Now() - start_inference_execution_time) / absl::Milliseconds(1);
  AddMetric(predict_response, "kInferenceRequestDuration",
            inference_execution_time_ms);

  return predict_response;
}

}  // namespace

std::unique_ptr<ModuleInterface> ModuleInterface::Create(
    const InferenceSidecarRuntimeConfig& config) {
  return std::make_unique<OnnxModule>(config);
}

absl::string_view ModuleInterface::GetModuleVersion() {
  return "onnxruntime_v1_20_0";
}

}  // namespace privacy_sandbox::bidding_auction_servers::inference
