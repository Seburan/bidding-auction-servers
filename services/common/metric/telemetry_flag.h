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

#ifndef SERVICES_COMMON_METRIC_FLAG_H_
#define SERVICES_COMMON_METRIC_FLAG_H_

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "services/common/metric/config.pb.h"

namespace privacy_sandbox::server_common::metric {

struct TelemetryFlag {
  ServerConfig server_config;
};

bool AbslParseFlag(absl::string_view text, TelemetryFlag* flag,
                   std::string* err);

std::string AbslUnparseFlag(const TelemetryFlag&);

class BuildDependentConfig {
 public:
  explicit BuildDependentConfig(ServerConfig config)
      : server_config_(std::move(config)) {}

  enum class BuildMode { kProd, kExperiment };

  // Get build mode. The implementation depend on build flag.
  BuildMode GetBuildMode() const;

  // Get the metric mode to use
  ServerConfig::TelemetryMode MetricMode() const {
    if (GetBuildMode() == BuildMode::kExperiment) {
      return server_config_.mode();
    } else {
      return server_config_.mode() == ServerConfig::OFF ? ServerConfig::OFF
                                                        : ServerConfig::PROD;
    }
  }

  // Should metric be collected and exported;
  // Used to initialize Open Telemetry.
  bool MetricAllowed() const {
    switch (server_config_.mode()) {
      case ServerConfig::PROD:
      case ServerConfig::EXPERIMENT:
      case ServerConfig::COMPARE:
        return true;
      default:
        return false;
    }
  }

  // Should trace be collected and exported;
  // Used to initialize Open Telemetry.
  bool TraceAllowed() const { return IsDebug(); }

  // Should metric collection run as debug mode(without nosie)
  bool IsDebug() const {
    switch (server_config_.mode()) {
      case ServerConfig::EXPERIMENT:
      case ServerConfig::COMPARE:
        return GetBuildMode() == BuildMode::kExperiment;
      default:
        return false;
    }
  }

 private:
  ServerConfig server_config_;
};
}  // namespace privacy_sandbox::server_common::metric

#endif  // SERVICES_COMMON_METRIC_FLAG_H_
