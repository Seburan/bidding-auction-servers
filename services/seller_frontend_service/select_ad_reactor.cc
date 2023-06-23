//  Copyright 2022 Google LLC
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

#include "services/seller_frontend_service/select_ad_reactor.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/functional/bind_front.h"
#include "absl/numeric/bits.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "api/bidding_auction_servers.grpc.pb.h"
#include "api/bidding_auction_servers.pb.h"
#include "glog/logging.h"
#include "quiche/oblivious_http/oblivious_http_gateway.h"
#include "services/common/compression/gzip.h"
#include "services/common/constants/user_error_strings.h"
#include "services/common/reporters/async_reporter.h"
#include "services/common/util/debug_reporting_util.h"
#include "services/common/util/request_response_constants.h"
#include "services/seller_frontend_service/util/web_utils.h"
#include "src/cpp/communication/ohttp_utils.h"
#include "src/cpp/encryption/key_fetcher/src/key_fetcher_manager.h"
#include "src/cpp/telemetry/telemetry.h"

namespace privacy_sandbox::bidding_auction_servers {
namespace {

using ScoreAdsRawRequest = ScoreAdsRequest::ScoreAdsRawRequest;
using AdScore = ScoreAdsResponse::AdScore;
using AdWithBidMetadata =
    ScoreAdsRequest::ScoreAdsRawRequest::AdWithBidMetadata;
using BiddingGroupMap =
    ::google::protobuf::Map<std::string, AuctionResult::InterestGroupIndex>;
using DecodedBuyerInputs = absl::flat_hash_map<absl::string_view, BuyerInput>;
using EncodedBuyerInputs = ::google::protobuf::Map<std::string, std::string>;
using ErrorVisibility::CLIENT_VISIBLE;

}  // namespace

SelectAdReactor::SelectAdReactor(grpc::CallbackServerContext* context,
                                 const SelectAdRequest* request,
                                 SelectAdResponse* response,
                                 const ClientRegistry& clients,
                                 const SellerFrontEndConfig& config,
                                 bool fail_fast)
    : context_(context),
      request_(request),
      protected_audience_input_(
          std::move(request->raw_protected_audience_input())),
      response_(response),
      clients_(clients),
      config_(config),
      // TODO(b/278039901): Add integration test for metadata forwarding.
      buyer_metadata_(GrpcMetadataToRequestMetadata(context->client_metadata(),
                                                    kBuyerMetadataKeysMap)),
      pending_bids_count_(request->auction_config().buyer_list_size()),
      error_accumulator_(&logger_),
      fail_fast_(fail_fast) {
  if (config_.enable_seller_frontend_benchmarking()) {
    benchmarking_logger_ =
        std::make_unique<BuildInputProcessResponseBenchmarkingLogger>(
            FormatTime(absl::Now()));
  } else {
    benchmarking_logger_ = std::make_unique<NoOpsLogger>();
  }
  CHECK_OK([this]() {
    PS_ASSIGN_OR_RETURN(metric_context_,
                        metric::SfeContextMap()->Remove(request_));
    return absl::OkStatus();
  }()) << "SfeContextMap()->Get(request) should have been called";
}

AdWithBidMetadata SelectAdReactor::BuildAdWithBidMetadata(
    const AdWithBid& input, absl::string_view interest_group_owner) {
  AdWithBidMetadata result;
  if (input.has_ad()) {
    *result.mutable_ad() = input.ad();
  }
  result.set_bid(input.bid());
  result.set_render(input.render());
  result.set_allow_component_auction(input.allow_component_auction());
  result.mutable_ad_component_render()->CopyFrom(input.ad_component_render());
  result.set_interest_group_name(input.interest_group_name());
  result.set_interest_group_owner(interest_group_owner);
  result.set_ad_cost(input.ad_cost());
  result.set_modeling_signals(input.modeling_signals());
  const BuyerInput& buyer_input =
      buyer_inputs_->find(interest_group_owner)->second;
  for (const auto& interest_group : buyer_input.interest_groups()) {
    if (std::strcmp(interest_group.name().c_str(),
                    result.interest_group_name().c_str())) {
      if (request_->client_type() == SelectAdRequest::BROWSER) {
        result.set_join_count(interest_group.browser_signals().join_count());
        result.set_recency(interest_group.browser_signals().recency());
      }
      break;
    }
  }
  return result;
}

bool SelectAdReactor::HaveClientVisibleErrors() {
  return !error_accumulator_.GetErrors(ErrorVisibility::CLIENT_VISIBLE).empty();
}

bool SelectAdReactor::HaveAdServerVisibleErrors() {
  return !error_accumulator_.GetErrors(ErrorVisibility::AD_SERVER_VISIBLE)
              .empty();
}

void SelectAdReactor::MayPopulateClientVisibleErrors() {
  const ErrorAccumulator::ErrorMap& error_map =
      error_accumulator_.GetErrors(ErrorVisibility::CLIENT_VISIBLE);
  if (!HaveClientVisibleErrors()) {
    return;
  }

  auto error = response_->mutable_raw_response()->mutable_error();
  error->set_code(static_cast<int>(ErrorCode::CLIENT_SIDE));
  error->set_message(
      GetAccumulatedErrorString(ErrorVisibility::CLIENT_VISIBLE));
}

void SelectAdReactor::ValidateProtectedAudienceInput(
    const ProtectedAudienceInput& protected_audience_input) {
  if (protected_audience_input.generation_id().empty()) {
    ReportError(CLIENT_VISIBLE, kMissingGenerationId, ErrorCode::CLIENT_SIDE);
  }

  if (protected_audience_input.publisher_name().empty()) {
    ReportError(CLIENT_VISIBLE, kMissingPublisherName, ErrorCode::CLIENT_SIDE);
  }

  // Validate Buyer Inputs.
  if (buyer_inputs_->empty()) {
    ReportError(CLIENT_VISIBLE, kMissingBuyerInputs, ErrorCode::CLIENT_SIDE);
  } else {
    bool is_any_buyer_input_valid = false;
    std::set<std::string> observed_errors;
    for (const auto& [buyer, buyer_input] : *buyer_inputs_) {
      bool any_error = false;
      if (buyer.empty()) {
        observed_errors.insert(kEmptyInterestGroupOwner);
        any_error = true;
      }
      if (buyer_input.interest_groups().empty()) {
        observed_errors.insert(absl::StrFormat(kMissingInterestGroups, buyer));
        any_error = true;
      }
      if (any_error) {
        continue;
      }
      is_any_buyer_input_valid = true;
    }
    // Buyer inputs have keys but none of the key/value pairs are usable to get
    // bids from buyers.
    if (!is_any_buyer_input_valid) {
      std::string error =
          absl::StrFormat(kNonEmptyBuyerInputMalformed,
                          absl::StrJoin(observed_errors, kErrorDelimiter));
      ReportError(CLIENT_VISIBLE, error, ErrorCode::CLIENT_SIDE);
    } else {
      // Log but don't report the errors for malformed buyer inputs because we
      // have found at least one buyer input that is well formed.
      for (const auto& observed_error : observed_errors) {
        logger_.vlog(2, observed_error);
      }
    }
  }
}

bool SelectAdReactor::DecryptRequest() {
  if (request_->protected_audience_ciphertext().empty()) {
    Finish(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        kEmptyRemarketingCiphertextError));
    return false;
  }

  absl::string_view encapsulated_req =
      request_->protected_audience_ciphertext();
  logger_.vlog(5, "Protected audience ciphertext: ", encapsulated_req);

  // Parse the encapsulated request for the key ID.
  absl::StatusOr<uint8_t> key_id = server_common::ParseKeyId(encapsulated_req);
  if (!key_id.ok()) {
    logger_.vlog(2, "Parsed key id error status: ", key_id.status().message());
    Finish(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        kInvalidOhttpKeyIdError));
    return false;
  }

  std::string str_key_id = std::to_string(*key_id);
  std::optional<server_common::PrivateKey> private_key =
      clients_.key_fetcher_manager_->GetPrivateKey(str_key_id);

  if (!private_key.has_value()) {
    logger_.vlog(2, "Unable to retrieve private key for key ID: ", str_key_id);
    Finish(
        grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, kMissingPrivateKey));
    return false;
  }

  // Decrypt the ciphertext.
  absl::StatusOr<quiche::ObliviousHttpRequest> ohttp_request =
      server_common::DecryptEncapsulatedRequest(*private_key, encapsulated_req);
  if (!ohttp_request.ok()) {
    logger_.vlog(2, "Unable to decrypt the ciphertext. Reason: ",
                 ohttp_request.status().message());
    Finish(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        absl::StrFormat(kMalformedEncapsulatedRequest,
                                        ohttp_request.status().message())));
    return false;
  }

  logger_.vlog(5,
               "Successfully decrypted the protected audience input "
               "ciphertext");
  quiche::ObliviousHttpRequest::Context ohttp_context =
      std::move(ohttp_request.value()).ReleaseContext();
  RequestContext request = {
      str_key_id, std::make_unique<quiche::ObliviousHttpRequest::Context>(
                      std::move(ohttp_context))};
  request_context_ = std::move(request);
  protected_audience_input_ =
      GetDecodedProtectedAudienceInput(ohttp_request->GetPlaintextData());
  buyer_inputs_ =
      GetDecodedBuyerinputs(protected_audience_input_.buyer_input());
  return true;
}

ContextLogger::ContextMap SelectAdReactor::GetLoggingContext() {
  return {{kGenerationId, protected_audience_input_.generation_id()},
          {kSellerDebugId, request_->auction_config().seller_debug_id()}};
}

void SelectAdReactor::MayPopulateAdServerVisibleErrors() {
  if (request_->auction_config().seller_signals().empty()) {
    ReportError(ErrorVisibility::AD_SERVER_VISIBLE, kEmptySellerSignals,
                ErrorCode::CLIENT_SIDE);
  }

  if (request_->auction_config().auction_signals().empty()) {
    ReportError(ErrorVisibility::AD_SERVER_VISIBLE, kEmptyAuctionSignals,
                ErrorCode::CLIENT_SIDE);
  }

  if (request_->auction_config().buyer_list().empty()) {
    ReportError(ErrorVisibility::AD_SERVER_VISIBLE, kEmptyBuyerList,
                ErrorCode::CLIENT_SIDE);
  }

  if (request_->auction_config().seller().empty()) {
    ReportError(ErrorVisibility::AD_SERVER_VISIBLE, kEmptySeller,
                ErrorCode::CLIENT_SIDE);
  }

  if (config_.seller_origin_domain() != request_->auction_config().seller()) {
    ReportError(ErrorVisibility::AD_SERVER_VISIBLE, kWrongSellerDomain,
                ErrorCode::CLIENT_SIDE);
  }

  for (const auto& [buyer, per_buyer_config] :
       request_->auction_config().per_buyer_config()) {
    if (buyer.empty()) {
      ReportError(ErrorVisibility::AD_SERVER_VISIBLE,
                  kEmptyBuyerInPerBuyerConfig, ErrorCode::CLIENT_SIDE);
    }
    if (per_buyer_config.buyer_signals().empty()) {
      ReportError(ErrorVisibility::AD_SERVER_VISIBLE,
                  absl::StrFormat(kEmptyBuyerSignals, buyer),
                  ErrorCode::CLIENT_SIDE);
    }
  }

  if (request_->client_type() == SelectAdRequest::UNKNOWN) {
    ReportError(ErrorVisibility::AD_SERVER_VISIBLE, kUnknownClientType,
                ErrorCode::CLIENT_SIDE);
  }
}

void SelectAdReactor::Execute() {
  if (!config_.enable_encryption()) {
    // Ensure we have buyer inputs in a form that we can work with.
    buyer_inputs_ =
        GetDecodedBuyerinputs(protected_audience_input_.buyer_input());
    if (!buyer_inputs_.ok()) {
      ReportError(ErrorVisibility::CLIENT_VISIBLE, kMalformedBuyerInput,
                  ErrorCode::CLIENT_SIDE);
    }
  } else if (!DecryptRequest()) {
    return;
  }

  logger_.Configure(GetLoggingContext());
  MayPopulateAdServerVisibleErrors();
  if (HaveAdServerVisibleErrors()) {
    // Finish the GRPC request if we have received bad data from the ad tech
    // server.
    OnScoreAdsDone(std::make_unique<ScoreAdsResponse>());
    return;
  }

  // Validate mandatory fields if decoding went through fine.
  if (!HaveClientVisibleErrors()) {
    ValidateProtectedAudienceInput(protected_audience_input_);
  }

  // Populate errors on the response immediately after decoding and input
  // validation so that when we stop processing the request due to errors, we
  // have correct errors set in the response.
  MayPopulateClientVisibleErrors();

  if (error_accumulator_.HasErrors()) {
    // Finish the GRPC request now.
    OnScoreAdsDone(std::make_unique<ScoreAdsResponse>());
    return;
  }

  auto scope = opentelemetry::trace::Scope(
      server_common::GetTracer()->StartSpan("SelectAdReactor_Execute"));
  benchmarking_logger_->Begin();

  for (const std::string& buyer_ig_owner :
       request_->auction_config().buyer_list()) {
    const auto& buyer_input_iterator = buyer_inputs_->find(buyer_ig_owner);
    if (buyer_input_iterator != buyer_inputs_->end()) {
      FetchBid(buyer_ig_owner, buyer_input_iterator->second,
               request_->auction_config().seller());
    } else {
      logger_.vlog(2, "No buyer input found for buyer: ", buyer_ig_owner);

      // Pending bids count is set on reactor construction to buyer_list_size().
      // If no BuyerInput is found for a buyer in buyer_list, must decrement
      // pending bids count.
      UpdatePendingBidsState();
    }
  }
}

void SelectAdReactor::FetchBid(const std::string& buyer_ig_owner,
                               const BuyerInput& buyer_input,
                               absl::string_view seller) {
  auto scope = opentelemetry::trace::Scope(
      server_common::GetTracer()->StartSpan("FetchBid"));
  auto buyer_client = clients_.buyer_factory.Get(buyer_ig_owner);
  if (buyer_client == nullptr) {
    logger_.vlog(2, "No buyer client found for buyer: ", buyer_ig_owner);
    UpdatePendingBidsState();
  } else {
    auto get_bids_request = std::make_unique<GetBidsRequest>();
    get_bids_request->mutable_raw_request()->set_is_chaff(false);
    get_bids_request->mutable_raw_request()->set_publisher_name(
        protected_audience_input_.publisher_name());
    get_bids_request->mutable_raw_request()->set_seller(seller);
    get_bids_request->mutable_raw_request()->set_auction_signals(
        request_->auction_config().auction_signals());
    absl::Duration timeout =
        absl::Milliseconds(config_.get_bid_rpc_timeout_ms());
    if (request_->auction_config().buyer_timeout_ms() > 0) {
      timeout =
          absl::Milliseconds(request_->auction_config().buyer_timeout_ms());
    }
    std::string buyer_debug_id;
    const auto& per_buyer_config_itr =
        request_->auction_config().per_buyer_config().find(buyer_ig_owner);
    if (per_buyer_config_itr !=
        request_->auction_config().per_buyer_config().end()) {
      buyer_debug_id = per_buyer_config_itr->second.buyer_debug_id();
      if (!per_buyer_config_itr->second.buyer_signals().empty()) {
        get_bids_request->mutable_raw_request()->set_buyer_signals(
            per_buyer_config_itr->second.buyer_signals());
      }
    }
    *get_bids_request->mutable_raw_request()->mutable_buyer_input() =
        buyer_input;
    get_bids_request->mutable_raw_request()->set_enable_debug_reporting(
        protected_audience_input_.enable_debug_reporting());
    auto* log_context =
        get_bids_request->mutable_raw_request()->mutable_log_context();
    log_context->set_generation_id(protected_audience_input_.generation_id());
    log_context->set_adtech_debug_id(buyer_debug_id);
    absl::Status execute_result = buyer_client->Execute(
        std::move(get_bids_request), this->buyer_metadata_,
        [buyer_ig_owner,
         this](absl::StatusOr<std::unique_ptr<GetBidsResponse>> response) {
          OnFetchBidsDone(std::move(response), buyer_ig_owner);
        },
        timeout);
    if (!execute_result.ok()) {
      logger_.error(
          absl::StrFormat("Failed to make async GetBids call: (buyer: %s, "
                          "seller: %s, error: "
                          "%s)",
                          buyer_ig_owner, seller, execute_result.ToString()));
      UpdatePendingBidsState();
    }
  }
}

void SelectAdReactor::OnFetchBidsDone(
    absl::StatusOr<std::unique_ptr<GetBidsResponse>> response,
    const std::string& buyer_ig_owner) {
  if (response.ok()) {
    logger_.vlog(2, "\nGetBidsResponse:\n", response.value()->DebugString());
    absl::MutexLock lock(&bid_data_mu_);
    if ((*response)->raw_response().bids().empty()) {
      logger_.vlog(2, "Skipping buyer ", buyer_ig_owner,
                   " due to empty GetBidsResponse.");
    } else {
      buyer_bids_.try_emplace(buyer_ig_owner, *std::move(response));
    }
  } else {
    logger_.vlog(1, "GetBidsRequest failed for buyer ", buyer_ig_owner,
                 "\nresponse status: ", response.status());
  }

  UpdatePendingBidsState();
}

void SelectAdReactor::UpdatePendingBidsState() {
  absl::MutexLock lock(&bid_data_mu_);
  pending_bids_count_--;

  if (pending_bids_count_ == 0) {
    for (auto& [buyer_ig_owner, response] : buyer_bids_) {
      buyer_bids_list_.emplace(buyer_ig_owner, std::move(response));
    }
    if (buyer_bids_list_.empty()) {
      logger_.vlog(2, kNoBidsReceived);
      OnScoreAdsDone(std::make_unique<ScoreAdsResponse>());
    } else {
      FetchScoringSignals();
    }
  }
}

void SelectAdReactor::FetchScoringSignals() {
  clients_.scoring_signals_async_provider.Get(
      buyer_bids_list_,
      [this](absl::StatusOr<std::unique_ptr<ScoringSignals>> result) {
        OnFetchScoringSignalsDone(std::move(result));
      },
      absl::Milliseconds(config_.key_value_signals_fetch_rpc_timeout_ms()));
}

void SelectAdReactor::OnFetchScoringSignalsDone(
    absl::StatusOr<std::unique_ptr<ScoringSignals>> result) {
  if (result.ok()) {
    scoring_signals_ = std::move(result.value());
  } else {
    // TODO(b/245982466): Handle early abort and errors.
    logger_.vlog(1, "Scoring signals fetch from key-value server failed: ",
                 result.status());
  }
  ScoreAds();
}

void SelectAdReactor::ScoreAds() {
  auto score_ads_request = std::make_unique<ScoreAdsRequest>();
  ScoreAdsRawRequest raw_request;
  for (const auto& [buyer, get_bid_response] : buyer_bids_list_) {
    for (int i = 0; i < get_bid_response->raw_response().bids_size(); i++) {
      AdWithBidMetadata ad_with_bid_metadata = BuildAdWithBidMetadata(
          get_bid_response->raw_response().bids().at(i), buyer);
      raw_request.mutable_ad_bids()->Add(std::move(ad_with_bid_metadata));
    }
  }
  *raw_request.mutable_auction_signals() =
      request_->auction_config().auction_signals();
  *raw_request.mutable_seller_signals() =
      request_->auction_config().seller_signals();
  if (scoring_signals_ != nullptr) {
    // Ad scoring signals cannot be used after this.
    raw_request.set_allocated_scoring_signals(
        scoring_signals_->scoring_signals.release());
  }
  raw_request.set_publisher_hostname(
      protected_audience_input_.publisher_name());
  raw_request.set_enable_debug_reporting(
      protected_audience_input_.enable_debug_reporting());

  auto* log_context = raw_request.mutable_log_context();
  log_context->set_generation_id(protected_audience_input_.generation_id());
  log_context->set_adtech_debug_id(
      request_->auction_config().seller_debug_id());
  *score_ads_request->mutable_raw_request() = raw_request;
  logger_.vlog(2, "\nScoreAdsRequest:\n", score_ads_request->DebugString());
  auto on_scoring_done =
      [this](absl::StatusOr<std::unique_ptr<ScoreAdsResponse>> result) {
        OnScoreAdsDone(std::move(result));
      };

  absl::Status execute_result = clients_.scoring.Execute(
      std::move(score_ads_request), {}, std::move(on_scoring_done),
      absl::Milliseconds(config_.score_ads_rpc_timeout_ms()));
  if (!execute_result.ok()) {
    logger_.error(
        absl::StrFormat("Failed to make async ScoreAds call: (error: %s)",
                        execute_result.ToString()));
    Finish(grpc::Status(grpc::INTERNAL, kInternalServerError));
  }
}

BiddingGroupMap SelectAdReactor::GetBiddingGroups() {
  BiddingGroupMap bidding_groups;
  for (const auto& [buyer, ad_with_bids] : buyer_bids_list_) {
    // Mapping from buyer to interest groups that are associated with non-zero
    // bids.
    absl::flat_hash_set<absl::string_view> buyer_interest_groups;
    for (const auto& ad_with_bid : ad_with_bids->raw_response().bids()) {
      if (ad_with_bid.bid() > 0) {
        buyer_interest_groups.insert(ad_with_bid.interest_group_name());
      }
    }
    const auto& buyer_input = buyer_inputs_->at(buyer);
    AuctionResult::InterestGroupIndex ar_interest_group_index;
    int ig_index = 0;
    for (const auto& interest_group : buyer_input.interest_groups()) {
      // If the interest group name is one of the groups returned by the bidding
      // service then record its index.
      if (buyer_interest_groups.contains(interest_group.name())) {
        ar_interest_group_index.add_index(ig_index);
      }
      ig_index++;
    }
    bidding_groups.try_emplace(buyer, std::move(ar_interest_group_index));
  }
  return bidding_groups;
}

void SelectAdReactor::FinishWithOkStatus() {
  benchmarking_logger_->End();
  Finish(grpc::Status::OK);
}

void SelectAdReactor::FinishWithInternalError(absl::string_view error) {
  logger_.error("RPC failed: ", error);
  benchmarking_logger_->End();
  Finish(grpc::Status(grpc::INTERNAL, ""));
}

void SelectAdReactor::PopulateRawResponse(
    const std::optional<AdScore>& high_score,
    BiddingGroupMap bidding_group_map) {
  if (!high_score.has_value()) {
    response_->mutable_raw_response()->set_is_chaff(true);
    return;
  }

  response_->mutable_raw_response()->set_is_chaff(false);
  response_->mutable_raw_response()->set_ad_render_url(high_score->render());
  response_->mutable_raw_response()->set_score(high_score->desirability());
  response_->mutable_raw_response()
      ->mutable_ad_component_render_urls()
      ->CopyFrom(high_score->component_renders());
  response_->mutable_raw_response()->set_interest_group_name(
      high_score->interest_group_name());
  response_->mutable_raw_response()->set_interest_group_owner(
      high_score->interest_group_owner());
  response_->mutable_raw_response()->set_bid(high_score->buyer_bid());
  *response_->mutable_raw_response()->mutable_bidding_groups() =
      std::move(bidding_group_map);
}

std::string SelectAdReactor::GetAccumulatedErrorString(
    ErrorVisibility error_visibility) {
  const ErrorAccumulator::ErrorMap& error_map =
      error_accumulator_.GetErrors(error_visibility);
  auto it = error_map.find(ErrorCode::CLIENT_SIDE);
  if (it == error_map.end()) {
    return "";
  }

  return absl::StrJoin(it->second, kErrorDelimiter);
}

void SelectAdReactor::OnScoreAdsDone(
    absl::StatusOr<std::unique_ptr<ScoreAdsResponse>> response) {
  if (HaveAdServerVisibleErrors()) {
    Finish(grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        GetAccumulatedErrorString(ErrorVisibility::AD_SERVER_VISIBLE)));
    return;
  }

  logger_.vlog(2, "ScoreAdsResponse status:", response.status());
  if (!response.ok()) {
    benchmarking_logger_->End();
    Finish(grpc::Status(static_cast<grpc::StatusCode>(response.status().code()),
                        std::string(response.status().message())));
    return;
  }

  std::optional<AdScore> high_score;
  if (response.value()->raw_response().has_ad_score() &&
      response.value()->raw_response().ad_score().buyer_bid() > 0) {
    high_score = response.value()->raw_response().ad_score();
  }
  BiddingGroupMap bidding_group_map = GetBiddingGroups();
  PerformDebugReporting(high_score);

  if (!config_.enable_encryption()) {
    PopulateRawResponse(high_score, std::move(bidding_group_map));
    FinishWithOkStatus();
    return;
  }

  std::optional<AuctionResult::Error> error;
  if (HaveClientVisibleErrors()) {
    error = response_->raw_response().error();
  }
  absl::StatusOr<std::string> non_encrypted_response = GetNonEncryptedResponse(
      high_score, std::move(bidding_group_map), std::move(error));
  if (!non_encrypted_response.ok()) {
    return;
  }

  std::string plaintext_response = std::move(*non_encrypted_response);
  if (!EncryptResponse(std::move(plaintext_response))) {
    return;
  }

  logger_.vlog(2, "\nSelectAdResponse:\n", response_->DebugString());
  FinishWithOkStatus();
}

absl::StatusOr<std::string> SelectAdReactor::GetNonEncryptedResponse(
    const std::optional<ScoreAdsResponse::AdScore>& high_score,
    const BiddingGroupMap& bidding_group_map,
    const std::optional<AuctionResult::Error>& error) {
  absl::StatusOr<std::vector<unsigned char>> encoded_data =
      Encode(high_score, bidding_group_map, error,
             absl::bind_front(&SelectAdReactor::FinishWithInternalError, this));

  absl::string_view data_to_compress = absl::string_view(
      reinterpret_cast<char*>(encoded_data->data()), encoded_data->size());

  absl::StatusOr<std::string> compressed_data = GzipCompress(data_to_compress);
  if (!compressed_data.ok()) {
    logger_.error("Failed to compress the CBOR serialized data: ",
                  compressed_data.status().message());
    FinishWithInternalError("Failed to compress CBOR data");
    return absl::InternalError("");
  }

  // Pad data so that its size becomes the next smallest power of 2.
  compressed_data->resize(std::max(absl::bit_ceil(compressed_data->size()),
                                   kMinAuctionResultBytes));
  return std::move(*compressed_data);
}

DecodedBuyerInputs SelectAdReactor::GetDecodedBuyerinputs(
    const EncodedBuyerInputs& encoded_buyer_inputs) {
  return DecodeBuyerInputs(encoded_buyer_inputs, error_accumulator_,
                           fail_fast_);
}

ProtectedAudienceInput SelectAdReactor::GetDecodedProtectedAudienceInput(
    absl::string_view encoded_data) {
  return ProtectedAudienceInput{};
}

bool SelectAdReactor::EncryptResponse(std::string plaintext_response) {
  std::optional<server_common::PrivateKey> private_key =
      clients_.key_fetcher_manager_->GetPrivateKey(request_context_.key_id);
  if (!private_key.has_value()) {
    logger_.vlog(
        4,
        absl::StrFormat(
            "Encryption key not found during response encryption: (key ID: %s)",
            request_context_.key_id));
    Finish(grpc::Status(grpc::StatusCode::INTERNAL, ""));
    return false;
  }

  absl::StatusOr<std::string> encapsulated_response =
      server_common::EncryptAndEncapsulateResponse(
          std::move(plaintext_response), private_key.value(),
          *request_context_.context);
  if (!encapsulated_response.ok()) {
    logger_.vlog(
        4, absl::StrFormat("Error during response encryption/encapsulation: %s",
                           encapsulated_response.status().message()));
    Finish(grpc::Status(grpc::StatusCode::INTERNAL, ""));
    return false;
  }

  response_->mutable_auction_result_ciphertext()->assign(
      std::move(*encapsulated_response));
  return true;
}

void SelectAdReactor::PerformDebugReporting(
    const std::optional<AdScore>& high_score) {
  std::unique_ptr<PostAuctionSignals> post_auction_signals =
      GeneratePostAuctionSignals(high_score);
  for (const auto& [buyer, get_bid_response] : buyer_bids_list_) {
    std::string ig_owner = buyer;
    for (int i = 0; i < get_bid_response->raw_response().bids_size(); i++) {
      AdWithBid adWithBid = get_bid_response->raw_response().bids().at(i);
      std::string ig_name = adWithBid.interest_group_name();
      if (adWithBid.has_debug_report_urls()) {
        auto done_cb = [&logger_ = logger_, ig_owner,
                        ig_name](absl::StatusOr<absl::string_view> result) {
          if (result.ok()) {
            logger_.vlog(2, "Performed debug reporting for:", ig_owner,
                         ", interest_group: ", ig_name);
          } else {
            logger_.vlog(
                1, "Error while performing debug reporting for: ", ig_owner,
                ",  interest_group: ", ig_name, " ,status:", result.status());
          }
        };
        std::string debug_url;
        if (post_auction_signals->winning_ig_owner == buyer &&
            adWithBid.interest_group_name() ==
                post_auction_signals->winning_ig_name) {
          debug_url = adWithBid.debug_report_urls().auction_debug_win_url();
        } else {
          debug_url = adWithBid.debug_report_urls().auction_debug_loss_url();
        }
        HTTPRequest http_request = CreateDebugReportingHttpRequest(
            debug_url, GetPlaceholderDataForInterestGroupOwner(
                           ig_owner, *post_auction_signals));
        clients_.reporting->DoReport(http_request, done_cb);
      }
    }
  }
}

void SelectAdReactor::OnDone() { delete this; }

void SelectAdReactor::OnCancel() {
  // TODO(b/245982466): Handle early abort and errors.
}

void SelectAdReactor::ReportError(
    ParamWithSourceLoc<ErrorVisibility> error_visibility_with_loc,
    const std::string& msg, ErrorCode error_code) {
  const auto& location = error_visibility_with_loc.location;
  ErrorVisibility error_visibility = error_visibility_with_loc.mandatory_param;
  error_accumulator_.ReportError(location, error_visibility, msg, error_code);
}

}  // namespace privacy_sandbox::bidding_auction_servers
