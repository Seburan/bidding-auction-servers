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

#ifndef SERVICES_COMMON_CLIENTS_MULTI_CURL_HTTP_FETCHER_ASYNC_H_
#define SERVICES_COMMON_CLIENTS_MULTI_CURL_HTTP_FETCHER_ASYNC_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <curl/multi.h>
#include <grpc/event_engine/event_engine.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "services/common/clients/http/http_fetcher_async.h"
#include "services/common/clients/http/multi_curl_request_manager.h"
#include "src/cpp/concurrent/event_engine_executor.h"

namespace privacy_sandbox::bidding_auction_servers {

// MultiCurlHttpFetcherAsync provides a thread-safe libcurl wrapper to perform
// asynchronous HTTP invocations with client caching(connection pooling), and
// TLS session sharing. It uses a single curl multi handle to perform
// all invoked HTTP actions. It runs a loop in the provided executor
// to provide computation to Libcurl for I/O, and schedules callbacks
// on the provided executor with the result from the HTTP invocation.
// Please note: MultiCurlHttpFetcherAsync makes the best effort to reject any
// calls after the class has started shutting down but does not guarantee
// thread safety if any method is invoked after destruction.
using OnDone = absl::AnyInvocable<void(absl::StatusOr<std::string>) &&>;
class MultiCurlHttpFetcherAsync final : public HttpFetcherAsync {
 public:
  // Constructs a new multi session for performing HTTP calls.
  // LIBCurl maintains persistent HTTP connections by default.
  // A TCP connection to all request servers is kept warm to reduce TCP
  // handshake latency for each request.
  // If no data is transferred over the TCP connection for keepalive_idle_sec,
  // OS sends a keep alive probe.
  // If the other endpoint does not reply, OS sends another keep alive
  // probe after keepalive_interval_sec.
  explicit MultiCurlHttpFetcherAsync(server_common::Executor* executor,
                                     int64_t keepalive_interval_sec = 2,
                                     int64_t keepalive_idle_sec = 2);

  // Cleans up all sessions and errors out any pending open HTTP calls.
  // Please note: Any class using this must ensure that the instance is only
  // destructed when they can ensure that the instance will no longer be invoked
  // from any threads.
  ~MultiCurlHttpFetcherAsync() override
      ABSL_LOCKS_EXCLUDED(in_loop_mu_, curl_data_map_lock_);

  // Not copyable or movable.
  MultiCurlHttpFetcherAsync(const MultiCurlHttpFetcherAsync&) = delete;
  MultiCurlHttpFetcherAsync& operator=(const MultiCurlHttpFetcherAsync&) =
      delete;

  // Fetches provided url with libcurl.
  //
  // http_request: The URL and headers for the HTTP GET request.
  // timeout_ms: The request timeout
  // done_callback: Output param. Invoked either on error or after finished
  // receiving a response. This method guarantees that the callback will be
  // invoked once with the obtained result or error.
  // Please note: done_callback will run in a threadpool and is not guaranteed
  // to be the FetchUrl client's thread.
  void FetchUrl(const HTTPRequest& request, int timeout_ms,
                OnDone done_callback) override
      ABSL_LOCKS_EXCLUDED(curl_data_map_lock_);

 private:
  // This struct maintains the data related to a Curl request, some of which
  // has to stay valid throughout the life of the request. The code maintains a
  // reference in curl_data_map_ till the request is completed. The destructor
  // is then to free the resources in this class after the request completes.
  struct CurlRequestData {
    // The easy handle provided by libcurl, registered to
    // multi_curl_request_manager_.
    CURL* req_handle;

    // The pointer to the linked list of the request HTTP headers.
    struct curl_slist* headers_list_ptr = nullptr;

    // The callback function for this request from FetchUrl.
    OnDone done_callback;

    // The pointer that is used by the req_handle to write the request output.
    std::unique_ptr<std::string> output;

    CurlRequestData(const std::vector<std::string>& headers, OnDone on_done);
    ~CurlRequestData();
  };
  // This method adds the curl handle and callback to the callback_map.
  // Only a single thread can execute this function at a time since it requires
  // the acquisition of the callback_map_lock_ mutex.
  void Add(CURL* handle, std::unique_ptr<CurlRequestData> done_callback)
      ABSL_LOCKS_EXCLUDED(curl_data_map_lock_);

  // This method executes PerformCurlUpdate on a loop in the executor_. It
  // will schedule itself as a new task to perform curl check again.
  void ExecuteLoop() ABSL_LOCKS_EXCLUDED(in_loop_mu_);

  // Performs the fetch and handles the response from libcurl. It checks if
  // the req_manager is done performing the fetch. This method provides
  // computation to the underlying Curl Multi to perform I/O and polls for
  // a response. Once the Curl multi interface indicates that a response is
  // available, it schedules the callback on the executor_.
  // Only a single thread can execute this function at a time since it requires
  // the acquisition of the in_loop_mu_ mutex.
  void PerformCurlUpdate() ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_loop_mu_)
      ABSL_LOCKS_EXCLUDED(curl_data_map_lock_);

  // The executor_ will receive tasks from PerformCurlUpdate. The tasks will
  // schedule future ExecuteLoop calls and schedule executions for
  // client callbacks. The executor is not owned by this class instance but is
  // required to outlive the lifetime of this class instance.
  server_common::Executor* executor_;

  // Wait time before sending keepalive probes.
  int64_t keepalive_idle_sec_;

  // Interval time between keep-alive probes in case of no response.
  int64_t keepalive_interval_sec_;

  // The multi session used for performing HTTP calls.
  MultiCurlRequestManager multi_curl_request_manager_;

  // Makes sure only one execution loop runs at a time.
  absl::Mutex in_loop_mu_;

  // Synchronizes the status of shutdown for destructor and execution loop.
  absl::Notification shutdown_requested_;
  absl::Notification shutdown_complete_;

  // A map of curl easy handles to curl data for easy tracking.
  absl::Mutex curl_data_map_lock_;
  absl::flat_hash_map<CURL*, std::unique_ptr<CurlRequestData>> curl_data_map_
      ABSL_GUARDED_BY(curl_data_map_lock_);
};

}  // namespace privacy_sandbox::bidding_auction_servers

#endif  // SERVICES_COMMON_CLIENTS_MULTI_CURL_HTTP_FETCHER_ASYNC_H_
