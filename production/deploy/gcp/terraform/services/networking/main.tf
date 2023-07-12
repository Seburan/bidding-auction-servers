/**
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

resource "google_compute_network" "default" {
  name                    = "${var.operator}-${var.environment}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "backends" {
  for_each = { for index, region in tolist(var.regions) : index => region }

  name          = "${var.operator}-${var.environment}-${each.value}-backend-subnet"
  network       = google_compute_network.default.id
  purpose       = "PRIVATE"
  region        = each.value
  ip_cidr_range = "10.${each.key}.2.0/24"
}

# Frontend address, used for frontend service LB only
resource "google_compute_global_address" "frontend" {
  name       = "${var.operator}-${var.environment}-${var.frontend_service}-lb"
  ip_version = "IPV4"
}

resource "google_compute_global_address" "collector" {
  name       = "${var.collector_service_name}-${var.operator}-${var.environment}-${var.frontend_service}-lb"
  ip_version = "IPV4"
}

resource "google_network_services_mesh" "default" {
  provider = google-beta
  name     = "${var.operator}-${var.environment}-mesh"
}


resource "google_compute_router" "routers" {
  for_each = var.regions

  name    = "${var.operator}-${var.environment}-${each.value}-router"
  network = google_compute_network.default.name
  region  = each.value
}

resource "google_compute_router_nat" "nat" {
  for_each = google_compute_router.routers

  name                               = "${var.operator}-${var.environment}-${each.value.region}-nat"
  router                             = each.value.name
  region                             = each.value.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
