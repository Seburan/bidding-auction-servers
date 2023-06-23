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

variable "operator" {
  description = "Assigned name of an operator in Bidding & Auction system, i.e. seller1, buyer1, buyer2."
  type        = string
}

variable "environment" {
  description = "Assigned environment name to group related resources."
  type        = string
}

variable "ssh_instance_type" {
  description = "type, that is, hardware resource configuration, for EC2 instance"
  type        = string
}

variable "ssh_instance_subnet_ids" {
  description = "A list of subnet ids to launch the SSH instance in."
  type        = list(string)
}

variable "instance_sg_id" {
  description = "Security group to attach to the SSH instance."
  type        = string
}

variable "instance_profile_name" {
  description = "IAM profile to attach to the SSH instance."
  type        = string
}
