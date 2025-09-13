/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #pragma once

 #include <raft/core/device_span.hpp>
 
 #include <cstdint>
 #include <set>
 #include <string>
 #include <tuple>
 namespace cuopt {
 namespace routing {
 /**
  * @brief Enumerated representation of supported objective function types
  *
  */
 enum class objective_t {
   COST,         // Cost of all the routes according to the cost matrix
   TRAVEL_TIME,  // Driving time (excludes the wait time) of all the routes according to the travel
                 // time matirx
   VARIANCE_ROUTE_SIZE,          // Variance in route sizes
   VARIANCE_ROUTE_SERVICE_TIME,  // Variance in route service times
   PRIZE,                        // Sum of prizes of all orders that are served
   VEHICLE_FIXED_COST,           // Used when fixed vehicle cost are enabled
   SOFT_TIME_WINDOW_PENALTY,     // Penalty for violating soft time window constraints
   SIZE  // Helper enum to keep track of number of supported objective functions
 };
 
 enum class node_type_t : uint8_t { DEPOT = 0, PICKUP, DELIVERY, BREAK };
 
 using demand_i_t = int32_t;
 using cap_i_t    = int32_t;
 
 namespace detail {
 
 template <typename i_t, typename f_t>
 class break_dimension_t {
  public:
   break_dimension_t(i_t const* break_earliest, i_t const* break_latest, i_t const* break_duration)
     : break_earliest_(break_earliest), break_latest_(break_latest), break_duration_(break_duration)
   {
   }
 
   std::tuple<i_t const*, i_t const*, i_t const*> get_breaks() const
   {
     return std::make_tuple(break_earliest_, break_latest_, break_duration_);
   }
 
  private:
   i_t const* break_earliest_;
   i_t const* break_latest_;
   i_t const* break_duration_;
 };
 
 template <typename i_t>
 class vehicle_break_t {
  public:
   vehicle_break_t(i_t earliest, i_t latest, i_t duration, raft::device_span<const i_t> locations)
     : earliest_(earliest), latest_(latest), duration_(duration), locations_(locations)
   {
   }
 
   i_t earliest_;
   i_t latest_;
   i_t duration_;
   raft::device_span<const i_t> locations_{};
 };
 
 template <typename i_t, typename f_t>
 class vehicle_time_window_t {
  public:
   vehicle_time_window_t(i_t const* earliest, i_t const* latest)
     : earliest_(earliest), latest_(latest)
   {
   }
   vehicle_time_window_t() = default;
   i_t const* get_earliest_time() const { return earliest_; }
   i_t const* get_latest_time() const { return latest_; }
 
  private:
   i_t const* earliest_{nullptr};
   i_t const* latest_{nullptr};
 };
 
 template <typename i_t, typename f_t>
 class capacity_t {
  public:
   capacity_t(std::string const& name, i_t const* demands, i_t const* vehicle_capacities)
     : name_(name), demands_(demands), vehicle_capacities_(vehicle_capacities)
   {
   }
   i_t const* get_demands() const { return demands_; }
   i_t const* get_vehicle_capacities() const { return vehicle_capacities_; }
 
  private:
   std::string name_{nullptr};
   i_t const* demands_{nullptr};
   i_t const* vehicle_capacities_{nullptr};
 };
 
 // internal
 template <typename i_t, typename f_t>
 class order_time_window_t {
  public:
   order_time_window_t(i_t const* earliest, i_t const* latest) : earliest_(earliest), latest_(latest)
   {
   }
 
   order_time_window_t() = default;
 
   i_t const* get_earliest_time() const { return earliest_; }
   i_t const* get_latest_time() const { return latest_; }
 
  private:
   i_t const* earliest_{nullptr};
   i_t const* latest_{nullptr};
 };
 
 // Soft time window constraints structure
 template <typename i_t, typename f_t>
 class soft_time_window_t {
  public:
   soft_time_window_t(uint8_t const* time_window_types, f_t const* penalties)
     : time_window_types_(time_window_types), penalties_(penalties)
   {
   }
 
   soft_time_window_t() = default;
 
   uint8_t const* get_time_window_types() const { return time_window_types_; }
   f_t const* get_penalties() const { return penalties_; }
 
  // Check if a specific order has a soft time window (1 = soft, 0 = strict)
  // Note: node_idx is the order ID (0, 1, 2, ...) - depot is not included in arrays
  __device__ inline bool is_soft_time_window(i_t node_idx) const {
    return time_window_types_ != nullptr && time_window_types_[node_idx] == 1;
  }

  // Get penalty rate for a specific order
  // Note: node_idx is the order ID (0, 1, 2, ...) - depot is not included in arrays
  __device__ inline f_t get_penalty_rate(i_t node_idx) const {
    return penalties_ != nullptr ? penalties_[node_idx] : f_t(0.0);
  }
 
  private:
   uint8_t const* time_window_types_{nullptr};  // 0 = strict, 1 = soft
   f_t const* penalties_{nullptr};               // penalty per unit violation
 };
 
 }  // namespace detail
 }  // namespace routing
 }  // namespace cuopt
 
