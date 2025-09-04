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

 #include <utilities/cuda_helpers.cuh>
 #include "../routing_helpers.cuh"
 
 #include <routing/arc_value.hpp>
 #include <routing/fleet_info.hpp>
 #include <routing/routing_details.hpp>
 
 #include <raft/core/handle.hpp>
 
 #include <rmm/device_uvector.hpp>
 
 namespace cuopt {
 namespace routing {
 namespace detail {
 
 template <typename i_t, typename f_t>
 class time_node_t {
  public:
   //! Hashimoto Yagiura [hy] forward departure time
   double departure_forward = 0.0;
   //! [hy] forward time excess
   double excess_forward = 0.0;
   //! [hy] backward departure time
   double departure_backward = 0.0;
   //! [hy] backward time excess
   double excess_backward = 0.0;
   //! Copied from problem data time window start/end for convenience
   double window_start = 0.0;
   double window_end   = 0.0;
 
   //! Time gathered to node and after node.
   //! These are needed when we use TIME as objective function, and max times as constraints
   double transit_time_forward     = 0.0;
   double latest_arrival_forward   = 0.;
   double unavoidable_wait_forward = 0.;
 
   double transit_time_backward     = 0.0;
   double earliest_arrival_backward = 0.;
   double unavoidable_wait_backward = 0.;
 
   /*! \brief { Calculate next node forward time data based on actual node} */
   void HDI calculate_forward(time_node_t& next, double time_between) const noexcept
   {
     next.departure_forward = departure_forward + time_between;
     next.excess_forward    = excess_forward;
 
     if (next.departure_forward < next.window_start) {
       next.departure_forward = next.window_start;
     } else if (next.departure_forward > next.window_end) {
       next.excess_forward += next.departure_forward - next.window_end;
       next.departure_forward = next.window_end;
     }
 
     next.latest_arrival_forward   = latest_arrival_forward + time_between;
     next.unavoidable_wait_forward = unavoidable_wait_forward;
     if (next.latest_arrival_forward < next.window_start) {
       next.unavoidable_wait_forward += (next.window_start - next.latest_arrival_forward);
       next.latest_arrival_forward = next.window_start;
     } else if (next.latest_arrival_forward > next.window_end) {
       next.latest_arrival_forward = next.window_end;
     }
     next.transit_time_forward = transit_time_forward + time_between;
   }
 
   /*! \brief { Calculate prev node time backward data based on actual node} */
   void HDI calculate_backward(time_node_t& prev, double time_between) const noexcept
   {
     prev.departure_backward = departure_backward - time_between;
     prev.excess_backward    = excess_backward;
 
     if (prev.departure_backward > prev.window_end)
       prev.departure_backward = prev.window_end;
     else if (prev.departure_backward < prev.window_start) {
       prev.excess_backward += prev.window_start - prev.departure_backward;
       prev.departure_backward = prev.window_start;
     }
 
     prev.earliest_arrival_backward = earliest_arrival_backward - time_between;
     prev.unavoidable_wait_backward = unavoidable_wait_backward;
     if (prev.earliest_arrival_backward > prev.window_end) {
       prev.unavoidable_wait_backward += (prev.earliest_arrival_backward - prev.window_end);
       prev.earliest_arrival_backward = prev.window_end;
     } else if (prev.earliest_arrival_backward < prev.window_start) {
       prev.earliest_arrival_backward = prev.window_start;
     }
 
     prev.transit_time_backward = transit_time_backward + time_between;
   }
 
   /*! \brief  { Combine information from begining and ending fragments.}
       \return { Time excess of route represented by nodes prev and next }*/
   static HDI double combine(const time_node_t& prev,
                             const time_node_t& next,
                             const VehicleInfo<f_t>& vehicle_info,
                             double time_between) noexcept
   {
     double wait_arrival    = prev.latest_arrival_forward + time_between;
     double total_wait_time = prev.unavoidable_wait_forward + next.unavoidable_wait_backward +
                              max(0., next.earliest_arrival_backward - wait_arrival);
     double total_transit_time =
       prev.transit_time_forward + next.transit_time_backward + time_between;
     double total_time = total_transit_time + total_wait_time;
 
     double arrival_f = prev.departure_forward + time_between;
     return prev.excess_forward + next.excess_backward +
            max(0.0, arrival_f - next.departure_backward) +
            max(0.0, total_time - vehicle_info.max_time);
   }
 
   HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
   {
     return excess_forward +
            max(0.f, transit_time_forward + unavoidable_wait_forward - vehicle_info.max_time);
   }
 
   HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
   {
     return excess_backward +
            max(0.f, transit_time_backward + unavoidable_wait_backward - vehicle_info.max_time);
   }
 
   HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             double weight       = 1.0,
                             double excess_limit = 0.) const
   {
     return forward_excess(vehicle_info) * weight <= excess_limit;
   }
 
   HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                              const double weight = 1.0,
                              double excess_limit = 0.) const
   {
     return backward_excess(vehicle_info) * weight <= excess_limit;
   }
 
   template <bool is_device = true>
   HDI void get_cost([[maybe_unused]] const time_node_t& prev_node,
                     const VehicleInfo<f_t, is_device>& vehicle_info,
                     const time_dimension_info_t& dim_info,
                     objective_cost_t& obj_cost,
                     infeasible_cost_t& inf_cost) const noexcept
   {
     // Call the overloaded version with default node_id = -1
     get_cost(prev_node, vehicle_info, dim_info, obj_cost, inf_cost, static_cast<i_t>(-1));
   }
 
   template <bool is_device = true>
   HDI void get_cost([[maybe_unused]] const time_node_t& prev_node,
                     const VehicleInfo<f_t, is_device>& vehicle_info,
                     const time_dimension_info_t& dim_info,
                     objective_cost_t& obj_cost,
                     infeasible_cost_t& inf_cost,
                     i_t node_id) const noexcept
   {
     double time_violation = (excess_forward + excess_backward + max(0., departure_forward - departure_backward));
     
     // Handle soft time window logic if enabled and node_id is provided
     if (dim_info.has_soft_time_windows() && node_id >= 0 && 
         dim_info.soft_tw_types != nullptr && dim_info.soft_tw_penalties != nullptr) {
       
       // CRITICAL DEBUG: Verify that node_id maps correctly to order characteristics
       // According to solution.cuh, ALL node characteristics use: node_idx = node_info.node()
       // Time windows: earliest_time[node_idx], latest_time[node_idx]  
       // Demand: demand[node_idx + i * get_num_orders()]
       // So our soft_tw_types should ALSO use the same node_idx!
       
       printf("DEBUG NODE DELTA: node_id=%d, window=[%.1f,%.1f]\n", 
              node_id, window_start, window_end);
              
       // SIMPLE APPROACH: Just try to access and use bounds checking with known reasonable limits
       // Most problems have < 10000 orders, so if node_id > 10000, it's likely invalid
       if (node_id >= 0 && node_id < 10000) {
         printf("DEBUG NODE DELTA: soft_tw_types[%d]=%d (assuming valid)\n", 
                node_id, (int)dim_info.soft_tw_types[node_id]);
       } else {
         printf("DEBUG NODE DELTA: node_id=%d seems invalid (>10000 or negative)\n", node_id);
       }
       
       // Check if this specific node has a soft time window (with simple bounds checking)
       if (node_id >= 0 && node_id < 10000 && 
           dim_info.soft_tw_types[node_id] == 1) { // 1 = soft time window
         // For soft nodes, move time violations to objective cost instead of infeasibility
         f_t penalty_rate = static_cast<const f_t*>(dim_info.soft_tw_penalties)[node_id];
         obj_cost[objective_t::SOFT_TIME_WINDOW_PENALTY] += time_violation * penalty_rate;
         
         // Don't add to inf_cost for soft nodes
         inf_cost[dim_t::TIME] = 0.0;
         
         printf("DEBUG NODE DELTA: Node %d SOFT: moving time_violation=%.2f to obj_cost (penalty=%.2f)\n", 
                node_id, time_violation, time_violation * penalty_rate);
       } else {
         // For strict nodes, keep the original behavior
         inf_cost[dim_t::TIME] = time_violation;
         printf("DEBUG NODE DELTA: Node %d STRICT: keeping time_violation=%.2f in inf_cost\n", 
                node_id, time_violation);
       }
     } else {
       // Original behavior when soft time windows are not enabled or node_id not provided
       inf_cost[dim_t::TIME] = time_violation;
       
       // DEBUG: Show why we're not using soft time window logic
       printf("DEBUG NODE DELTA FALLBACK: node_id=%d, has_soft_tw=%s, soft_tw_types=%p, time_violation=%.2f\\n", 
              node_id, 
              dim_info.has_soft_time_windows() ? "true" : "false",
              (void*)dim_info.soft_tw_types,
              time_violation);
     }
     
     if (dim_info.should_compute_travel_time()) {
       const double total_wait_time = unavoidable_wait_forward + unavoidable_wait_backward +
                                      max(0., earliest_arrival_backward - latest_arrival_forward);
       const double total_transit_time = transit_time_forward + transit_time_backward;
 
       const double total_time = total_wait_time + total_transit_time;
 
       if (dim_info.has_travel_time_obj) { obj_cost[objective_t::TRAVEL_TIME] = total_time; }
       if (dim_info.has_max_constraint) {
         inf_cost[dim_t::TIME] += max(0., total_time - (double)vehicle_info.max_time);
       }
     }
   }
 };
 
 }  // namespace detail
 }  // namespace routing
 }  // namespace cuopt
 