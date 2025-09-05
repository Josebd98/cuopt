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
#include "../node/time_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class time_route_t {
 public:
  time_route_t(solution_handle_t<i_t, f_t> const* sol_handle_, time_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      departure_forward(0, sol_handle_->get_stream()),
      excess_forward(0, sol_handle_->get_stream()),
      soft_excess_forward(0, sol_handle_->get_stream()),
      departure_backward(0, sol_handle_->get_stream()),
      excess_backward(0, sol_handle_->get_stream()),
      soft_excess_backward(0, sol_handle_->get_stream()),
      window_start(0, sol_handle_->get_stream()),
      window_end(0, sol_handle_->get_stream()),
      transit_time_forward(0, sol_handle_->get_stream()),
      latest_arrival_forward(0, sol_handle_->get_stream()),
      unavoidable_wait_forward(0, sol_handle_->get_stream()),
      transit_time_backward(0, sol_handle_->get_stream()),
      earliest_arrival_backward(0, sol_handle_->get_stream()),
      unavoidable_wait_backward(0, sol_handle_->get_stream()),
      actual_arrival(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero time_route_t copy_ctr");
  }

  time_route_t(const time_route_t& time_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(time_route.dim_info),
      departure_forward(time_route.departure_forward, sol_handle_->get_stream()),
      excess_forward(time_route.excess_forward, sol_handle_->get_stream()),
      soft_excess_forward(time_route.soft_excess_forward, sol_handle_->get_stream()),
      departure_backward(time_route.departure_backward, sol_handle_->get_stream()),
      excess_backward(time_route.excess_backward, sol_handle_->get_stream()),
      soft_excess_backward(time_route.soft_excess_backward, sol_handle_->get_stream()),
      window_start(time_route.window_start, sol_handle_->get_stream()),
      window_end(time_route.window_end, sol_handle_->get_stream()),
      transit_time_forward(time_route.transit_time_forward, sol_handle_->get_stream()),
      latest_arrival_forward(time_route.latest_arrival_forward, sol_handle_->get_stream()),
      unavoidable_wait_forward(time_route.unavoidable_wait_forward, sol_handle_->get_stream()),
      transit_time_backward(time_route.transit_time_backward, sol_handle_->get_stream()),
      earliest_arrival_backward(time_route.earliest_arrival_backward, sol_handle_->get_stream()),
      unavoidable_wait_backward(time_route.unavoidable_wait_backward, sol_handle_->get_stream()),
      actual_arrival(time_route.actual_arrival, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("time route copy_ctr");
  }

  time_route_t& operator=(time_route_t&& time_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    departure_forward.resize(max_nodes_per_route, stream);
    excess_forward.resize(max_nodes_per_route, stream);
    soft_excess_forward.resize(max_nodes_per_route, stream);
    departure_backward.resize(max_nodes_per_route, stream);
    excess_backward.resize(max_nodes_per_route, stream);
    soft_excess_backward.resize(max_nodes_per_route, stream);
    window_start.resize(max_nodes_per_route, stream);
    window_end.resize(max_nodes_per_route, stream);
    actual_arrival.resize(max_nodes_per_route, stream);

    if (dim_info.should_compute_travel_time()) {
      transit_time_forward.resize(max_nodes_per_route, stream);
      latest_arrival_forward.resize(max_nodes_per_route, stream);
      unavoidable_wait_forward.resize(max_nodes_per_route, stream);

      transit_time_backward.resize(max_nodes_per_route, stream);
      earliest_arrival_backward.resize(max_nodes_per_route, stream);
      unavoidable_wait_backward.resize(max_nodes_per_route, stream);
    }
  }

  struct forward_view_t {};

  struct backward_view_t {};

  struct view_t {
    bool is_empty() const { return window_start.empty(); }

    DI time_node_t<i_t, f_t> get_node(i_t idx) const
    {
      time_node_t<i_t, f_t> time_node;
      time_node.departure_forward  = departure_forward[idx];
      time_node.excess_forward     = excess_forward[idx];
      time_node.soft_excess_forward = soft_excess_forward[idx];
      time_node.departure_backward = departure_backward[idx];
      time_node.excess_backward    = excess_backward[idx];
      time_node.soft_excess_backward = soft_excess_backward[idx];
      time_node.window_start       = window_start[idx];
      time_node.window_end         = window_end[idx];
      
      // 🔧 SOLUCIÓN: Establecer is_soft_node basado en dim_info
      if (dim_info.has_soft_time_windows() && dim_info.soft_tw_types != nullptr && idx < 10000) {
        time_node.is_soft_node = (dim_info.soft_tw_types[idx] == 1);
        // printf("🔄 GET_NODE[%d]: soft_tw_types[%d]=%d → is_soft_node=%s\n", 
        //        idx, idx, (int)dim_info.soft_tw_types[idx],
        //        time_node.is_soft_node ? "TRUE" : "FALSE");
      }
      if (dim_info.should_compute_travel_time()) {
        time_node.transit_time_forward     = transit_time_forward[idx];
        time_node.latest_arrival_forward   = latest_arrival_forward[idx];
        time_node.unavoidable_wait_forward = unavoidable_wait_forward[idx];

        time_node.transit_time_backward     = transit_time_backward[idx];
        time_node.earliest_arrival_backward = earliest_arrival_backward[idx];
        time_node.unavoidable_wait_backward = unavoidable_wait_backward[idx];
      }
      return time_node;
    }

    DI void set_node(i_t idx, const time_node_t<i_t, f_t>& node)
    {
      window_start[idx] = node.window_start;
      window_end[idx]   = node.window_end;
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

      DI void set_forward_data(i_t idx, const time_node_t<i_t, f_t>& node)
  {
    departure_forward[idx] = node.departure_forward;
    excess_forward[idx]    = node.excess_forward;
    soft_excess_forward[idx] = node.soft_excess_forward;
    
    // Solo logs importantes para violaciones
    if (idx < 10 && (node.excess_forward > 0.0 || node.soft_excess_forward > 0.0)) {
      printf("📋 SET[%d]: excess=%.0f, soft_excess=%.0f\n", 
             idx, node.excess_forward, node.soft_excess_forward);
    }

      if (dim_info.should_compute_travel_time()) {
        transit_time_forward[idx]     = node.transit_time_forward;
        latest_arrival_forward[idx]   = node.latest_arrival_forward;
        unavoidable_wait_forward[idx] = node.unavoidable_wait_forward;
      }
    }

      DI void set_backward_data(i_t idx, const time_node_t<i_t, f_t>& node)
  {
    departure_backward[idx] = node.departure_backward;
    excess_backward[idx]    = node.excess_backward;
    soft_excess_backward[idx] = node.soft_excess_backward;

      if (dim_info.should_compute_travel_time()) {
        transit_time_backward[idx]     = node.transit_time_backward;
        earliest_arrival_backward[idx] = node.earliest_arrival_backward;
        unavoidable_wait_backward[idx] = node.unavoidable_wait_backward;
      }
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(departure_forward.subspan(write_start),
                 orig_route.departure_forward.subspan(start_idx),
                 size);
      block_copy(
        excess_forward.subspan(write_start), orig_route.excess_forward.subspan(start_idx), size);
      block_copy(
        soft_excess_forward.subspan(write_start), orig_route.soft_excess_forward.subspan(start_idx), size);

      if (dim_info.should_compute_travel_time()) {
        block_copy(transit_time_forward.subspan(write_start),
                   orig_route.transit_time_forward.subspan(start_idx),
                   size);
        block_copy(latest_arrival_forward.subspan(write_start),
                   orig_route.latest_arrival_forward.subspan(start_idx),
                   size);
        block_copy(unavoidable_wait_forward.subspan(write_start),
                   orig_route.unavoidable_wait_forward.subspan(start_idx),
                   size);
      }
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(departure_backward.subspan(write_start),
                 orig_route.departure_backward.subspan(start_idx),
                 size);
      block_copy(
        excess_backward.subspan(write_start), orig_route.excess_backward.subspan(start_idx), size);
      block_copy(
        soft_excess_backward.subspan(write_start), orig_route.soft_excess_backward.subspan(start_idx), size);

      if (dim_info.should_compute_travel_time()) {
        block_copy(transit_time_backward.subspan(write_start),
                   orig_route.transit_time_backward.subspan(start_idx),
                   size);
        block_copy(earliest_arrival_backward.subspan(write_start),
                   orig_route.earliest_arrival_backward.subspan(start_idx),
                   size);
        block_copy(unavoidable_wait_backward.subspan(write_start),
                   orig_route.unavoidable_wait_backward.subspan(start_idx),
                   size);
      }
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      auto size = to_idx - from_idx;
      block_copy(
        window_start.subspan(write_start), orig_route.window_start.subspan(from_idx), size);
      block_copy(window_end.subspan(write_start), orig_route.window_end.subspan(from_idx), size);
    }

    template<typename RouteView>
    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost,
                         const RouteView* route = nullptr) const noexcept
    {
      inf_cost[dim_t::TIME] = static_cast<double>(excess_forward[n_nodes_route]);

      if (dim_info.should_compute_travel_time()) {
        const double total_time =
          transit_time_forward[n_nodes_route] + unavoidable_wait_forward[n_nodes_route];

        if (dim_info.has_travel_time_obj) { obj_cost[objective_t::TRAVEL_TIME] = total_time; }

        if (dim_info.has_max_constraint) {
          inf_cost[dim_t::TIME] += max(0., total_time - (double)vehicle_info.max_time);
        }
      }

      // ===== SOFT TIME WINDOW PROCESSING =====
      printf("🎯 RUTA[%d]: excess=%.0f, soft_excess=%.0f → inf_cost=%.0f\n", 
             n_nodes_route, 
             static_cast<double>(excess_forward[n_nodes_route]),
             static_cast<double>(soft_excess_forward[n_nodes_route]),
             inf_cost[dim_t::TIME]);
      
      if (dim_info.has_soft_time_windows() && dim_info.soft_tw_types != nullptr && dim_info.soft_tw_penalties != nullptr) {
        // printf("DEBUG: Entering soft time window penalty calculation\n");
        double total_soft_penalty = 0.0;
        
        for (i_t i = 0; i < n_nodes_route; ++i) {
          // Get the actual node ID (not route position)
          i_t node_id = (route != nullptr) ? route->node_id(i) : i; // fallback to position if no route
          
          // Check if this node has a soft time window using node ID
          // printf("DEBUG: Node %d (pos %d): soft_tw_types[%d] = %d, window=[%.1f,%.1f]\n", 
          //        node_id, i, node_id, (int)dim_info.soft_tw_types[node_id], window_start[i], window_end[i]);
          if (dim_info.soft_tw_types[node_id] == 1) { // 1 = soft time window
            double earliest_time = window_start[i];
            double latest_time = window_end[i];
            f_t penalty_rate = static_cast<const f_t*>(dim_info.soft_tw_penalties)[order_idx];
            
            // Reconstruct the unconstrained arrival time
            // departure_forward is clamped to [earliest, latest] for strict windows
            // excess_forward contains the amount of late violation for strict windows
            double unconstrained_arrival = departure_forward[i] + excess_forward[i];
            
            // printf("DEBUG: Node %d SOFT: window=[%.1f,%.1f], arrival=%.1f, penalty_rate=%.1f\n", 
            //        node_id, earliest_time, latest_time, unconstrained_arrival, penalty_rate);
            
            // For soft windows, we want to penalize based on the unconstrained time
            double early_violation = max(0.0, earliest_time - unconstrained_arrival);
            double late_violation = max(0.0, unconstrained_arrival - latest_time);
            
            // printf("DEBUG: Node %d violations: early=%.1f, late=%.1f\n", node_id, early_violation, late_violation);
            
            // Calculate penalty for this soft node
            total_soft_penalty += (early_violation + late_violation) * penalty_rate;
          }
        }
        
        // ===== CRITICAL INSIGHT =====
        // The beauty of our two-tier system:
        // - excess_forward[n_nodes_route] = ONLY strict violations
        // - soft_excess_forward[n_nodes_route] = ONLY soft violations
        // They are ALREADY SEPARATED by design!
        
        // printf("DEBUG ROUTE: ===== FINAL COST SEPARATION =====\n");
        printf("💰 SOFT penalty: %.0f, STRICT violations: %.0f → inf_cost: %.0f\n", 
               total_soft_penalty, static_cast<double>(excess_forward[n_nodes_route]), inf_cost[dim_t::TIME]);
        
        obj_cost[objective_t::SOFT_TIME_WINDOW_PENALTY] = total_soft_penalty;
        
        // KEY: inf_cost[TIME] already contains ONLY strict violations from excess_forward
        // We don't need to subtract anything because soft violations never went there!
        // Verification: show total violations if any
        double total_violations = static_cast<double>(excess_forward[n_nodes_route]) + 
                                 static_cast<double>(soft_excess_forward[n_nodes_route]);
        if (total_violations > 0) {
          printf("🔍 TOTAL: %.0f violations (strict=%.0f + soft=%.0f)\n", 
                 total_violations, static_cast<double>(excess_forward[n_nodes_route]), 
                 static_cast<double>(soft_excess_forward[n_nodes_route]));
        }
      }
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem,
                                                              const time_dimension_info_t dim_info,
                                                              i_t n_nodes_route)
    {
      view_t v;
      size_t sz                                 = n_nodes_route + 1;
      i_t* sh_ptr                               = shmem;
      v.dim_info                                = dim_info;
      thrust::tie(v.departure_forward, sh_ptr)  = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.excess_forward, sh_ptr)     = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.soft_excess_forward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.departure_backward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.excess_backward, sh_ptr)    = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.soft_excess_backward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.window_start, sh_ptr)       = wrap_ptr_as_span<double>(sh_ptr, sz);
      thrust::tie(v.window_end, sh_ptr)         = wrap_ptr_as_span<double>(sh_ptr, sz);

      if (dim_info.should_compute_travel_time()) {
        thrust::tie(v.transit_time_forward, sh_ptr)     = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.latest_arrival_forward, sh_ptr)   = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.unavoidable_wait_forward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);

        thrust::tie(v.transit_time_backward, sh_ptr)     = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.earliest_arrival_backward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
        thrust::tie(v.unavoidable_wait_backward, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, sz);
      }

      return thrust::make_tuple(v, sh_ptr);
    }

    time_dimension_info_t dim_info;
    raft::device_span<double> departure_forward;
    raft::device_span<double> excess_forward;
    raft::device_span<double> soft_excess_forward;
    raft::device_span<double> departure_backward;
    raft::device_span<double> excess_backward;
    raft::device_span<double> soft_excess_backward;
    raft::device_span<double> window_start;
    raft::device_span<double> window_end;
    raft::device_span<double> transit_time_forward;
    raft::device_span<double> latest_arrival_forward;
    raft::device_span<double> unavoidable_wait_forward;
    raft::device_span<double> transit_time_backward;
    raft::device_span<double> earliest_arrival_backward;
    raft::device_span<double> unavoidable_wait_backward;
    raft::device_span<double> actual_arrival;
    // Soft time window support
    raft::device_span<uint8_t const> soft_tw_types;
    raft::device_span<f_t const> soft_tw_penalties;
  };

  view_t view()
  {
    view_t v;
    v.dim_info = dim_info;
    v.departure_forward =
      raft::device_span<double>{departure_forward.data(), departure_forward.size()};
    v.excess_forward = raft::device_span<double>{excess_forward.data(), excess_forward.size()};
    v.soft_excess_forward = raft::device_span<double>{soft_excess_forward.data(), soft_excess_forward.size()};
    v.departure_backward =
      raft::device_span<double>{departure_backward.data(), departure_backward.size()};
    v.excess_backward = raft::device_span<double>{excess_backward.data(), excess_backward.size()};
    v.soft_excess_backward = raft::device_span<double>{soft_excess_backward.data(), soft_excess_backward.size()};
    v.window_start    = raft::device_span<double>{window_start.data(), window_start.size()};
    v.window_end      = raft::device_span<double>{window_end.data(), window_end.size()};

    v.transit_time_forward =
      raft::device_span<double>{transit_time_forward.data(), transit_time_forward.size()};
    v.latest_arrival_forward =
      raft::device_span<double>{latest_arrival_forward.data(), latest_arrival_forward.size()};
    v.unavoidable_wait_forward =
      raft::device_span<double>{unavoidable_wait_forward.data(), unavoidable_wait_forward.size()};

    v.transit_time_backward =
      raft::device_span<double>{transit_time_backward.data(), transit_time_backward.size()};
    v.earliest_arrival_backward =
      raft::device_span<double>{earliest_arrival_backward.data(), earliest_arrival_backward.size()};
    v.unavoidable_wait_backward =
      raft::device_span<double>{unavoidable_wait_backward.data(), unavoidable_wait_backward.size()};

    v.actual_arrival = raft::device_span<double>{actual_arrival.data(), actual_arrival.size()};
    
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a time route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size, time_dimension_info_t dim_info)
  {
    // departure_forward, excess_forward, soft_excess_forward, departure_backward, 
    // excess_backward, soft_excess_backward, window_start, window_end
    return (8 + 6 * dim_info.should_compute_travel_time()) * route_size * sizeof(double);
  }

  time_dimension_info_t dim_info;

  // forward data
  rmm::device_uvector<double> departure_forward;
  // excess forward
  rmm::device_uvector<double> excess_forward;
  // soft excess forward (for soft time window violations)
  rmm::device_uvector<double> soft_excess_forward;
  // backward info
  rmm::device_uvector<double> departure_backward;
  // excess backward
  rmm::device_uvector<double> excess_backward;
  // soft excess backward (for soft time window violations)
  rmm::device_uvector<double> soft_excess_backward;
  // windows_start
  rmm::device_uvector<double> window_start;
  // window end
  rmm::device_uvector<double> window_end;
  // forward accumulated data
  rmm::device_uvector<double> transit_time_forward;
  rmm::device_uvector<double> latest_arrival_forward;
  rmm::device_uvector<double> unavoidable_wait_forward;

  // backward accumulated data
  rmm::device_uvector<double> transit_time_backward;
  rmm::device_uvector<double> earliest_arrival_backward;
  rmm::device_uvector<double> unavoidable_wait_backward;

  // only used when we want to get the actual arrival time. currently it is used only for reporting
  // purpose and so is not included in shared route
  rmm::device_uvector<double> actual_arrival;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
