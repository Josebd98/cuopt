/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gtest/gtest.h>

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/routing_structures.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <vector>

class SoftTimeWindowsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    handle = std::make_unique<raft::handle_t>();
    n_locations = 4;
    n_vehicles = 2;
    n_orders = 4;
  }

  std::unique_ptr<raft::handle_t> handle;
  int n_locations;
  int n_vehicles;
  int n_orders;
};

TEST_F(SoftTimeWindowsTest, BasicSoftTimeWindowSetup)
{
  // Create data model
  cuopt::routing::data_model_view_t<int, float> data_model(
    handle.get(), n_locations, n_vehicles, n_orders);

  // Set up basic cost matrix
  std::vector<float> cost_matrix_data(n_locations * n_locations, 1.0f);
  for (int i = 0; i < n_locations; ++i) {
    cost_matrix_data[i * n_locations + i] = 0.0f;  // diagonal = 0
  }
  
  rmm::device_uvector<float> cost_matrix(n_locations * n_locations, handle->get_stream());
  raft::copy(cost_matrix.data(), cost_matrix_data.data(), 
             cost_matrix_data.size(), handle->get_stream());
  
  data_model.add_cost_matrix(cost_matrix.data(), 0);

  // Set up order locations
  std::vector<int> order_locations = {0, 1, 2, 3};
  rmm::device_uvector<int> d_order_locations(n_orders, handle->get_stream());
  raft::copy(d_order_locations.data(), order_locations.data(), 
             n_orders, handle->get_stream());
  
  data_model.set_order_locations(d_order_locations.data());

  // Set up regular time windows
  std::vector<int> earliest_times = {0, 10, 20, 30};
  std::vector<int> latest_times = {100, 110, 120, 130};
  
  rmm::device_uvector<int> d_earliest(n_orders, handle->get_stream());
  rmm::device_uvector<int> d_latest(n_orders, handle->get_stream());
  
  raft::copy(d_earliest.data(), earliest_times.data(), n_orders, handle->get_stream());
  raft::copy(d_latest.data(), latest_times.data(), n_orders, handle->get_stream());
  
  data_model.set_order_time_windows(d_earliest.data(), d_latest.data());

  // Set up soft time windows: mix of strict (0) and soft (1)
  std::vector<uint8_t> time_window_types = {0, 1, 0, 1};  // strict, soft, strict, soft
  std::vector<float> penalties = {0.0f, 100.0f, 0.0f, 50.0f};
  
  rmm::device_uvector<uint8_t> d_types(n_orders, handle->get_stream());
  rmm::device_uvector<float> d_penalties(n_orders, handle->get_stream());
  
  raft::copy(d_types.data(), time_window_types.data(), n_orders, handle->get_stream());
  raft::copy(d_penalties.data(), penalties.data(), n_orders, handle->get_stream());
  
  // This should not throw
  EXPECT_NO_THROW(data_model.set_soft_time_windows(d_types.data(), d_penalties.data()));

  // Verify that the soft time windows were set correctly
  auto soft_tw_info = data_model.get_soft_time_windows();
  EXPECT_NE(soft_tw_info.get_time_window_types(), nullptr);
  EXPECT_NE(soft_tw_info.get_penalties(), nullptr);
}

TEST_F(SoftTimeWindowsTest, InvalidSoftTimeWindowTypes)
{
  cuopt::routing::data_model_view_t<int, float> data_model(
    handle.get(), n_locations, n_vehicles, n_orders);

  // Invalid time window types (should only be 0 or 1)
  std::vector<uint8_t> invalid_types = {0, 2, 0, 1};  // 2 is invalid
  std::vector<float> penalties = {0.0f, 100.0f, 0.0f, 50.0f};
  
  rmm::device_uvector<uint8_t> d_types(n_orders, handle->get_stream());
  rmm::device_uvector<float> d_penalties(n_orders, handle->get_stream());
  
  raft::copy(d_types.data(), invalid_types.data(), n_orders, handle->get_stream());
  raft::copy(d_penalties.data(), penalties.data(), n_orders, handle->get_stream());
  
  // This should throw due to validation
  EXPECT_THROW(data_model.set_soft_time_windows(d_types.data(), d_penalties.data()),
               std::exception);
}

TEST_F(SoftTimeWindowsTest, NegativePenalties)
{
  cuopt::routing::data_model_view_t<int, float> data_model(
    handle.get(), n_locations, n_vehicles, n_orders);

  std::vector<uint8_t> types = {0, 1, 0, 1};
  std::vector<float> negative_penalties = {0.0f, -100.0f, 0.0f, 50.0f};  // negative penalty
  
  rmm::device_uvector<uint8_t> d_types(n_orders, handle->get_stream());
  rmm::device_uvector<float> d_penalties(n_orders, handle->get_stream());
  
  raft::copy(d_types.data(), types.data(), n_orders, handle->get_stream());
  raft::copy(d_penalties.data(), negative_penalties.data(), n_orders, handle->get_stream());
  
  // This should throw due to negative penalty validation
  EXPECT_THROW(data_model.set_soft_time_windows(d_types.data(), d_penalties.data()),
               std::exception);
}

TEST_F(SoftTimeWindowsTest, AllStrictTimeWindows)
{
  cuopt::routing::data_model_view_t<int, float> data_model(
    handle.get(), n_locations, n_vehicles, n_orders);

  // All strict time windows
  std::vector<uint8_t> all_strict = {0, 0, 0, 0};
  std::vector<float> penalties = {0.0f, 0.0f, 0.0f, 0.0f};
  
  rmm::device_uvector<uint8_t> d_types(n_orders, handle->get_stream());
  rmm::device_uvector<float> d_penalties(n_orders, handle->get_stream());
  
  raft::copy(d_types.data(), all_strict.data(), n_orders, handle->get_stream());
  raft::copy(d_penalties.data(), penalties.data(), n_orders, handle->get_stream());
  
  EXPECT_NO_THROW(data_model.set_soft_time_windows(d_types.data(), d_penalties.data()));
}

TEST_F(SoftTimeWindowsTest, AllSoftTimeWindows)
{
  cuopt::routing::data_model_view_t<int, float> data_model(
    handle.get(), n_locations, n_vehicles, n_orders);

  // All soft time windows
  std::vector<uint8_t> all_soft = {1, 1, 1, 1};
  std::vector<float> penalties = {100.0f, 200.0f, 150.0f, 75.0f};
  
  rmm::device_uvector<uint8_t> d_types(n_orders, handle->get_stream());
  rmm::device_uvector<float> d_penalties(n_orders, handle->get_stream());
  
  raft::copy(d_types.data(), all_soft.data(), n_orders, handle->get_stream());
  raft::copy(d_penalties.data(), penalties.data(), n_orders, handle->get_stream());
  
  EXPECT_NO_THROW(data_model.set_soft_time_windows(d_types.data(), d_penalties.data()));
}
