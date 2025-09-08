#include <gtest/gtest.h>

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/routing_structures.hpp>
#include <cuopt/routing/solve.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/copy.hpp>

#include <rmm/device_uvector.hpp>

#include <vector>
#include <iostream>

class SC25SimpleTest : public ::testing::Test {
protected:
  void SetUp() override {
    handle = std::make_unique<raft::handle_t>();
  }

  std::unique_ptr<raft::handle_t> handle;
};

TEST_F(SC25SimpleTest, MinimalTest)
{
  std::cout << "\n🚛 === SC25 MINIMAL TEST ===" << std::endl;
  
  // Minimal test with hardcoded data (exactly like soft_time_windows_test.cu)
  const int n_locations = 6;  // depot + 5 orders
  const int n_vehicles = 2;
  const int n_orders = 5;
  
  std::cout << "📊 CONFIGURACIÓN MINIMAL:" << std::endl;
  std::cout << "- Órdenes: " << n_orders << std::endl;
  std::cout << "- Ubicaciones: " << n_locations << std::endl;
  std::cout << "- Vehículos: " << n_vehicles << std::endl;
  
  // Hardcoded travel times matrix (same as soft_time_windows_test.cu)
  std::vector<std::vector<double>> travel_times = {
    {0, 10, 10, 10, 10, 10},  // From depot
    {10, 0, 10, 20, 30, 40},  // From order 0
    {20, 10, 0, 10, 20, 30},  // From order 1
    {30, 20, 10, 0, 10, 20},  // From order 2
    {40, 30, 20, 10, 0, 10},  // From order 3
    {50, 40, 30, 20, 10, 0}   // From order 4
  };
  
  // Setup data model using internal API (exactly like soft_time_windows_test.cu)
  std::cout << "🔧 Creando data model..." << std::endl;
  cuopt::routing::data_model_view_t<int, float> data_model(
    handle.get(), n_locations, n_vehicles, n_orders);

  // Set up cost matrix from travel times (exactly like soft_time_windows_test.cu)
  std::cout << "🔄 Configurando matriz de costes..." << std::endl;
  std::vector<float> cost_matrix_data(n_locations * n_locations);
  for (int i = 0; i < n_locations; ++i) {
    for (int j = 0; j < n_locations; ++j) {
      cost_matrix_data[i * n_locations + j] = static_cast<float>(travel_times[i][j]);
    }
  }
  
  rmm::device_uvector<float> cost_matrix(n_locations * n_locations, handle->get_stream());
  raft::copy(cost_matrix.data(), cost_matrix_data.data(), 
             cost_matrix_data.size(), handle->get_stream());
  
  data_model.add_cost_matrix(cost_matrix.data(), 0);
  std::cout << "✅ Cost matrix configurada" << std::endl;

  // Set up order locations
  std::vector<int> order_locations = {1, 2, 3, 4, 5};
  rmm::device_uvector<int> d_order_locations(n_orders, handle->get_stream());
  raft::copy(d_order_locations.data(), order_locations.data(), 
             n_orders, handle->get_stream());
  
  data_model.set_order_locations(d_order_locations.data());
  std::cout << "✅ Order locations configuradas" << std::endl;

  // Set up time windows
  std::vector<int> earliest_times_int = {0, 0, 0, 0, 0};
  std::vector<int> latest_times_int = {5, 25, 5, 35, 5};
  
  rmm::device_uvector<int> d_earliest(n_orders, handle->get_stream());
  rmm::device_uvector<int> d_latest(n_orders, handle->get_stream());
  
  raft::copy(d_earliest.data(), earliest_times_int.data(), n_orders, handle->get_stream());
  raft::copy(d_latest.data(), latest_times_int.data(), n_orders, handle->get_stream());
  
  data_model.set_order_time_windows(d_earliest.data(), d_latest.data());
  std::cout << "✅ Time windows configuradas" << std::endl;

  // Set up service times (2 minutes per order)
  std::vector<int> service_times(n_orders, 2);  // 2 minutes service time for each order
  rmm::device_uvector<int> d_service_times(n_orders, handle->get_stream());
  raft::copy(d_service_times.data(), service_times.data(), n_orders, handle->get_stream());
  
  data_model.set_order_service_times(d_service_times.data(), -1);  // -1 = default for all vehicles
  std::cout << "✅ Service times configurados" << std::endl;

  // Set up soft time windows
  std::vector<uint8_t> soft_tw_types_uint8 = {1, 0, 1, 0, 1};  // SOFT-STRICT-SOFT-STRICT-SOFT
  std::vector<float> soft_tw_penalties_float = {100.0f, 0.0f, 200.0f, 0.0f, 150.0f};
  
  rmm::device_uvector<uint8_t> d_types(n_orders, handle->get_stream());
  rmm::device_uvector<float> d_penalties(n_orders, handle->get_stream());
  
  raft::copy(d_types.data(), soft_tw_types_uint8.data(), n_orders, handle->get_stream());
  raft::copy(d_penalties.data(), soft_tw_penalties_float.data(), n_orders, handle->get_stream());
  
  data_model.set_soft_time_windows(d_types.data(), d_penalties.data());
  std::cout << "✅ Soft time windows configuradas" << std::endl;

  // Set up vehicle capacities (both vehicles can handle all orders)
  std::vector<int> order_demands(n_orders, 1);  // Each order demands 1 unit
  std::vector<int> vehicle_capacities(n_vehicles, 10);  // Each vehicle can carry 10 units

  rmm::device_uvector<int> d_demands(n_orders, handle->get_stream());
  rmm::device_uvector<int> d_capacities(n_vehicles, handle->get_stream());
  
  raft::copy(d_demands.data(), order_demands.data(), n_orders, handle->get_stream());
  raft::copy(d_capacities.data(), vehicle_capacities.data(), n_vehicles, handle->get_stream());
  
  data_model.add_capacity_dimension("capacity", d_demands.data(), d_capacities.data());
  std::cout << "✅ Capacidades configuradas" << std::endl;

  // Configure objectives (like soft_time_windows_test.cu)
  std::vector<cuopt::routing::objective_t> objectives = {
    cuopt::routing::objective_t::COST,
    cuopt::routing::objective_t::SOFT_TIME_WINDOW_PENALTY
  };
  std::vector<float> objective_weights = {1.0f, 1000.0f}; // High penalty for soft violations
  
  rmm::device_uvector<cuopt::routing::objective_t> d_objectives(2, handle->get_stream());
  rmm::device_uvector<float> d_obj_weights(2, handle->get_stream());
  raft::copy(d_objectives.data(), objectives.data(), 2, handle->get_stream());
  raft::copy(d_obj_weights.data(), objective_weights.data(), 2, handle->get_stream());
  
  data_model.set_objective_function(d_objectives.data(), d_obj_weights.data(), 2);
  std::cout << "✅ Objetivos configurados" << std::endl;

  // Solver configuration
  std::cout << "⚙️ Configurando solver..." << std::endl;
  cuopt::routing::solver_settings_t<int, float> solver_settings;
  solver_settings.set_time_limit(30);
  solver_settings.set_verbose_mode(true);
  
  // Solve
  std::cout << "🚀 Ejecutando solver..." << std::endl;
  auto solution = cuopt::routing::solve(data_model, solver_settings);
  
  if (solution.get_status() == cuopt::routing::solution_status_t::SUCCESS) {
    std::cout << "✅ Solver succeeded!" << std::endl;
    
    // Get and display solution info (like soft_time_windows_test.cu)
    std::cout << "Solution status: " << solution.get_status_string() << std::endl;
    std::cout << "Total objective: " << solution.get_total_objective() << std::endl;
    std::cout << "Vehicle count: " << solution.get_vehicle_count() << std::endl;
    
    // Get objective breakdown
    auto objective_values = solution.get_objectives();
    auto soft_penalty = objective_values.find(cuopt::routing::objective_t::SOFT_TIME_WINDOW_PENALTY);
    auto travel_cost = objective_values.find(cuopt::routing::objective_t::COST);
    
    if (travel_cost != objective_values.end()) {
      std::cout << "Travel cost: " << travel_cost->second << std::endl;
    }
    
    if (soft_penalty != objective_values.end()) {
      std::cout << "Soft time window penalty: " << soft_penalty->second << std::endl;
    }
    
    // Get route details
    auto& routes = solution.get_route();
    std::vector<int> h_routes(routes.size());
    raft::copy(h_routes.data(), routes.data(), routes.size(), handle->get_stream());
    handle->sync_stream();
    
    std::cout << "📋 Route: ";
    for (size_t i = 0; i < h_routes.size(); i++) {
      std::cout << h_routes[i];
      if (i < h_routes.size() - 1) std::cout << " → ";
    }
    std::cout << std::endl;
    
  } else {
    std::cout << "❌ Solver failed with status: " << static_cast<int>(solution.get_status()) << std::endl;
    FAIL() << "Solver should succeed for minimal test";
  }
}
