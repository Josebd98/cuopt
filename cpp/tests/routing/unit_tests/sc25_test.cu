// sc25_test_fixed_raw.cu
#include <gtest/gtest.h>

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/routing_structures.hpp>
#include <cuopt/routing/solve.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/copy.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>

class SC25Test : public ::testing::Test {
protected:
  void SetUp() override { handle = std::make_unique<raft::handle_t>(); }
  std::unique_ptr<raft::handle_t> handle;
};

static std::vector<std::string> split_semis(const std::string& s) {
  std::vector<std::string> out; std::stringstream ss(s); std::string x;
  while (std::getline(ss, x, ';')) out.push_back(x);
  return out;
}
static int time_to_minutes(const std::string& s) {
  if (s.empty()) return 0;
  std::stringstream ss(s); std::string H,M,S;
  std::getline(ss,H,':'); std::getline(ss,M,':'); std::getline(ss,S,':');
  if (H.empty()) return 0;
  return std::stoi(H)*60 + (M.empty()?0:std::stoi(M));
}
static float parse_float(std::string s) {
  auto l = s.find_first_not_of(" \t\r\n\"'");
  auto r = s.find_last_not_of(" \t\r\n\"'");
  if (l==std::string::npos) return 0.f;
  s = s.substr(l, r-l+1);
  if (s=="nan"||s=="NaN") return 0.f;
  for (char& c: s) if (c==',') c='.';
  std::string t; bool dot=false;
  for (char c: s) {
    if ((c>='0'&&c<='9')||c=='-'||c=='+') { t.push_back(c); continue; }
    if (c=='.' && !dot) { t.push_back(c); dot=true; }
  }
  if (t.empty()||t=="-"||t=="+") return 0.f;
  return std::strtof(t.c_str(), nullptr);
}

TEST_F(SC25Test, Turno2_Completo_ConTransitTime_RAW_DUMP)
{
  std::cout << "🚛 === SC25 TEST (RAW + análisis de TW) ===\n";

  // 1) Leer nodos turno 2
  std::ifstream fnodes("../../../../datasets/SC25/nodes_df.csv");
  ASSERT_TRUE(fnodes.is_open()) << "No se pudo abrir nodes_df.csv";
  std::string line;
  std::getline(fnodes, line); // header
  struct Row { std::vector<std::string> v; };
  std::vector<Row> rows2;
  while (std::getline(fnodes, line)) {
    auto f = split_semis(line);
    if (f.size()>10 && f[10]=="2") rows2.push_back({std::move(f)});
  }
  fnodes.close();
  ASSERT_FALSE(rows2.empty()) << "No hay órdenes turno 2";

  // 2) Atributos
  std::vector<std::string> order_ids;
  std::vector<int> earliest, latest, service, demand;
  std::vector<uint8_t> soft_type;   // 0=STRICT, 1=SOFT
  std::vector<float> soft_pen;
  order_ids.reserve(rows2.size());
  earliest.reserve(rows2.size());
  latest.reserve(rows2.size());
  service.reserve(rows2.size());
  demand.reserve(rows2.size());
  soft_type.reserve(rows2.size());
  soft_pen.reserve(rows2.size());

  for (auto& r : rows2) {
    const auto& v = r.v;
    // [1]=node_id, [11]=node_demand, [12]=tw_start, [13]=tw_end, [14]=service, [17]=priority
    order_ids.push_back(v[1]);
    demand .push_back(v.size()>11 ? std::stoi(v[11]) : 0);
    earliest.push_back(v.size()>12 ? time_to_minutes(v[12]) : 0);
    latest  .push_back(v.size()>13 ? time_to_minutes(v[13]) : 24*60);
    service .push_back(v.size()>14 ? std::stoi(v[14]) : 0);

    const std::string pr = (v.size()>17 ? v[17] : "");
    if (pr=="A"||pr=="B") { soft_type.push_back(0); soft_pen.push_back(0.f); }
    else                  { soft_type.push_back(1); soft_pen.push_back(1.f); }
  }
  for (size_t i=0;i<earliest.size();++i) {
    if (latest[i] < earliest[i]) std::swap(latest[i], earliest[i]);
    if (latest[i] == earliest[i]) latest[i] = earliest[i] + 1;
  }

  int n_orders    = (int)order_ids.size();
  int n_vehicles  = 20;
  int n_locations = n_orders + 1;

  // 3) Matriz
  std::ifstream fmat("../../../../datasets/SC25/matrix_df.csv");
  ASSERT_TRUE(fmat.is_open()) << "No se pudo abrir matrix_df.csv";
  std::getline(fmat, line); // header

  std::map<std::string,int> node2loc;
  node2loc["SC25"]=0; node2loc["DEPOT"]=0; node2loc["SC25_DEPOT"]=0;
  for (int i=0;i<n_orders;++i) node2loc[order_ids[i]] = i+1;

  const float BIG = 1e6f;
  std::vector<std::vector<float>> timeM(n_locations, std::vector<float>(n_locations, BIG));
  std::vector<std::vector<float>> costM(n_locations, std::vector<float>(n_locations, BIG));
  for (int i=0;i<n_locations;++i){ timeM[i][i]=0.f; costM[i][i]=0.f; }

  int lines=0, used=0;
  while (std::getline(fmat,line)) {
    ++lines;
    auto f = split_semis(line);
    if (f.size()<5) continue;
    auto itO = node2loc.find(f[0]);
    auto itD = node2loc.find(f[1]);
    if (itO==node2loc.end()||itD==node2loc.end()) continue;
    const int oi=itO->second, di=itD->second;

    float dist = parse_float(f[3]);
    float mins = parse_float(f[4]);
    if (!std::isfinite(dist) || dist<0) dist = BIG;
    if (!std::isfinite(mins) || mins<0) mins = BIG;

    if (oi==di){ dist=0.f; mins=0.f; }
    costM[oi][di] = dist;
    timeM[oi][di] = mins;
    ++used;
  }
  fmat.close();

  // 3.a) Conectividad básica depot <-> order
  std::vector<int> keep_idx; keep_idx.reserve(n_orders);
  for (int i=0;i<n_orders;++i){
    int loc = i+1;
    bool dep_to = (timeM[0][loc] < BIG);
    bool to_dep = (timeM[loc][0] < BIG);
    if (dep_to && to_dep) keep_idx.push_back(i);
  }
  if ((int)keep_idx.size() != n_orders) {
    auto compact = [&](auto& vec){
      using T = typename std::decay<decltype(vec[0])>::type;
      std::vector<T> tmp; tmp.reserve(keep_idx.size());
      for (int k: keep_idx) tmp.push_back(vec[k]);
      vec.swap(tmp);
    };
    compact(order_ids); compact(earliest); compact(latest);
    compact(service);   compact(demand);   compact(soft_type); compact(soft_pen);

    n_orders    = (int)order_ids.size();
    n_locations = n_orders + 1;

    std::map<std::string,int> node2loc2; node2loc2["SC25"]=0;
    for (int i=0;i<n_orders;++i) node2loc2[order_ids[i]] = i+1;

    std::vector<std::vector<float>> time2(n_locations, std::vector<float>(n_locations, BIG));
    std::vector<std::vector<float>> cost2(n_locations, std::vector<float>(n_locations, BIG));
    for (int i=0;i<n_locations;++i){ time2[i][i]=0.f; cost2[i][i]=0.f; }
    for (int i=0;i<n_orders;++i) {
      int oldi = keep_idx[i]+1;
      time2[0][i+1] = timeM[0][oldi];
      time2[i+1][0] = timeM[oldi][0];
      cost2[0][i+1] = costM[0][oldi];
      cost2[i+1][0] = costM[oldi][0];
    }
    for (int i=0;i<n_orders;++i) {
      int oldi = keep_idx[i]+1;
      for (int j=0;j<n_orders;++j) {
        int oldj = keep_idx[j]+1;
        time2[i+1][j+1] = timeM[oldi][oldj];
        cost2[i+1][j+1] = costM[oldi][oldj];
      }
    }
    node2loc.swap(node2loc2);
    timeM.swap(time2);
    costM.swap(cost2);
  }

  // 4) Data Model
  cuopt::routing::data_model_view_t<int,float> dm(
      handle.get(), n_locations, n_vehicles, n_orders);

  // 5) Matrices aplanadas
  std::vector<float> h_cost(n_locations*n_locations), h_time(n_locations*n_locations);
  for (int i=0;i<n_locations;++i){
    for (int j=0;j<n_locations;++j){
      float c = costM[i][j];
      float t = timeM[i][j];
      if (!std::isfinite(c)) c = BIG;
      if (!std::isfinite(t)) t = BIG;
      if (i==j) { c=0.f; t=0.f; }
      h_cost[i*n_locations+j] = c;
      h_time[i*n_locations+j] = t;
    }
  }

  rmm::device_uvector<float> d_cost(h_cost.size(), handle->get_stream());
  rmm::device_uvector<float> d_time(h_time.size(), handle->get_stream());
  raft::copy(d_cost.data(), h_cost.data(), h_cost.size(), handle->get_stream());
  raft::copy(d_time.data(), h_time.data(), h_time.size(), handle->get_stream());
  dm.add_cost_matrix(d_cost.data(), 0);
  dm.add_transit_time_matrix(d_time.data(), 0);

  // 6) order_locations: 1..n_orders
  std::vector<int> h_loc(n_orders); std::iota(h_loc.begin(), h_loc.end(), 1);
  rmm::device_uvector<int> d_loc(n_orders, handle->get_stream());
  raft::copy(d_loc.data(), h_loc.data(), n_orders, handle->get_stream());
  dm.set_order_locations(d_loc.data());

  // 7) TW
  rmm::device_uvector<int> d_e(n_orders, handle->get_stream());
  rmm::device_uvector<int> d_l(n_orders, handle->get_stream());
  raft::copy(d_e.data(), earliest.data(), n_orders, handle->get_stream());
  raft::copy(d_l.data(), latest.data(),   n_orders, handle->get_stream());
  dm.set_order_time_windows(d_e.data(), d_l.data());

  // 8) Service
  rmm::device_uvector<int> d_srv(n_orders, handle->get_stream());
  raft::copy(d_srv.data(), service.data(), n_orders, handle->get_stream());
  dm.set_order_service_times(d_srv.data(), -1);

  // 9) Soft/Strict
  rmm::device_uvector<uint8_t> d_soft(n_orders, handle->get_stream());
  rmm::device_uvector<float>   d_pen(n_orders, handle->get_stream());
  raft::copy(d_soft.data(), soft_type.data(), n_orders, handle->get_stream());
  raft::copy(d_pen.data(),  soft_pen.data(),  n_orders, handle->get_stream());
  dm.set_soft_time_windows(d_soft.data(), d_pen.data());

  // 10) Ventanas de tiempo para vehículos (turno de 15:00 a 21:00)
  std::vector<int> veh_earliest(n_vehicles, 15 * 60);  // 15:00 = 900 minutos
  std::vector<int> veh_latest(n_vehicles, 21 * 60);    // 21:00 = 1260 minutos
  rmm::device_uvector<int> d_veh_earliest(n_vehicles, handle->get_stream());
  rmm::device_uvector<int> d_veh_latest(n_vehicles, handle->get_stream());
  raft::copy(d_veh_earliest.data(), veh_earliest.data(), n_vehicles, handle->get_stream());
  raft::copy(d_veh_latest.data(), veh_latest.data(), n_vehicles, handle->get_stream());
  dm.set_vehicle_time_windows(d_veh_earliest.data(), d_veh_latest.data());

  // 11) Capacidad
  std::vector<int> veh_caps(n_vehicles, 120);
  rmm::device_uvector<int> d_caps(n_vehicles, handle->get_stream());
  rmm::device_uvector<int> d_dem (n_orders,   handle->get_stream());
  raft::copy(d_caps.data(), veh_caps.data(), n_vehicles, handle->get_stream());
  raft::copy(d_dem.data(),  demand.data(),   n_orders,   handle->get_stream());
  dm.add_capacity_dimension("capacity", d_dem.data(), d_caps.data());

  // 12) Objetivos
  std::vector<cuopt::routing::objective_t> objs = {
    cuopt::routing::objective_t::COST,
    cuopt::routing::objective_t::TRAVEL_TIME,
    cuopt::routing::objective_t::SOFT_TIME_WINDOW_PENALTY
  };
  std::vector<float> w = {1.f, 1.f, 1.f};
  rmm::device_uvector<cuopt::routing::objective_t> d_objs(objs.size(), handle->get_stream());
  rmm::device_uvector<float> d_w(w.size(), handle->get_stream());
  raft::copy(d_objs.data(), objs.data(), objs.size(), handle->get_stream());
  raft::copy(d_w.data(), w.data(), w.size(), handle->get_stream());
  dm.set_objective_function(d_objs.data(), d_w.data(), (int)objs.size());

  // 13) Solver
  cuopt::routing::solver_settings_t<int,float> set;
  set.set_time_limit(60.0f);
  set.set_soft_to_hard_time_window_thresh(25.0f);
  set.set_verbose_mode(true);

  std::cout << "🚛 Vehículos configurados con turno de 15:00 a 21:00 (900-1260 min)\n";
  std::cout << "🚀 Ejecutando solver...\n";
  auto sol = cuopt::routing::solve(dm, set);

  // ============================
  //   RAW SOLUTION DUMP
  // ============================
  std::cout << "\n===== RAW SOLUTION =====\n";
  std::cout << "status_string: " << sol.get_status_string() << "\n";
  std::cout << "total_objective: " << sol.get_total_objective() << "\n";

  auto objectives = sol.get_objectives();
  std::cout << "objectives_map_size: " << objectives.size() << "\n";
  for (auto &kv : objectives) {
    std::cout << "  objective_key_int=" << static_cast<int>(kv.first)
              << " value=" << kv.second << "\n";
  }

  auto route_host          = cuopt::host_copy(sol.get_route());
  auto node_types_host     = cuopt::host_copy(sol.get_node_types());
  auto truck_id_host       = cuopt::host_copy(sol.get_truck_id());
  auto order_locs_host     = cuopt::host_copy(sol.get_order_locations());
  auto arrival_host        = cuopt::host_copy(sol.get_arrival_stamp());

  auto dump_vec_int = [&](const char* name, const std::vector<int>& v){
    std::cout << name << " [size=" << v.size() << "]: ";
    for (size_t i=0;i<v.size();++i){ if (i) std::cout << ","; std::cout << v[i]; }
    std::cout << "\n";
  };
  auto dump_vec_dbl = [&](const char* name, const std::vector<double>& v){
    std::cout << name << " [size=" << v.size() << "]: ";
    for (size_t i=0;i<v.size();++i){ if (i) std::cout << ","; std::cout << v[i]; }
    std::cout << "\n";
  };

  dump_vec_int("route_raw",              route_host);
  dump_vec_int("node_types_raw",         node_types_host);
  dump_vec_int("truck_id_raw",           truck_id_host);
  dump_vec_int("order_locations_raw",    order_locs_host);
  dump_vec_dbl("arrival_stamp_raw",      arrival_host);

  // ============================
  //        ==== ANALYSIS ====
  // ============================
  std::cout << "\n===== TRANSLATED & TW CHECK =====\n";

  auto is_depot = [&](size_t i)->bool {
    return (i < node_types_host.size()) && (node_types_host[i] == (int)cuopt::routing::node_type_t::DEPOT);
  };

  // Agrupar por camión y por tour (separado por DEPOT)
  std::map<int, std::vector<std::vector<size_t>>> visits_by_vehicle; // indices de visita
  {
    std::map<int, std::vector<size_t>> current;
    auto flush = [&](int v){
      if (!current[v].empty()) {
        visits_by_vehicle[v].push_back(current[v]);
        current[v].clear();
      }
    };
    for (size_t i=0;i<route_host.size();++i) {
      int truck = (i < truck_id_host.size()) ? truck_id_host[i] : 0;
      if (is_depot(i)) { flush(truck); continue; }
      current[truck].push_back(i);
    }
    for (auto &kv : current) flush(kv.first);
  }

  long strict_cnt=0, soft_cnt=0;
  long long strict_minutes=0, soft_minutes=0;
  double soft_penalty_total=0.0;

  auto order_name = [&](int order_index)->std::string{
    if (order_index >=0 && order_index < (int)order_ids.size()) return order_ids[order_index];
    return std::string("ORDER?(") + std::to_string(order_index) + ")";
  };

  for (auto &kv : visits_by_vehicle) {
    int truck = kv.first;
    auto &tours = kv.second;
    for (size_t t=0; t<tours.size(); ++t) {
      std::cout << "🚛 Truck " << truck << " | Tour " << t << "\n";
      for (size_t pos=0; pos<tours[t].size(); ++pos) {
        size_t i = tours[t][pos];

        // Map robusto a order_index
        int loc = (i < order_locs_host.size()) ? order_locs_host[i] : 0;
        int order_index = -1;
        if (loc > 0) order_index = loc - 1;
        else {
          // fallback: algunos builds codifican order_index en route()
          int vid = (i < route_host.size() ? route_host[i] : -1);
          if (vid >= 0 && vid < n_orders) order_index = vid; // ORDER_0..ORDER_(n-1)
        }
        if (order_index < 0 || order_index >= n_orders) {
          std::cout << "  ⚠️  visita " << i << " no mapeable a order_index.\n";
          continue;
        }

        double arr = (i < arrival_host.size()) ? arrival_host[i] : std::numeric_limits<double>::quiet_NaN();
        int e = earliest[order_index];
        int l = latest[order_index];
        bool is_soft_tw = (soft_type[order_index] != 0);
        double late = (std::isfinite(arr) && arr > l) ? (arr - l) : 0.0;

        std::cout << "  • " << order_name(order_index)
                  << "  idx=" << order_index
                  << "  window=[" << e << "," << l << "]"
                  << "  type=" << (is_soft_tw ? "SOFT" : "STRICT")
                  << "  arrival=" << arr;

        if (late > 0.0) {
          std::cout << "  → LATE +" << late << " min";
          if (is_soft_tw) {
            soft_cnt++; soft_minutes += (long)std::llround(late);
            soft_penalty_total += late * soft_pen[order_index];
          } else {
            strict_cnt++; strict_minutes += (long)std::llround(late);
          }
        } else if (std::isfinite(arr) && arr < e) {
          std::cout << "  (early " << (e - arr) << " min)";
        } else {
          std::cout << "  ✓ OK";
        }
        std::cout << "\n";
      }
    }
  }

  std::cout << "\n📘 RESUMEN TW\n";
  std::cout << "  STRICT: violaciones=" << strict_cnt
            << " | minutos tarde=" << strict_minutes << "\n";
  std::cout << "  SOFT  : violaciones=" << soft_cnt
            << " | minutos tarde=" << soft_minutes
            << " | penalización total (calc)=" << soft_penalty_total << "\n";

  bool infeasible_time_dimension = (strict_cnt > 0);
  if (infeasible_time_dimension) {
    std::cout << "❌ INFEASIBLE por STRICT TW. Minutos totales fuera de ventana (STRICT): "
              << strict_minutes << "\n";
  } else {
    std::cout << "✅ Todos los STRICT respetados.\n";
  }

  SUCCEED() << "RAW + análisis TW ejecutado.";
}
