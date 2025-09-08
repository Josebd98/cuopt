// sc25_test.cu
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
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <iomanip>

class SC25Test : public ::testing::Test {
protected:
  void SetUp() override {
    // AISLAMIENTO COMPLETO: Crear contexto limpio
    std::cout << "🔄 SC25Test: Reseteando dispositivo CUDA..." << std::endl;
    cudaDeviceSynchronize();
    cudaDeviceReset();  // CRÍTICO: Reset completo del dispositivo
    
    handle = std::make_unique<raft::handle_t>();
    std::cout << "✅ SC25Test: Contexto CUDA limpio creado" << std::endl;
  }
  
  void TearDown() override {
    // Limpieza exhaustiva al finalizar
    if (handle) {
      handle->sync_stream();
      handle.reset();
    }
    cudaDeviceSynchronize();
    std::cout << "✅ SC25Test: Contexto limpiado" << std::endl;
  }
  
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

TEST_F(SC25Test, Turno2_Completo_ConTransitTime_DebugDump)
{
  std::cout << "🚛 === SC25 MINIMAL-LIKE TEST (desde CSV) ===\n";

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

  // 2) Preparar ids y atributos
  std::vector<std::string> order_ids;
  std::vector<int> earliest, latest, service, demand;
  std::vector<uint8_t> soft_type;
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
    demand.push_back(v.size()>11 ? std::stoi(v[11]) : 0);
    earliest.push_back(v.size()>12 ? time_to_minutes(v[12]) : 0);
    latest.push_back(v.size()>13 ? time_to_minutes(v[13]) : 24*60);
    service.push_back(v.size()>14 ? std::stoi(v[14]) : 0);

    const std::string pr = (v.size()>17 ? v[17] : "");
    if (pr=="A"||pr=="B") { soft_type.push_back(0); soft_pen.push_back(0.f); }
    else { soft_type.push_back(1); soft_pen.push_back(100.f); }
  }

  // Sanitizar ventanas (nunca latest < earliest, ancho >= 1)
  for (size_t i=0;i<earliest.size();++i) {
    if (latest[i] < earliest[i]) std::swap(latest[i], earliest[i]);
    if (latest[i] == earliest[i]) latest[i] = earliest[i] + 1;
  }

  int n_orders = std::min(20, static_cast<int>(order_ids.size()));  // ESCALAR a 20 órdenes
  int n_vehicles = 3;  // 3 vehículos para 20 órdenes
  int n_locations = n_orders + 1; // depot + orders
  std::cout << "📊 Órdenes turno2: " << n_orders
            << " | Vehículos: " << n_vehicles
            << " | Ubicaciones: " << n_locations << "\n";

  // 3) TEMPORAL: Usar matriz hardcodeada como sc25_harcodedtest.cu
  std::cout << "⚠️ USANDO MATRIZ HARDCODEADA para debug\n";
  
  const float BIG = 1e6f; // Para compatibilidad con el código existente
  
  // Map node_id->loc (para compatibilidad)
  std::map<std::string,int> node2loc;
  node2loc["SC25"]=0; node2loc["DEPOT"]=0; node2loc["SC25_DEPOT"]=0;
  for (int i=0;i<n_orders;++i) node2loc[order_ids[i]] = i+1;

  // MATRIZ HARDCODEADA escalada para 21x21 (depot + 20 órdenes)
  std::vector<std::vector<double>> travel_times(n_locations, std::vector<double>(n_locations));
  
  // Generar matriz simétrica con distancias razonables
  for (int i = 0; i < n_locations; i++) {
    for (int j = 0; j < n_locations; j++) {
      if (i == j) {
        travel_times[i][j] = 0.0;  // Distancia a sí mismo = 0
      } else {
        // Distancia proporcional a la diferencia de índices + algo de variación
        double base_distance = std::abs(i - j) * 10.0;  // 10 min por "salto"
        double variation = (i + j) % 5;  // Variación 0-4 min
        travel_times[i][j] = base_distance + variation;
        travel_times[j][i] = travel_times[i][j];  // Simétrica
      }
    }
  }
  
  std::vector<std::vector<float>> timeM(n_locations, std::vector<float>(n_locations));
  std::vector<std::vector<float>> costM(n_locations, std::vector<float>(n_locations));
  
  for (int i=0; i<n_locations; ++i) {
    for (int j=0; j<n_locations; ++j) {
      timeM[i][j] = static_cast<float>(travel_times[i][j]);
      costM[i][j] = static_cast<float>(travel_times[i][j]);  // cost = time
    }
  }

  std::cout << "✅ Matriz hardcodeada configurada: " << n_locations << "x" << n_locations << "\n";

  // 3.a) Check cobertura matriz + ejemplos de huecos
  int missing = 0;
  std::vector<std::pair<int,int>> missing_examples;
  for (int i=0;i<n_locations;++i){
    for (int j=0;j<n_locations;++j){
      if (!(std::isfinite(timeM[i][j]) && timeM[i][j] < BIG)) {
        missing++;
        if (missing_examples.size()<20) missing_examples.emplace_back(i,j);
      }
    }
  }
  std::cout << "🔎 Arcos faltantes (tiempo >= BIG): " << missing
            << " de " << (n_locations*n_locations) << "\n";
  for (auto& p: missing_examples) {
    int i=p.first, j=p.second;
    auto id_i = (i==0? std::string("SC25"): order_ids[i-1]);
    auto id_j = (j==0? std::string("SC25"): order_ids[j-1]);
    std::cout << "  ⚠️ Falta arco " << i << "→" << j << " (" << id_i << " → " << id_j << ")\n";
  }

  // 3.b) Eliminar nodos sin conectividad básica con SC25
  std::vector<int> keep_idx; keep_idx.reserve(n_orders);
  int dropped = 0;
  for (int i=0;i<n_orders;++i){
    int loc = i+1;
    bool dep_to = (timeM[0][loc] < BIG);
    bool to_dep = (timeM[loc][0] < BIG);
    if (!dep_to || !to_dep) {
      std::cout << "⚠️ Desechando nodo sin conectividad con depósito: " << order_ids[i]
                << " (SC25→node=" << dep_to << ", node→SC25=" << to_dep << ")\n";
      ++dropped;
    } else keep_idx.push_back(i);
  }
  if (dropped>0) {
    std::cout << "ℹ️ Nodos descartados: " << dropped << "\n";
    auto compactS = [&](auto& vec){
      using T=typename std::decay<decltype(vec[0])>::type;
      std::vector<T> tmp; tmp.reserve(keep_idx.size());
      for (int k: keep_idx) tmp.push_back(vec[k]);
      vec.swap(tmp);
    };
    compactS(order_ids); compactS(earliest); compactS(latest);
    compactS(service); compactS(demand); compactS(soft_type); compactS(soft_pen);

    // rehacer node2loc y matrices
    n_orders = static_cast<int>(order_ids.size());
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

  // 3.c) Dump resumen de TW/servicio/demanda/soft (primeros 20)
  std::cout << std::fixed << std::setprecision(0);
  std::cout << "🧾 Primeros 20 pedidos:\n";
  for (int i=0;i<std::min(20, n_orders); ++i) {
    std::cout << "  [" << i << "] id=" << order_ids[i]
              << " | tw=[" << earliest[i] << "," << latest[i] << "]"
              << " | srv=" << service[i]
              << " | dem=" << demand[i]
              << " | " << (soft_type[i] ? "SOFT" : "STRICT")
              << " (pen=" << soft_pen[i] << ")\n";
  }

  // 3.d) Dump submatriz 0..5 x 0..5 de tiempos (si alcanza)
  int K = std::min(6, n_locations);
  std::cout << "🧩 Submatriz tiempos (0.." << K-1 << "):\n";
  for (int i=0;i<K;++i){
    std::cout << "   ";
    for (int j=0;j<K;++j){
      float v = timeM[i][j];
      if (v >= BIG) std::cout << " BIG ";
      else std::cout << std::setw(4) << (int)v << " ";
    }
    std::cout << "\n";
  }

  // 4) Crear data model
  std::cout << "🔧 Creando data model...\n";
  cuopt::routing::data_model_view_t<int,float> dm(
      handle.get(), n_locations, n_vehicles, n_orders);

  // 5) Aplanar matrices y copiar a GPU
  std::vector<float> h_cost; h_cost.reserve(n_locations*n_locations);
  std::vector<float> h_time; h_time.reserve(n_locations*n_locations);
  float min_time=std::numeric_limits<float>::infinity(), max_time=0.f;
  float min_cost=std::numeric_limits<float>::infinity(), max_cost=0.f;
  int bad=0;
  for (int i=0;i<n_locations;++i){
    for (int j=0;j<n_locations;++j){
      float c = costM[i][j];
      float t = timeM[i][j];
      if (!std::isfinite(c)) { c = BIG; bad++; }
      if (!std::isfinite(t)) { t = BIG; bad++; }
      if (i==j) { c=0.f; t=0.f; }
      min_time = std::min(min_time, t); max_time = std::max(max_time, t);
      min_cost = std::min(min_cost, c); max_cost = std::max(max_cost, c);
      h_cost.push_back(c);
      h_time.push_back(t);
    }
  }
  std::cout << "📐 Matriz tiempo: min=" << min_time << " max=" << max_time
            << " | Matriz coste: min=" << min_cost << " max=" << max_cost
            << " | no-finitos reparados=" << bad << "\n";

  rmm::device_uvector<float> d_cost(h_cost.size(), handle->get_stream());
  rmm::device_uvector<float> d_time(h_time.size(), handle->get_stream());
  raft::copy(d_cost.data(), h_cost.data(), h_cost.size(), handle->get_stream());
  raft::copy(d_time.data(), h_time.data(), h_time.size(), handle->get_stream());

  dm.add_cost_matrix(d_cost.data(), 0);
  dm.add_transit_time_matrix(d_time.data(), 0);   // matriz de minutos para TW
  std::cout << "✅ Cost/Transit matrices configuradas\n";

  // 6) order_locations: **IMPORTANTE: 1..n_orders**
  std::vector<int> h_loc(n_orders); std::iota(h_loc.begin(), h_loc.end(), 1);
  rmm::device_uvector<int> d_loc(n_orders, handle->get_stream());
  raft::copy(d_loc.data(), h_loc.data(), n_orders, handle->get_stream());
  dm.set_order_locations(d_loc.data());
  std::cout << "✅ Order locations configuradas (1..n_orders)\n";

  // 7) TW tal cual vienen del CSV (ya saneadas)
  rmm::device_uvector<int> d_e(n_orders, handle->get_stream());
  rmm::device_uvector<int> d_l(n_orders, handle->get_stream());
  raft::copy(d_e.data(), earliest.data(), n_orders, handle->get_stream());
  raft::copy(d_l.data(), latest.data(),   n_orders, handle->get_stream());
  dm.set_order_time_windows(d_e.data(), d_l.data());
  std::cout << "✅ Time windows configuradas\n";

  // 8) Service times
  rmm::device_uvector<int> d_srv(n_orders, handle->get_stream());
  raft::copy(d_srv.data(), service.data(), n_orders, handle->get_stream());
  dm.set_order_service_times(d_srv.data(), -1);
  std::cout << "✅ Service times configurados\n";

  // 9) Soft/Strict
  rmm::device_uvector<uint8_t> d_soft(n_orders, handle->get_stream());
  rmm::device_uvector<float>   d_pen(n_orders, handle->get_stream());
  raft::copy(d_soft.data(), soft_type.data(), n_orders, handle->get_stream());
  raft::copy(d_pen.data(),  soft_pen.data(),  n_orders, handle->get_stream());
  dm.set_soft_time_windows(d_soft.data(), d_pen.data());
  
  // CRÍTICO: Forzar sincronización después de configurar soft time windows
  handle->sync_stream();
  std::cout << "✅ GPU sync forzado después de soft time windows\n";
  
  int strict_cnt=0, soft_cnt=0;
  for (auto s: soft_type) (s?soft_cnt:strict_cnt)++;
  std::cout << "✅ Soft/Strict configurados | STRICT="<<strict_cnt<<" | SOFT="<<soft_cnt<<"\n";

  // 10) Capacidades
  std::vector<int> veh_caps(n_vehicles, 120);
  rmm::device_uvector<int> d_caps(n_vehicles, handle->get_stream());
  rmm::device_uvector<int> d_dem (n_orders,   handle->get_stream());
  raft::copy(d_caps.data(), veh_caps.data(), n_vehicles, handle->get_stream());
  raft::copy(d_dem.data(),  demand.data(),   n_orders,   handle->get_stream());
  dm.add_capacity_dimension("capacity", d_dem.data(), d_caps.data());
  
  // CRÍTICO: Forzar sincronización final antes del solver
  handle->sync_stream();
  std::cout << "✅ Capacidades configuradas (veh=120)\n";
  std::cout << "✅ SYNC FINAL: Todos los datos GPU sincronizados\n";

  // 11) Objetivos
  std::vector<cuopt::routing::objective_t> objs = {
    cuopt::routing::objective_t::COST,
    cuopt::routing::objective_t::SOFT_TIME_WINDOW_PENALTY
  };
  std::vector<float> w = {1.f, 1.f};
  rmm::device_uvector<cuopt::routing::objective_t> d_objs(objs.size(), handle->get_stream());
  rmm::device_uvector<float> d_w(w.size(), handle->get_stream());
  raft::copy(d_objs.data(), objs.data(), objs.size(), handle->get_stream());
  raft::copy(d_w.data(), w.data(), w.size(), handle->get_stream());
  dm.set_objective_function(d_objs.data(), d_w.data(), (int)objs.size());
  std::cout << "✅ Objetivos configurados\n";

  // 12) Solver
  cuopt::routing::solver_settings_t<int,float> set;
  set.set_time_limit(30.0f);
  set.set_verbose_mode(true);

  std::cout << "🚀 Ejecutando solver...\n";
  auto sol = cuopt::routing::solve(dm, set);
  std::cout << "📌 Status: " << sol.get_status_string() << "\n";
  ASSERT_EQ(sol.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto obj = sol.get_objectives();
  auto itC = obj.find(cuopt::routing::objective_t::COST);
  auto itS = obj.find(cuopt::routing::objective_t::SOFT_TIME_WINDOW_PENALTY);
  std::cout << "Coste: " << (itC!=obj.end()? itC->second : 0.0) << "\n";
  std::cout << "Soft penalty: " << (itS!=obj.end()? itS->second : 0.0) << "\n";

  // Dump de la ruta (ids de location visitados)
  auto& routes = sol.get_route();
  std::vector<int> h_routes(routes.size());
  raft::copy(h_routes.data(), routes.data(), routes.size(), handle->get_stream());
  handle->sync_stream();
  std::cout << "🗺️  Ruta: ";
  for (size_t i=0;i<h_routes.size();++i) {
    int loc = h_routes[i];
    if (loc==0) std::cout << "SC25";
    else std::cout << order_ids[loc-1];
    if (i+1<h_routes.size()) std::cout << " → ";
  }
  std::cout << "\n";
}
