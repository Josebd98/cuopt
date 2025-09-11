import cudf
import pandas as pd
import numpy as np
from cuopt import routing
import os


def time_to_minutes(time_str):
    """Convierte tiempo HH:MM:SS a minutos desde medianoche - replica la función C++"""
    if not time_str or time_str == "":
        return 0
    
    parts = time_str.split(':')
    if not parts[0]:  # Si H está vacío
        return 0
    
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return hours * 60 + minutes


def test_soft_time_windows():
    """
    Test SC25 con soft time windows - versión Python del test C++
    
    Replica exactamente el comportamiento del sc25_test.cu:
    - Lee datos reales del dataset SC25 turno 2
    - Configura soft time windows basado en prioridades
    - Lee matriz de distancias real
    - Resuelve con penalizaciones de soft time windows
    """
    print("\n" + "="*70)
    print("🚛 === SC25 TEST PYTHON (RAW + análisis de TW) ===")
    print("="*70)
    
    # 1) Leer nodos turno 2 - exactamente como en C++
    nodes_path = "datasets/SC25/nodes_df.csv"
    matrix_path = "datasets/SC25/matrix_df.csv"
    
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"No se pudo abrir {nodes_path}")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"No se pudo abrir {matrix_path}")
    
    print("📁 Leyendo nodos del turno 2...")
    
    # Leer línea por línea como en C++ (no usar pandas para replicar exactamente)
    rows2 = []
    # Probar diferentes encodings para manejar caracteres especiales
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            with open(nodes_path, 'r', encoding=encoding) as f:
                header = f.readline()  # Saltar header
                for line in f:
                    fields = line.strip().split(';')
                    # Filtrar solo turno 2: campo [10] == "2"
                    if len(fields) > 10 and fields[10] == "2":
                        rows2.append(fields)
            print(f"✅ Archivo leído con encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("No se pudo leer el archivo con ningún encoding")
    
    if not rows2:
        raise ValueError("No hay órdenes turno 2")
    
    print(f"✅ Encontradas {len(rows2)} órdenes del turno 2")
    
    # 2) Extraer atributos - exactamente como en C++
    order_ids = []
    earliest_times = []
    latest_times = []
    service_times = []
    demands = []
    soft_types = []  # 0=STRICT, 1=SOFT
    soft_penalties = []
    
    for fields in rows2:
        # [1]=node_id, [11]=node_demand, [12]=tw_start, [13]=tw_end, [14]=service, [17]=priority
        order_id = fields[1]
        demand = int(fields[11]) if len(fields) > 11 and fields[11].isdigit() else 0
        earliest = time_to_minutes(fields[12]) if len(fields) > 12 else 0
        latest = time_to_minutes(fields[13]) if len(fields) > 13 else 24*60
        service = int(fields[14]) if len(fields) > 14 and fields[14].isdigit() else 0
        priority = fields[17] if len(fields) > 17 else ""
        
        order_ids.append(order_id)
        demands.append(demand)
        earliest_times.append(earliest)
        latest_times.append(latest)
        service_times.append(service)
        
        # Configurar soft/strict basado en prioridad (como en C++)
        if priority in ["A", "B"]:
            soft_types.append(0)      # STRICT para prioridades altas
            soft_penalties.append(0.0)
        else:
            soft_types.append(1)      # SOFT para otras prioridades  
            soft_penalties.append(1.0)
    
    # Corregir time windows inválidas (como en C++)
    for i in range(len(earliest_times)):
        if latest_times[i] < earliest_times[i]:
            earliest_times[i], latest_times[i] = latest_times[i], earliest_times[i]
        if latest_times[i] == earliest_times[i]:
            latest_times[i] = earliest_times[i] + 1
    
    n_orders = len(order_ids)
    n_vehicles = 20  # Como en el test C++
    n_locations = n_orders + 1  # Órdenes + depot
    
    print(f"\n📊 Configuración inicial:")
    print(f"  Órdenes totales: {len(rows2)}")
    print(f"  Vehículos: {n_vehicles}")
    
    print(f"\n📊 Configuración filtrada:")
    print(f"  Órdenes válidas: {n_orders}")
    print(f"  Localizaciones: {n_locations}")
    print(f"  Ventanas STRICT: {soft_types.count(0)}")
    print(f"  Ventanas SOFT: {soft_types.count(1)}")
    
    # 3) Leer matriz de distancias y filtrar nodos válidos
    print("🗺️  Leyendo matriz de distancias...")
    
    # Primero, identificar qué nodos están en la matriz
    nodes_in_matrix = set()
    for encoding in encodings_to_try:
        try:
            with open(matrix_path, 'r', encoding=encoding) as f:
                header = f.readline()  # Saltar header
                for line in f:
                    fields = line.strip().split(';')
                    if len(fields) >= 5:
                        origin = fields[0]
                        destination = fields[1]
                        nodes_in_matrix.add(origin)
                        nodes_in_matrix.add(destination)
            print(f"✅ Matriz leída con encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("No se pudo leer la matriz con ningún encoding")
    
    print(f"📍 Nodos encontrados en matriz: {len(nodes_in_matrix)}")
    
    # Filtrar solo órdenes que están en la matriz
    filtered_order_ids = []
    filtered_earliest_times = []
    filtered_latest_times = []
    filtered_service_times = []
    filtered_demands = []
    filtered_soft_types = []
    filtered_soft_penalties = []
    
    orders_filtered_out = 0
    for i, order_id in enumerate(order_ids):
        if order_id in nodes_in_matrix:
            filtered_order_ids.append(order_id)
            filtered_earliest_times.append(earliest_times[i])
            filtered_latest_times.append(latest_times[i])
            filtered_service_times.append(service_times[i])
            filtered_demands.append(demands[i])
            filtered_soft_types.append(soft_types[i])
            filtered_soft_penalties.append(soft_penalties[i])
        else:
            orders_filtered_out += 1
    
    print(f"⚠️  Órdenes filtradas (no en matriz): {orders_filtered_out}")
    print(f"✅ Órdenes válidas: {len(filtered_order_ids)}")
    
    # Usar las listas filtradas
    order_ids = filtered_order_ids
    earliest_times = filtered_earliest_times
    latest_times = filtered_latest_times
    service_times = filtered_service_times
    demands = filtered_demands
    soft_types = filtered_soft_types
    soft_penalties = filtered_soft_penalties
    
    # Recalcular dimensiones
    n_orders = len(order_ids)
    n_locations = n_orders + 1  # Órdenes válidas + depot
    
    # Crear mapeo node_id -> location_index (como en C++)
    node2loc = {}
    node2loc["SC25"] = 0
    node2loc["DEPOT"] = 0  
    node2loc["SC25_DEPOT"] = 0
    for i, order_id in enumerate(order_ids):
        node2loc[order_id] = i + 1
    
    # Inicializar matriz con valores grandes (como en C++)
    BIG = 1e6
    time_matrix = [[BIG for _ in range(n_locations)] for _ in range(n_locations)]
    
    # Leer matriz línea por línea OTRA VEZ para poblar la matriz
    for encoding in encodings_to_try:
        try:
            with open(matrix_path, 'r', encoding=encoding) as f:
                header = f.readline()  # Saltar header
                for line in f:
                    fields = line.strip().split(';')
                    if len(fields) >= 5:
                        origin = fields[0]
                        destination = fields[1] 
                        try:
                            time_value = float(fields[4].replace(',', '.'))  # Convertir coma a punto
                            
                            # Mapear a índices de matriz
                            if origin in node2loc and destination in node2loc:
                                i = node2loc[origin]
                                j = node2loc[destination]
                                time_matrix[i][j] = time_value
                        except (ValueError, IndexError):
                            continue
            break
        except UnicodeDecodeError:
            continue
    
    # Convertir a cuDF DataFrame
    cost_matrix = cudf.DataFrame(time_matrix, dtype="float32")
    
    # 4) Crear modelo de datos
    data_model = routing.DataModel(n_locations, n_vehicles, n_orders)
    data_model.add_cost_matrix(cost_matrix)
    
    # 6) Configurar localizaciones de órdenes (1, 2, ..., n_orders)
    order_locations = cudf.Series(range(1, n_orders + 1), dtype="int32")
    data_model.set_order_locations(order_locations)
    
    # 7) Configurar time windows
    earliest_series = cudf.Series(earliest_times, dtype="int32")
    latest_series = cudf.Series(latest_times, dtype="int32")
    data_model.set_order_time_windows(earliest_series, latest_series)
    
    # 8) Configurar soft time windows
    soft_types_series = cudf.Series(soft_types, dtype="uint8")
    soft_penalties_series = cudf.Series(soft_penalties, dtype="float32")
    data_model.set_soft_time_windows(soft_types_series, soft_penalties_series)
    
    # 9) Configurar tiempos de servicio
    service_series = cudf.Series(service_times, dtype="int32")
    data_model.set_order_service_times(service_series)
    
    # 10) Configurar time windows de vehículos (turno 15:00-21:00)
    vehicle_earliest = cudf.Series([15 * 60] * n_vehicles, dtype="int32")  # 15:00
    vehicle_latest = cudf.Series([21 * 60] * n_vehicles, dtype="int32")    # 21:00
    data_model.set_vehicle_time_windows(vehicle_earliest, vehicle_latest)
    
    # 11) Configurar capacidades
    vehicle_capacities = cudf.Series([120] * n_vehicles, dtype="int32")
    order_demands = cudf.Series(demands, dtype="int32")
    data_model.add_capacity_dimension("capacity", order_demands, vehicle_capacities)
    
    # 12) Configurar objetivos (COST + TRAVEL_TIME + SOFT_TIME_WINDOW_PENALTY)
    objectives = cudf.Series([
        routing.Objective.COST, 
        routing.Objective.TRAVEL_TIME,
        routing.Objective.SOFT_TIME_WINDOW_PENALTY
    ], dtype="int32")
    weights = cudf.Series([0.3, 1.0, 1.5], dtype="float32")
    data_model.set_objective_function(objectives, weights)
    
    # 13) Configurar solver
    settings = routing.SolverSettings()
    settings.set_time_limit(60.0)
    settings.set_soft_to_hard_time_window_thresh(25.0)
    settings.set_verbose_mode(True)
    
    print("🚀 Ejecutando solver...")
    solution = routing.Solve(data_model, settings)
    
    # 14) Analizar resultados
    print("\n" + "="*50)
    print("📈 RESULTADOS")
    print("="*50)
    
    print(f"Estado: {solution.get_status()}")
    print(f"Objetivo total: {solution.get_total_objective()}")
    
    # Mostrar objetivos individuales
    try:
        objectives_dict = solution.get_objective_values()
        print("Objetivos individuales:")
        for obj, value in objectives_dict.items():
            print(f"  {obj}: {value}")
    except Exception as e:
        print(f"Objetivos no disponibles: {e}")
    
    # Analizar rutas y violaciones de time windows
    routes_df = solution.get_route()
    print(f"\nRutas generadas:")
    print(routes_df.head(20))  # Mostrar primeras 20 filas
    
    # Análisis de violaciones de time windows
    analyze_time_window_violations(
        routes_df, earliest_times, latest_times, soft_types, 
        soft_penalties, order_ids
    )
    
    print("="*70)
    print("✅ Test SC25 completado exitosamente")
    
    # Verificar que se encontró una solución (estado 0 o 1 son válidos)
    assert solution.get_status() in [0, 1], f"Solver falló con estado: {solution.get_status()}"
    
    if solution.get_status() == 0:
        print("🎯 Solución óptima encontrada")
    elif solution.get_status() == 1:
        print("⏰ Límite de tiempo alcanzado, pero solución válida encontrada")


def analyze_time_window_violations(routes_df, earliest_times, latest_times, 
                                 soft_types, soft_penalties, order_ids):
    """Analiza violaciones de time windows en la solución - replica análisis del C++"""
    print("\n🔍 ANÁLISIS DE TIME WINDOWS")
    print("-" * 50)
    
    strict_violations = 0
    soft_violations = 0
    total_soft_penalty = 0.0
    
    # Filtrar solo entregas (no depot) y convertir a pandas para iteración
    delivery_routes = routes_df[routes_df['type'] != 'Depot'].copy().to_pandas()
    
    for idx, row in delivery_routes.iterrows():
        location = row['location'] 
        arrival_time = row['arrival_stamp']
        
        # Mapear location a order index (location 1 = order 0, etc.)
        order_idx = location - 1  # location 0 es depot
        
        if 0 <= order_idx < len(earliest_times):
            earliest = earliest_times[order_idx]
            latest = latest_times[order_idx]
            is_soft = soft_types[order_idx] == 1
            penalty_rate = soft_penalties[order_idx]
            order_id = order_ids[order_idx]
            
            # Calcular violaciones
            early_violation = max(0, earliest - arrival_time)
            late_violation = max(0, arrival_time - latest)
            total_violation = early_violation + late_violation
            
            if total_violation > 0:
                violation_type = "SOFT" if is_soft else "STRICT"
                penalty = total_violation * penalty_rate if is_soft else 0
                
                print(f"  ⚠️  {order_id}: TW[{earliest}, {latest}] → llegada {arrival_time:.1f} "
                      f"({violation_type}, violación: {total_violation:.1f}min, "
                      f"penalización: {penalty:.1f})")
                
                if is_soft:
                    soft_violations += 1
                    total_soft_penalty += penalty
                else:
                    strict_violations += 1
    
    print(f"\n📊 RESUMEN DE VIOLACIONES:")
    print(f"  🔴 STRICT violadas: {strict_violations}")
    print(f"  🟡 SOFT violadas: {soft_violations}")  
    print(f"  💰 Penalización total SOFT: {total_soft_penalty:.2f}")
    
    if strict_violations > 0:
        print("❌ SOLUCIÓN INFACTIBLE por violaciones STRICT")
    else:
        print("✅ Todas las ventanas STRICT respetadas")
    
    return {
        'strict_violations': strict_violations,
        'soft_violations': soft_violations, 
        'total_penalty': total_soft_penalty
    }


if __name__ == "__main__":
    test_soft_time_windows()
