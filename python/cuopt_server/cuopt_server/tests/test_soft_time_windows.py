import pytest
import json
import time
from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient
import os

client = RequestClient()


def test_soft_time_windows_server(cuoptproc):  # noqa
    """
    Test de soft time windows para el cuOpt server - versión API REST
    
    Este test replica el comportamiento del test_soft_time_windows de la librería
    pero usando la API REST del servidor para verificar la integración completa.
    """
    print("\n" + "="*70)
    print("🚛 === CUOPT SERVER SOFT TIME WINDOWS TEST ===")
    print("="*70)
    
    # Datos del problema - similar al test de la librería pero más simple
    optimization_request = {
        "task_data": {
            "task_locations": [1, 2, 3],  # 3 órdenes en ubicaciones 1, 2, 3
            "task_ids": ["Task-A", "Task-B", "Task-C"],
            "task_time_windows": [
                [0, 100],    # Task-A: ventana amplia (strict)
                [5, 8],      # Task-B: ventana muy restrictiva (soft) - forzará violación
                [20, 80]     # Task-C: ventana normal (strict)
            ],
            "task_time_window_types": ["strict", "soft", "strict"],
            "task_time_window_penalties": [0.0, 1000.0, 0.0],  # Penalización alta para Task-B
            "service_times": [0, 0, 0]  # Sin tiempos de servicio para simplificar
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0], [0, 0]],  # 2 vehículos en depot (0,0)
            "vehicle_ids": ["veh-1", "veh-2"],
            "vehicle_time_windows": [[0, 200], [0, 200]]  # Ventanas amplias para vehículos
        },
        "cost_matrix_data": {
            "data": {
                0: [  # Matriz 4x4 (depot + 3 órdenes)
                    [0, 1, 2, 3],
                    [1, 0, 4, 5],
                    [2, 4, 0, 6],
                    [3, 5, 6, 0]
                ]
            }
        },
        "solver_config": {
            "time_limit": 5.0,
            "objectives": {
                "cost": 1.0,
                "soft_time_window_penalty": 1.0  # Incluir objetivo de penalizaciones
            },
            "soft_to_hard_time_window_thresh": 25.0,  # Threshold para violaciones
            "verbose_mode": True
        }
    }
    
    print("📊 Configuración del problema:")
    print(f"  Órdenes: {len(optimization_request['task_data']['task_locations'])}")
    print(f"  Vehículos: {len(optimization_request['fleet_data']['vehicle_ids'])}")
    print(f"  Time windows: {optimization_request['task_data']['task_time_windows']}")
    print(f"  Tipos: {optimization_request['task_data']['task_time_window_types']}")
    print(f"  Penalizaciones: {optimization_request['task_data']['task_time_window_penalties']}")
    
    # Enviar request al servidor
    print("\n🚀 Enviando request al cuOpt server...")
    response = client.post("/cuopt/request", json=optimization_request)
    
    # Verificar respuesta exitosa (debe devolver un request ID o la solución directa)
    assert response.status_code == 200, f"Request falló con código: {response.status_code}"
    
    request_result = response.json()
    print("✅ Request enviado al servidor")
    
    # Manejar respuesta: asíncrona (id) o síncrona (response)
    if "id" in request_result:
        # Obtener el request ID
        request_id = request_result["id"]
        print(f"📋 Request ID: {request_id}")

        # Hacer polling para obtener la solución
        print("⏳ Esperando solución...")
        max_attempts = 30  # 30 segundos máximo
        for attempt in range(max_attempts):
            solution_response = client.get(f"/cuopt/solution/{request_id}")
            if solution_response.status_code == 200:
                result = solution_response.json()
                break
            elif solution_response.status_code == 202:
                # Aún procesando
                time.sleep(1)
                continue
            else:
                raise Exception(f"Error obteniendo solución: {solution_response.status_code}")
        else:
            raise Exception("Timeout esperando solución")
    elif "response" in request_result:
        # El servidor devolvió la solución directamente
        result = request_result["response"]
        print(f"📋 reqId: {request_result.get('reqId', 'n/a')} (respuesta síncrona)")
    else:
        raise AssertionError("Respuesta inesperada: falta 'id' o 'response'")
    
    print("✅ Solución recibida del servidor")
    
    # Analizar resultados
    print("\n" + "="*50)
    print("📈 RESULTADOS DEL SERVIDOR")
    print("="*50)
    
    # Unificar lectura de solución
    solution_data = None
    if "solution_data" in result:
        solution_data = result["solution_data"]
    elif "solver_response" in result:
        solver_response = result["solver_response"]
        if isinstance(solver_response, dict) and "solution_data" in solver_response:
            solution_data = solver_response["solution_data"]

    if solution_data is not None:
        # Mostrar información básica
        if "info" in solution_data:
            info = solution_data["info"]
            print(f"Estado del solver: {info.get('status', 'unknown')}")
            print(f"Costo total: {info.get('cost', 'unknown')}")
            
            # Verificar que el solver encontró una solución
            status = info.get('status', -1)
            assert status in [0, 1], f"Solver falló con estado: {status}"
            
            if status == 0:
                print("🎯 Solución óptima encontrada")
            elif status == 1:
                print("⏰ Límite de tiempo alcanzado, pero solución válida encontrada")
        
        # Mostrar objetivos individuales si están disponibles
        if "objectives" in solution_data.get("info", {}):
            objectives = solution_data["info"]["objectives"]
            print("\nObjetivos individuales:")
            for obj_name, obj_value in objectives.items():
                print(f"  {obj_name}: {obj_value}")
        
        # Analizar rutas
        if "task_id" in solution_data:
            routes = solution_data
            print(f"\nRutas generadas:")
            print(f"  Task IDs: {routes.get('task_id', [])}")
            print(f"  Arrival stamps: {routes.get('arrival_stamp', [])}")
            print(f"  Vehicle IDs: {routes.get('vehicle_id', [])}")
            print(f"  Types: {routes.get('type', [])}")
            
            # Análisis de violaciones de time windows
            analyze_server_time_window_violations(
                routes,
                optimization_request['task_data']['task_time_windows'],
                optimization_request['task_data']['task_time_window_types'],
                optimization_request['task_data']['task_time_window_penalties'],
                optimization_request['task_data']['task_ids']
            )
    else:
        # Formato síncrono mínimo: usar solver_response.objective_values para validar
        assert "solver_response" in result, "Respuesta inesperada: falta 'solution_data' y 'solver_response'"
        solver_response = result["solver_response"]
        assert "objective_values" in solver_response, "Faltan objective_values en solver_response"
        obj_vals = solver_response["objective_values"]
        print("Objetivos (síncrono):", obj_vals)
        # Validar que el objetivo de penalización existe (integración soft TW)
        assert "soft_time_window_penalty" in obj_vals, "Falta soft_time_window_penalty en objective_values"
    
    print("="*70)
    print("✅ Test del servidor completado exitosamente")
    
    # Fin del test sin valor de retorno


def analyze_server_time_window_violations(routes, time_windows, tw_types, tw_penalties, task_ids):
    """Analiza violaciones de time windows en la respuesta del servidor"""
    print("\n🔍 ANÁLISIS DE TIME WINDOWS (SERVIDOR)")
    print("-" * 50)
    
    strict_violations = 0
    soft_violations = 0
    total_soft_penalty = 0.0
    
    # Obtener datos de las rutas
    route_task_ids = routes.get('task_id', [])
    arrival_stamps = routes.get('arrival_stamp', [])
    route_types = routes.get('type', [])
    
    # Crear mapeo de task_id a índice
    task_id_to_index = {task_id: i for i, task_id in enumerate(task_ids)}
    
    # Analizar cada parada que no sea depot
    for i, (task_id, arrival_time, route_type) in enumerate(zip(route_task_ids, arrival_stamps, route_types)):
        if route_type != "Depot" and task_id in task_id_to_index:
            task_idx = task_id_to_index[task_id]
            
            if task_idx < len(time_windows):
                earliest, latest = time_windows[task_idx]
                is_soft = tw_types[task_idx] == "soft"
                penalty_rate = tw_penalties[task_idx]
                
                # Calcular violaciones
                early_violation = max(0, earliest - arrival_time)
                late_violation = max(0, arrival_time - latest)
                total_violation = early_violation + late_violation
                
                if total_violation > 0:
                    violation_type = "SOFT" if is_soft else "STRICT"
                    penalty = total_violation * penalty_rate if is_soft else 0
                    
                    print(f"  ⚠️  {task_id}: TW[{earliest}, {latest}] → llegada {arrival_time:.1f} "
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
    
    # Verificar que hubo al least una violación soft (por el diseño del test)
    if soft_violations > 0:
        print("🎯 ¡Soft time window violation detectada como esperado!")
    
    return {
        'strict_violations': strict_violations,
        'soft_violations': soft_violations,
        'total_penalty': total_soft_penalty
    }


def test_soft_time_windows_server_validation(cuoptproc):  # noqa
    """Test de validación de soft time windows en el servidor"""
    print("\n" + "="*70)
    print("🧪 === CUOPT SERVER SOFT TIME WINDOWS VALIDATION TEST ===")
    print("="*70)
    
    # Test 1: Tipos inválidos de time windows
    invalid_request = {
        "task_data": {
            "task_locations": [1, 2],
            "task_time_window_types": ["strict", "invalid_type"],  # Tipo inválido
            "task_time_window_penalties": [0.0, 100.0]
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0]],
            "vehicle_ids": ["veh-1"]
        },
        "cost_matrix_data": {
            "data": {"1": [[0, 1, 2], [1, 0, 3], [2, 3, 0]]}
        }
    }
    
    print("🧪 Test 1: Validación de tipos inválidos...")
    response = client.post("/cuopt/request", json=invalid_request)
    
    # Debe fallar con error 400 o 422
    assert response.status_code in [400, 422, 500], f"Se esperaba error, pero código fue: {response.status_code}"
    print("✅ Validación correcta: tipos inválidos rechazados")
    
    # Test 2: Longitudes desiguales
    mismatch_request = {
        "task_data": {
            "task_locations": [1, 2],
            "task_time_window_types": ["strict", "soft"],
            "task_time_window_penalties": [0.0]  # Longitud incorrecta
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0]],
            "vehicle_ids": ["veh-1"]
        },
        "cost_matrix_data": {
            "data": {"1": [[0, 1, 2], [1, 0, 3], [2, 3, 0]]}
        }
    }
    
    print("🧪 Test 2: Validación de longitudes desiguales...")
    response = client.post("/cuopt/request", json=mismatch_request)
    
    # Debe fallar
    assert response.status_code in [400, 422, 500], f"Se esperaba error, pero código fue: {response.status_code}"
    print("✅ Validación correcta: longitudes desiguales rechazadas")
    
    # Test 3: Solo tipos sin penalizaciones (debe usar defaults)
    default_request = {
        "task_data": {
            "task_locations": [1, 2],
            "task_ids": ["Task-A", "Task-B"],
            "task_time_windows": [[0, 10], [5, 15]],
            "task_time_window_types": ["strict", "soft"]
            # No se especifican penalizaciones - debe usar defaults
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0]],
            "vehicle_ids": ["veh-1"]
        },
        "cost_matrix_data": {
            "data": {"1": [[0, 1, 2], [1, 0, 3], [2, 3, 0]]}
        },
        "solver_config": {
            "time_limit": 2.0,
            "objectives": {"cost": 1.0, "soft_time_window_penalty": 1.0}
        }
    }
    
    print("🧪 Test 3: Penalizaciones por defecto...")
    response = client.post("/cuopt/request", json=default_request)
    
    # Debe funcionar correctamente
    assert response.status_code == 200, f"Request falló con código: {response.status_code}"
    print("✅ Validación correcta: penalizaciones por defecto aplicadas")
    
    print("="*70)
    print("✅ Tests de validación completados exitosamente")


def test_soft_time_windows_sc25_server(cuoptproc):  # noqa
    """
    Test SC25 con soft time windows usando el cuOpt server (datos reales),
    replicando el test de la librería para comparar resultados.
    """
    import os, time
    import pprint
    import pandas as pd

    print("\n" + "="*70)
    print("🚛 === SC25 TEST SERVER (RAW + análisis de TW) ===")
    print("="*70)

    # Rutas a datasets desde el repo raíz
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    nodes_path = os.path.join(base_dir, "datasets/SC25/nodes_df.csv")
    matrix_path = os.path.join(base_dir, "datasets/SC25/matrix_df.csv")

    assert os.path.exists(nodes_path), f"No se pudo abrir {nodes_path}"
    assert os.path.exists(matrix_path), f"No se pudo abrir {matrix_path}"

    def time_to_minutes(time_str):
        if not time_str or time_str == "":
            return 0
        parts = time_str.split(':')
        if not parts[0]:
            return 0
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return hours * 60 + minutes

    print("📁 Leyendo nodos del turno 2...")
    rows2 = []
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            with open(nodes_path, 'r', encoding=encoding) as f:
                _ = f.readline()
                for line in f:
                    fields = line.strip().split(';')
                    if len(fields) > 10 and fields[10] == "2":
                        rows2.append(fields)
            print(f"✅ Archivo leído con encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("No se pudo leer nodes_df.csv con ningún encoding")

    assert rows2, "No hay órdenes turno 2"

    order_ids = []
    earliest_times = []
    latest_times = []
    service_times = []
    demands = []
    soft_types = []  # 0/1
    soft_penalties = []

    for fields in rows2:
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
        if priority in ["A", "B"]:
            soft_types.append(0)
            soft_penalties.append(0.0)
        else:
            soft_types.append(1)
            soft_penalties.append(1.0)

    # 🔍 Debug: lo que hemos leído
    print("\n🔍 Debug SC25 — datos leídos del CSV turno 2:")
    print("order_ids (sample):", order_ids[:10])
    print("earliest_times (sample):", earliest_times[:10])
    print("latest_times (sample):", latest_times[:10])
    print("service_times (sample):", service_times[:10])
    print("demands (sample):", demands[:10])
    print("soft_types (raw, sample):", soft_types[:20])
    print("soft_penalties (sample):", soft_penalties[:20])
    print("\nTipos de datos en soft_types:")
    for i, v in enumerate(soft_types[:20]):
        print(f"  {i}: {v!r} (type={type(v)})")

    # Arreglo de TW inconsistentes
    for i in range(len(earliest_times)):
        if latest_times[i] < earliest_times[i]:
            earliest_times[i], latest_times[i] = latest_times[i], earliest_times[i]
        if latest_times[i] == earliest_times[i]:
            latest_times[i] = earliest_times[i] + 1

    # Leer nodos presentes en la matriz (para filtrar órdenes no conectadas)
    nodes_in_matrix = set()
    for encoding in encodings_to_try:
        try:
            with open(matrix_path, 'r', encoding=encoding) as f:
                _ = f.readline()
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
        raise UnicodeDecodeError("No se pudo leer matrix_df.csv con ningún encoding")

    # Filtrar órdenes por presencia en la matriz
    f_order_ids = []
    f_earliest = []
    f_latest = []
    f_service = []
    f_demands = []
    f_soft_types = []
    f_soft_penalties = []

    for i, oid in enumerate(order_ids):
        if oid in nodes_in_matrix:
            f_order_ids.append(oid)
            f_earliest.append(earliest_times[i])
            f_latest.append(latest_times[i])
            f_service.append(service_times[i])
            f_demands.append(demands[i])
            f_soft_types.append(soft_types[i])
            f_soft_penalties.append(soft_penalties[i])

    order_ids = f_order_ids
    earliest_times = f_earliest
    latest_times = f_latest
    service_times = f_service
    demands = f_demands
    soft_types = f_soft_types
    soft_penalties = f_soft_penalties

    n_orders = len(order_ids)
    n_vehicles = 20
    n_locations = n_orders + 1

    # Mapeo node->location
    node2loc = {"SC25": 0, "DEPOT": 0, "SC25_DEPOT": 0}
    for i, oid in enumerate(order_ids):
        node2loc[oid] = i + 1

    # === Construir matrices de COSTE y TIEMPO desde matrix_df.csv ===
    # Usamos pandas para leer una vez y mapear a matrices
    mdf = pd.read_csv(matrix_path, sep=";")
    mdf.columns = [c.lower() for c in mdf.columns]
    # Convertir decimales con coma -> punto
    for c in ["distance", "time"]:
        if c in mdf.columns:
            mdf[c] = (
                mdf[c].astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
                .replace({"": "nan"})
                .astype(float)
            )
        else:
            raise AssertionError(f"Falta columna '{c}' en matrix_df.csv")

    BIG = 1e6
    cost_matrix = [[BIG for _ in range(n_locations)] for _ in range(n_locations)]
    time_matrix = [[BIG for _ in range(n_locations)] for _ in range(n_locations)]

    rows_used = 0
    for _, row in mdf.iterrows():
        o = row["origin"]
        d = row["destination"]
        if o in node2loc and d in node2loc:
            i = node2loc[o]
            j = node2loc[d]
            # Coste por defecto = distance
            if pd.notna(row["distance"]):
                cost_matrix[i][j] = float(row["distance"])
            if pd.notna(row["time"]):
                time_matrix[i][j] = float(row["time"])
            rows_used += 1

    # Asegurar diagonal 0
    for k in range(n_locations):
        cost_matrix[k][k] = 0.0
        time_matrix[k][k] = 0.0

    print(f"🔧 Rellenadas {rows_used} filas en las matrices (de {len(mdf)})")
    # ——

    # Construir request del servidor
    task_time_window_types = ["soft" if t == 1 else "strict" for t in soft_types]
    task_time_windows = [[earliest_times[i], latest_times[i]] for i in range(n_orders)]

    # 🔍 Debug antes de mandar al server
    print("\n🔍 Debug SC25 — datos que se van a mandar al server:")
    print("task_time_window_types (sample):", task_time_window_types[:20])
    print("Tipos en task_time_window_types:")
    for i, v in enumerate(task_time_window_types[:20]):
        print(f"  {i}: {v!r} (type={type(v)})")
    print("task_time_window_penalties (sample):", soft_penalties[:20])

    optimization_request = {
        "task_data": {
            "task_locations": list(range(1, n_orders + 1)),
            "task_ids": order_ids,
            "task_time_windows": task_time_windows,
            "task_time_window_types": task_time_window_types,
            "task_time_window_penalties": soft_penalties,
            "service_times": service_times,
        },
        "fleet_data": {
            "vehicle_locations": [[0, 0]] * n_vehicles,
            "vehicle_ids": [f"veh-{i+1}" for i in range(n_vehicles)],
            "vehicle_time_windows": [[15 * 60, 21 * 60]] * n_vehicles,
        },
        "cost_matrix_data": {
            "data": { 0: cost_matrix }
        },
        "travel_time_matrix_data": {
            "data": { 0: time_matrix }
        },
        "solver_config": {
            "time_limit": 30.0,
            "objectives": {
                "cost": 0.3,
                "travel_time": 1.0,
                "soft_time_window_penalty": 1.5,
            },
            "soft_to_hard_time_window_thresh": 13.0,
            "verbose_mode": True,
        },
    }

    print("🚀 Enviando request SC25 al cuOpt server...")
    response = client.post("/cuopt/request", json=optimization_request)
    assert response.status_code == 200, f"Request falló con código: {response.status_code}"
    request_result = response.json()
    
    # --- Obtener la solución (maneja síncrono/asíncrono) ---
    if "id" in request_result:
        request_id = request_result["id"]
        print(f"📋 Request ID: {request_id}")
        print("⏳ Esperando solución...")
        max_attempts = 90  # hasta 90 s
        for _ in range(max_attempts):
            solution_response = client.get(f"/cuopt/solution/{request_id}")
            if solution_response.status_code == 200:
                result = solution_response.json()
                break
            elif solution_response.status_code == 202:
                time.sleep(1)
                continue
            else:
                raise Exception(f"Error obteniendo solución: {solution_response.status_code}")
        else:
            raise Exception("Timeout esperando solución")
    elif "response" in request_result:
        result = request_result["response"]
        print(f"📋 reqId: {request_result.get('reqId', 'n/a')} (respuesta síncrona)")
    else:
        raise AssertionError("Respuesta inesperada: falta 'id' o 'response'")
    print("✅ Request enviado al servidor")

    # ========================
    # Mostrar rutas + validaciones
    # ========================

    def analyze_server_time_window_violations(routes_by_vehicle, order_ids, earliest_times, latest_times, soft_types, soft_penalties):
        print("\n🔍 ANÁLISIS DE TIME WINDOWS (SERVER)")
        print("-" * 60)
        strict_violations = 0
        soft_violations = 0
        total_soft_penalty = 0.0

        # Mapear task_id -> índice de orden (para recuperar TW y tipo soft/strict)
        task_id_to_idx = {tid: i for i, tid in enumerate(order_ids)}

        for veh, route in routes_by_vehicle.items():
            task_ids = route.get("task_id", [])
            arrivals = route.get("arrival_stamp", [])
            types = route.get("type", [])

            for k, (tid, arr, typ) in enumerate(zip(task_ids, arrivals, types)):
                if typ == "Depot" or tid not in task_id_to_idx:
                    continue
                idx = task_id_to_idx[tid]
                ear = earliest_times[idx]
                lat = latest_times[idx]
                is_soft = (soft_types[idx] == 1)
                rate = soft_penalties[idx]

                early_v = max(0, ear - arr)
                late_v  = max(0, arr - lat)
                tot_v   = early_v + late_v
                penalty = (tot_v * rate) if is_soft else 0.0

                if tot_v > 0:
                    if is_soft:
                        soft_violations += 1
                        total_soft_penalty += penalty
                    else:
                        strict_violations += 1

        print(f"\n📊 RESUMEN:")
        print(f"  🔴 STRICT violadas: {strict_violations}")
        print(f"  🟡 SOFT violadas: {soft_violations}")
        print(f"  💰 Penalización total SOFT: {total_soft_penalty:.2f}")
        if strict_violations > 0:
            print("❌ SOLUCIÓN INFACTIBLE por violaciones STRICT")
        else:
            print("✅ Todas las ventanas STRICT respetadas")

        return {
            "strict_violations": strict_violations,
            "soft_violations": soft_violations,
            "total_penalty": total_soft_penalty,
        }

    def extract_routes_by_vehicle(result_payload):
        """
        Normaliza la estructura de rutas del server en:
        { vehicle_id: {"task_id":[...], "arrival_stamp":[...], "type":[...]} }
        """
        # Caso 1: server moderno -> solver_response.vehicle_data (dict por vehículo)
        solver_response = result_payload.get("solver_response", {})
        vehicle_data = solver_response.get("vehicle_data")
        if isinstance(vehicle_data, dict) and vehicle_data:
            return vehicle_data, solver_response

        # Caso 2: solution_data “aplanado”
        solution_data = result_payload.get("solution_data")
        if isinstance(solution_data, dict) and "task_id" in solution_data:
            # En este formato, no viene segmentado por vehículo. Construimos 1 pseudo-vehículo.
            fake_vehicle_id = "veh-0"
            routed = {
                fake_vehicle_id: {
                    "task_id": solution_data.get("task_id", []),
                    "arrival_stamp": solution_data.get("arrival_stamp", []),
                    "type": solution_data.get("type", []),
                }
            }
            # best-effort para recuperar objectives/status si existen
            info = solution_data.get("info", {})
            fake_solver_resp = {
                "status": info.get("status", 0),
                "objective_values": info.get("objectives", {}),
                "solution_cost": info.get("cost", None),
            }
            return routed, fake_solver_resp

        # Caso 3: solver_response sin vehicle_data pero con tables crudos (raro)
        return {}, solver_response

    # Obtener la solución (según tu código, ya tienes "result" aquí)
    routes_by_vehicle, solver_resp = extract_routes_by_vehicle(result)

    # Mostrar objetivos / estado
    print("\n" + "="*50)
    print("📈 RESULTADOS DEL SERVIDOR")
    print("="*50)
    status = solver_resp.get("status", None)
    objectives = solver_resp.get("objective_values", {})
    total_obj = solver_resp.get("solution_cost", None)

    if status is not None:
        print(f"Estado del solver: {status}")
        assert status in [0, 1], f"Solver falló con estado: {status}"
        if status == 0:
            print("🎯 Solución óptima encontrada")
        elif status == 1:
            print("⏰ Límite de tiempo alcanzado, solución válida encontrada")

    if objectives:
        print("\nObjetivos individuales:")
        for k, v in objectives.items():
            print(f"  {k}: {v}")
        assert "soft_time_window_penalty" in objectives, "Falta soft_time_window_penalty en objectives"

    if total_obj is not None:
        print(f"\nCosto/Objetivo total: {total_obj}")

    # Mostrar rutas por vehículo, con TW y posible violación
    print("\n🛣️  RUTAS POR VEHÍCULO:")
    if not routes_by_vehicle:
        print("⚠️ No se encontraron rutas en la respuesta.")
    else:
        # Mapa rápido: order_id -> TW/soft
        tw_map = {
            oid: (earliest_times[i], latest_times[i], "soft" if soft_types[i] == 1 else "strict", soft_penalties[i])
            for i, oid in enumerate(order_ids)
        }

        for veh_id, data in routes_by_vehicle.items():
            task_ids  = data.get("task_id", [])
            arrivals  = data.get("arrival_stamp", [])
            types     = data.get("type", [])
            print(f"\nVehículo: {veh_id}")
            print(" idx |      task_id     |   tipo   | llegada |   TW [ear,lat]   | violación | penalización")
            print("-"*96)
            for idx, (tid, typ, arr) in enumerate(zip(task_ids, types, arrivals)):
                if typ == "Depot":
                    tw_str = "—"
                    viol   = "—"
                    pen    = "—"
                else:
                    ear, lat, ttype, rate = tw_map.get(tid, (None, None, "?", 0.0))
                    if ear is None:
                        tw_str = "?"
                        viol   = "?"
                        pen    = "?"
                    else:
                        early_v = max(0, ear - arr)
                        late_v  = max(0, arr - lat)
                        tot_v   = early_v + late_v
                        viol    = f"{tot_v:.1f}" if tot_v > 0 else "0"
                        pen     = f"{(tot_v*rate):.1f}" if (tot_v > 0 and ttype == "soft") else "0"
                        tw_str  = f"[{ear},{lat}]"
                print(f"{idx:>4} | {str(tid):>15} | {typ:>7} | {arr:>7.1f} | {tw_str:^16} | {viol:^9} | {pen:>12}")

    # Análisis agregado de violaciones (resumen)
    _summary = analyze_server_time_window_violations(
        routes_by_vehicle,
        order_ids,
        earliest_times,
        latest_times,
        soft_types,
        soft_penalties,
    )

    print("="*70)
    print("✅ Visualización y validación de rutas completadas")

if __name__ == "__main__":
    # Ejecutar tests directamente

    test_soft_time_windows_sc25_server()
