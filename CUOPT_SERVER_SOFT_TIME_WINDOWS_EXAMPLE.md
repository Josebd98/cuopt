# Ejemplo Completo: Ventanas de Tiempo Blandas en cuOpt Server API

## Resumen

Este ejemplo muestra cómo usar las ventanas de tiempo estrictas y blandas a través de la API del servidor de cuOpt.

## Ejemplo de Payload Completo

```json
{
  "cost_matrix_data": {
    "data": {
      "0": [
        [0.0, 10.0, 15.0, 20.0, 25.0],
        [10.0, 0.0, 12.0, 18.0, 22.0],
        [15.0, 12.0, 0.0, 8.0, 16.0],
        [20.0, 18.0, 8.0, 0.0, 14.0],
        [25.0, 22.0, 16.0, 14.0, 0.0]
      ]
    }
  },
  "fleet_data": {
    "vehicle_locations": [[0, 0], [0, 0]],
    "vehicle_ids": ["truck_1", "truck_2"],
    "capacities": [[100], [120]]
  },
  "task_data": {
    "task_locations": [1, 2, 3, 4],
    "demand": [[10, 20, 15, 25]],
    "task_time_windows": [
      [0, 100],    // Depósito: horario amplio
      [30, 45],    // Cliente 1: ventana estrecha
      [60, 80],    // Cliente 2: ventana media
      [90, 110]    // Cliente 3: ventana flexible
    ],
    "task_time_window_types": [
      "strict",    // Depósito: siempre estricto
      "soft",      // Cliente 1: puede llegar tarde con penalización
      "strict",    // Cliente 2: horario crítico
      "soft"       // Cliente 3: flexible
    ],
    "task_time_window_penalties": [
      0.0,         // Depósito: sin penalización
      200.0,       // Cliente 1: penalización alta (cliente VIP)
      0.0,         // Cliente 2: sin penalización (estricto)
      50.0         // Cliente 3: penalización baja (flexible)
    ]
  },
  "solver_config": {
    "time_limit": 30,
    "objectives": {
      "cost": 1.0,
      "soft_time_window_penalty": 1.0
    }
  }
}
```

## Casos de Uso por Tipo de Cliente

### 1. Depósito (Índice 0)
```json
{
  "task_time_windows": [[0, 100]],
  "task_time_window_types": ["strict"],
  "task_time_window_penalties": [0.0]
}
```
- **Tipo**: Estricto
- **Razón**: Horarios de apertura/cierre no negociables
- **Penalización**: N/A

### 2. Cliente VIP (Índice 1)
```json
{
  "task_time_windows": [[30, 45]],
  "task_time_window_types": ["soft"],
  "task_time_window_penalties": [200.0]
}
```
- **Tipo**: Blando con penalización alta
- **Razón**: Cliente importante, preferible llegar a tiempo pero se permite flexibilidad costosa
- **Penalización**: 200 por minuto de retraso/adelanto

### 3. Cliente Crítico (Índice 2)
```json
{
  "task_time_windows": [[60, 80]],
  "task_time_window_types": ["strict"],
  "task_time_window_penalties": [0.0]
}
```
- **Tipo**: Estricto
- **Razón**: Entrega crítica (medicamentos, perecederos)
- **Penalización**: N/A

### 4. Cliente Flexible (Índice 3)
```json
{
  "task_time_windows": [[90, 110]],
  "task_time_window_types": ["soft"],
  "task_time_window_penalties": [50.0]
}
```
- **Tipo**: Blando con penalización baja
- **Razón**: Cliente comprensivo, acepta retrasos menores
- **Penalización**: 50 por minuto de retraso/adelanto

## Configuración de Objetivos

### Balanceado (Costo vs Penalizaciones)
```json
{
  "objectives": {
    "cost": 1.0,
    "soft_time_window_penalty": 1.0
  }
}
```

### Priorizar Costo
```json
{
  "objectives": {
    "cost": 1.0,
    "soft_time_window_penalty": 0.3
  }
}
```

### Priorizar Puntualidad
```json
{
  "objectives": {
    "cost": 0.5,
    "soft_time_window_penalty": 2.0
  }
}
```

## Ejemplo de Respuesta

```json
{
  "solution": {
    "routes": {
      "truck_1": [0, 1, 2, 0],
      "truck_2": [0, 3, 0]
    },
    "task_id": ["depot", "client_1", "client_2", "client_3"],
    "arrival_time": [
      [0, 35, 70, 0],      // truck_1: llegó 5 min tarde al cliente_1
      [0, 95, 0, 0]        // truck_2: llegó 5 min tarde al cliente_3
    ]
  },
  "metadata": {
    "total_cost": 85.0,
    "soft_time_window_penalties": {
      "client_1": 1000.0,    // 5 min × 200.0 = 1000
      "client_3": 250.0,     // 5 min × 50.0 = 250
      "total": 1250.0
    },
    "objective_breakdown": {
      "cost": 85.0,
      "soft_time_window_penalty": 1250.0,
      "total": 1335.0
    }
  }
}
```

## Ejemplos de Llamadas cURL

### Ejemplo Básico
```bash
curl -X POST "http://localhost:5000/cuopt/request" \
  -H "Content-Type: application/json" \
  -d '{
    "cost_matrix_data": {
      "data": {
        "0": [[0, 10, 15], [10, 0, 12], [15, 12, 0]]
      }
    },
    "fleet_data": {
      "vehicle_locations": [[0, 0]],
      "vehicle_ids": ["truck_1"],
      "capacities": [[100]]
    },
    "task_data": {
      "task_locations": [1, 2],
      "task_time_windows": [[0, 100], [30, 40]],
      "task_time_window_types": ["strict", "soft"],
      "task_time_window_penalties": [0.0, 150.0]
    },
    "solver_config": {
      "time_limit": 10,
      "objectives": {
        "cost": 1.0,
        "soft_time_window_penalty": 1.0
      }
    }
  }'
```

### Ejemplo con Múltiples Objetivos
```bash
curl -X POST "http://localhost:5000/cuopt/request" \
  -H "Content-Type: application/json" \
  -d '{
    "cost_matrix_data": {
      "data": {
        "0": [[0, 10, 15, 20], [10, 0, 12, 18], [15, 12, 0, 8], [20, 18, 8, 0]]
      }
    },
    "fleet_data": {
      "vehicle_locations": [[0, 0], [0, 0]],
      "vehicle_ids": ["truck_1", "truck_2"],
      "capacities": [[100], [120]]
    },
    "task_data": {
      "task_locations": [1, 2, 3],
      "demand": [[20, 30, 25]],
      "task_time_windows": [[0, 100], [20, 30], [50, 70]],
      "task_time_window_types": ["strict", "soft", "soft"],
      "task_time_window_penalties": [0.0, 200.0, 100.0]
    },
    "solver_config": {
      "time_limit": 30,
      "objectives": {
        "cost": 1.0,
        "travel_time": 0.5,
        "soft_time_window_penalty": 1.5
      }
    }
  }'
```

## Estrategias de Configuración

### 1. **Escenario Logístico Urbano**
- Depósitos: Estrictos (horarios fijos)
- Clientes comerciales: Blandos con penalización media
- Clientes residenciales: Blandos con penalización baja

### 2. **Escenario Médico/Farmacéutico**
- Hospitales: Estrictos (crítico)
- Farmacias: Blandos con penalización alta
- Consultorios: Blandos con penalización media

### 3. **Escenario de Construcción**
- Sitios activos: Estrictos (horarios de trabajo)
- Almacenes: Blandos con penalización baja
- Oficinas: Blandos con penalización media

## Validaciones de la API

La API valida automáticamente:
- ✅ Tipos de ventana válidos (`"strict"` o `"soft"`)
- ✅ Penalizaciones no negativas
- ✅ Longitudes consistentes entre arrays
- ✅ Compatibilidad con objetivos existentes

## Beneficios

1. **Flexibilidad Real**: Control granular por cliente/ubicación
2. **Optimización Inteligente**: Balance automático entre costo y puntualidad
3. **Compatibilidad Total**: Funciona con todas las características existentes
4. **Escalabilidad**: Maneja miles de ubicaciones eficientemente
5. **Fácil Integración**: API simple y consistente
