# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import pytest
from fastapi.testclient import TestClient

from cuopt_server.cuopt_server.main import app


client = TestClient(app)

# Base valid data for testing
valid_data = {
    "fleet_data": {
        "vehicle_locations": [[0, 0], [0, 0]],
        "capacities": [[2, 2], [4, 1]],
        "vehicle_ids": ["veh-1", "veh-2"],
    },
    "task_data": {
        "task_locations": [1, 2, 3, 4],
        "demand": [[1, 1, 1, 1], [2, 1, 1, 2]],
        "task_time_windows": [[0, 100], [0, 100], [0, 100], [0, 100]],
    },
    "cost_matrix_data": {
        "data": {
            "1": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "2": [[0, 1, 1], [1, 0, 1], [1, 2, 0]],
        }
    },
}


def test_soft_time_windows_basic():
    """Test basic soft time window functionality"""
    test_data = copy.deepcopy(valid_data)
    
    # Add soft time window configuration
    test_data["task_data"]["task_time_window_types"] = ["strict", "soft", "strict", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 100.0, 0.0, 50.0]
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200


def test_soft_time_windows_with_objective():
    """Test soft time windows with penalty objective"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["strict", "soft", "strict", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 100.0, 0.0, 50.0]
    
    # Add soft time window penalty to objectives
    test_data["solver_config"] = {
        "time_limit": 5,
        "objectives": {
            "cost": 1.0,
            "soft_time_window_penalty": 1.0
        }
    }
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200


def test_invalid_time_window_types():
    """Test validation with invalid time window types"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["strict", "invalid", "strict", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 100.0, 0.0, 50.0]
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 400
    assert "Invalid time window type" in response.json()["error"]


def test_negative_penalties():
    """Test validation with negative penalties"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["strict", "soft", "strict", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [0.0, -100.0, 0.0, 50.0]
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 400
    assert "must be greater than or equal to 0" in response.json()["error"]


def test_mismatched_lengths():
    """Test validation with mismatched array lengths"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["strict", "soft", "strict"]  # 3 elements
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 100.0, 0.0, 50.0]  # 4 elements
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 400
    assert "must have the same length" in response.json()["error"]


def test_all_strict_time_windows():
    """Test that strict-only time windows work"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["strict", "strict", "strict", "strict"]
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 0.0, 0.0, 0.0]
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200


def test_all_soft_time_windows():
    """Test that soft-only time windows work"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["soft", "soft", "soft", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [100.0, 200.0, 150.0, 75.0]
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200


def test_backward_compatibility():
    """Test that existing code without time window types still works"""
    test_data = copy.deepcopy(valid_data)
    # Don't add the new fields - should work as strict time windows
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200


def test_partial_soft_configuration():
    """Test with only time window types but no penalties"""
    test_data = copy.deepcopy(valid_data)
    
    # Only provide types, no penalties
    test_data["task_data"]["task_time_window_types"] = ["strict", "soft", "strict", "soft"]
    
    response = client.post("/cuopt/request", json=test_data)
    # Should work since penalties are optional when types are provided
    assert response.status_code == 200


def test_only_penalties_no_types():
    """Test with only penalties but no types"""
    test_data = copy.deepcopy(valid_data)
    
    # Only provide penalties, no types
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 100.0, 0.0, 50.0]
    
    response = client.post("/cuopt/request", json=test_data)
    # Should work since penalties are ignored without types
    assert response.status_code == 200


def test_comprehensive_example():
    """Test a comprehensive example with multiple features"""
    test_data = copy.deepcopy(valid_data)
    
    # Complex scenario with mixed time windows
    test_data["task_data"]["task_time_window_types"] = ["strict", "soft", "strict", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 200.0, 0.0, 100.0]
    
    # Add tight time windows to force violations
    test_data["task_data"]["task_time_windows"] = [[0, 100], [10, 15], [50, 60], [80, 85]]
    
    # Configure objectives to balance cost and penalties
    test_data["solver_config"] = {
        "time_limit": 10,
        "objectives": {
            "cost": 1.0,
            "soft_time_window_penalty": 0.5  # Lower weight for penalties
        }
    }
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200
    
    # Verify solution structure
    result = response.json()
    assert "solution" in result
    assert "routes" in result["solution"]


def test_high_penalty_values():
    """Test with very high penalty values"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["soft", "soft", "soft", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [1000.0, 2000.0, 1500.0, 3000.0]
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200


def test_zero_penalties():
    """Test with zero penalties for soft time windows"""
    test_data = copy.deepcopy(valid_data)
    
    test_data["task_data"]["task_time_window_types"] = ["soft", "soft", "soft", "soft"]
    test_data["task_data"]["task_time_window_penalties"] = [0.0, 0.0, 0.0, 0.0]
    
    response = client.post("/cuopt/request", json=test_data)
    assert response.status_code == 200
