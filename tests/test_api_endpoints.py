"""
Test FastAPI endpoints in backend/query_api.py

Brief tests for core API functionality:
- Health check endpoints
- Graph save/load operations  
- Trajectory computation
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.query_api import (
    get_status, 
    save_pylink_graph, 
    load_pylink_graph,
    compute_pylink_trajectory,
    list_pylink_graphs
)
from configs.appconfig import USER_DIR


def test_status_endpoint():
    """Test /status endpoint returns operational status"""
    result = get_status()
    assert result['status'] == 'operational', f"Status check failed: {result}"
    assert 'message' in result, "Missing message in status response"


def test_save_and_load_graph():
    """Test saving and loading a pylink graph"""
    # Load test data
    test_file = Path(__file__).parent / "4bar_test.json"
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Save the graph
    save_result = save_pylink_graph(test_data)
    assert save_result['status'] == 'success', f"Save failed: {save_result.get('message')}"
    assert 'filename' in save_result, "Missing filename in save response"
    filename = save_result['filename']
    
    # Load it back
    load_result = load_pylink_graph(filename=filename)
    assert load_result['status'] == 'success', f"Load failed: {load_result.get('message')}"
    assert 'data' in load_result, "Missing data in load response"
    
    # Verify structure matches (data is nested under 'pylinkage')
    loaded_data = load_result['data']
    assert 'pylinkage' in loaded_data, "Missing 'pylinkage' in loaded data"
    assert 'joints' in loaded_data['pylinkage'], "Missing 'joints' in pylinkage data"
    assert len(loaded_data['pylinkage']['joints']) == len(test_data['pylinkage']['joints']), \
        "Joint count mismatch"

def test_list_graphs():
    """Test listing saved pylink graphs"""
    result = list_pylink_graphs()
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result['status'] == 'success', f"List failed: {result.get('message')}"
    assert 'files' in result, "Missing 'files' in response"
    assert isinstance(result['files'], list), "Files should be a list"
    # Should have at least one graph from previous test
    assert len(result['files']) > 0, "Expected at least one saved graph"

def test_compute_trajectory():
    """Test trajectory computation endpoint"""
    # Load test data
    test_file = Path(__file__).parent / "4bar_test.json"
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Compute trajectory
    result = compute_pylink_trajectory(test_data)
    
    assert result['status'] == 'success', f"Trajectory computation failed: {result.get('message')}"
    assert 'trajectories' in result, "Missing trajectories in response"
    assert 'joint_types' in result, "Missing joint_types in response"
    assert result['n_steps'] > 0, "No steps computed"
    assert len(result['trajectories']) > 0, "No trajectories generated"


def test_invalid_graph_data():
    """Test handling of invalid graph data"""
    invalid_data = {"invalid": "data"}
    
    result = compute_pylink_trajectory(invalid_data)
    assert result['status'] == 'error', \
        f"Expected error for invalid data, got: {result.get('status')}"


def test_load_nonexistent_graph():
    """Test loading a graph that doesn't exist"""
    result = load_pylink_graph(filename="nonexistent_graph.json")
    assert result['status'] == 'error', \
        f"Expected error for nonexistent file, got: {result.get('status')}"


if __name__ == "__main__":
    test_status_endpoint()
    test_save_and_load_graph()
    test_list_graphs()
    test_compute_trajectory()
    test_invalid_graph_data()
    test_load_nonexistent_graph()
    print("✅ All API endpoint tests passed!")
