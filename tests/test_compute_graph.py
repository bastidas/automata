
import json
import os
from pathlib import Path
from backend.query_api import compute_graph

def test_compute_graph():
    """Test compute_graph function using test_graph.json"""
    
    # Load the test graph JSON
    test_dir = Path(__file__).parent
    test_graph_path = test_dir / "test_graph.json"
    
    with open(test_graph_path, 'r') as f:
        graph_data = json.load(f)
    
    # Add required n_iterations field to nodes if missing
    for node in graph_data.get('nodes', []):
        if 'n_iterations' not in node:
            node['n_iterations'] = 24
        # Convert id to name for backend compatibility
        if 'id' in node and 'name' not in node:
            node['name'] = node['id']
    
    # Call compute_graph function
    result = compute_graph(graph_data)
    
    # Verify the result
    assert result['status'] == 'success', f"Computation failed: {result.get('message', 'Unknown error')}"
    assert result['n_iterations'] == 24
    assert 'path_data' in result
    assert 'graph_info' in result
    
    print(f"✓ Test passed: {result['message']}")
    print(f"  - Nodes: {result['nodes_count']}")
    print(f"  - Links: {result['links_count']}")  
    print(f"  - Connections: {result['connections_count']}")
    
    return result

if __name__ == "__main__":
    test_compute_graph()