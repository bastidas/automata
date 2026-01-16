#!/usr/bin/env python3
"""Test matplotlib GUI warning fix by calling make_graph directly"""

# First, let's test the backend configuration
print("Testing matplotlib backend configuration...")

# This should configure matplotlib before any plotting imports
from configs.matplotlib_config import configure_matplotlib_for_backend
backend = configure_matplotlib_for_backend()
print(f"Matplotlib backend: {backend}")

# Now test the actual function
print("\nTesting make_graph function...")
try:
    from structs.basic import make_graph
    from configs.link_models import Link, Node
    
    # Create minimal test data
    test_nodes = [
        Node(id="node1", pos=(100, 200), fixed=False),
        Node(id="node2", pos=(300, 200), fixed=False)
    ]
    
    test_connections = [{
        "from_node": "node1",
        "to_node": "node2", 
        "link": {"name": "test_rod", "length": 4.0, "n_iterations": 5}
    }]
    
    test_links = [Link(name="test_rod", length=4.0, n_iterations=5)]
    
    print("Calling make_graph with n_iterations=5...")
    result = make_graph(test_connections, test_links, test_nodes, n_iterations=5)
    
    print(f"✅ SUCCESS! Graph created: {result['node_count']} nodes, {result['edge_count']} edges")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()