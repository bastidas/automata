"""
Test that demo.py runs successfully

This validates that pylinkage is properly installed and working.
If this test fails, it indicates fundamental problems with the pylinkage
installation or configuration, not issues with our code.

Critical validation:
- pylinkage library can be imported
- Demo linkages can be created
- Simulation can be executed
- Basic trajectory data is generated
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import pylinkage at module level
try:
    from pylinkage.joints import Crank, Revolute
    from pylinkage.linkage import Linkage
    PYLINKAGE_AVAILABLE = True
except ImportError:
    PYLINKAGE_AVAILABLE = False

from pylink_tools.demo import make_demo_linkage, optimization

def test_pylinkage_import():
    """Test that pylinkage can be imported"""
    if not PYLINKAGE_AVAILABLE:
        print("❌ Cannot import pylinkage - Fix: pip install pylinkage")
    assert PYLINKAGE_AVAILABLE, "pylinkage not installed - run: pip install pylinkage"


def test_demo():
    demo_linkage = make_demo_linkage()
    optimization(demo_linkage)