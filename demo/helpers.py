"""
Shared utilities for demo scripts.

This module provides common functions used across multiple demos:
- Mechanism loading from test_graphs/
- Dimension extraction for optimization
- Formatted output helpers
"""
from __future__ import annotations

import json
from pathlib import Path


# =============================================================================
# MECHANISM REGISTRY
# =============================================================================
# Available test mechanisms with their configurations.

TEST_GRAPHS_DIR = Path(__file__).parent / 'test_graphs'

MECHANISMS = {
    'simple': {
        'file': TEST_GRAPHS_DIR / '4bar.json',
        'target_joint': 'coupler_rocker_joint',
        'description': 'Simple 4-bar linkage (4 joints, ~3 dimensions)',
    },
    'intermediate': {
        'file': TEST_GRAPHS_DIR / 'intermediate.json',
        'target_joint': 'final',
        'description': 'Intermediate 6-link mechanism (5 joints, ~5 dimensions)',
    },
    'complex': {
        'file': TEST_GRAPHS_DIR / 'complex.json',
        'target_joint': 'final_joint',
        'description': 'Complex multi-link mechanism (10 joints, ~15 dimensions)',
    },
    'leg': {
        'file': TEST_GRAPHS_DIR / 'leg.json',
        'target_joint': 'toe',
        'description': 'Leg mechanism (17 joints, ~28 dimensions)',
    },
}


def load_mechanism(mechanism_type: str, n_steps: int = 32) -> tuple[dict, str, str]:
    """
    Load a mechanism from test_graphs/.

    Args:
        mechanism_type: One of 'simple', 'intermediate', 'complex', 'leg'
        n_steps: Number of simulation steps per revolution

    Returns:
        (pylink_data, target_joint, description)

    Raises:
        ValueError: If mechanism_type is unknown
        FileNotFoundError: If mechanism file doesn't exist
    """
    if mechanism_type not in MECHANISMS:
        available = list(MECHANISMS.keys())
        raise ValueError(f"Unknown mechanism '{mechanism_type}'. Available: {available}")

    config = MECHANISMS[mechanism_type]
    json_path = config['file']

    if not json_path.exists():
        raise FileNotFoundError(f'Mechanism file not found: {json_path}')

    with open(json_path) as f:
        pylink_data = json.load(f)

    pylink_data['n_steps'] = n_steps

    return pylink_data, config['target_joint'], config['description']


def get_dimension_spec(pylink_data: dict, mechanism_type: str, bounds_factor: float = 1.5, min_length: float = 3.0):
    """
    Extract optimizable dimensions from a mechanism.

    Uses the appropriate extraction method based on mechanism format.

    Args:
        pylink_data: Mechanism data
        mechanism_type: Type of mechanism (affects extraction method)
        bounds_factor: How much dimensions can vary (e.g., 1.5 = [val/1.5, val*1.5])
        min_length: Minimum allowed link length

    Returns:
        DimensionSpec with names, initial_values, bounds
    """
    from pylink_tools.optimize import extract_dimensions
    from pylink_tools.optimization_helpers import extract_dimensions_from_edges

    # Complex mechanisms use edge-based extraction
    if mechanism_type in ('complex', 'intermediate', 'leg'):
        return extract_dimensions_from_edges(
            pylink_data,
            bounds_factor=bounds_factor,
            min_length=min_length,
        )
    else:
        return extract_dimensions(
            pylink_data,
            bounds_factor=bounds_factor,
            min_length=min_length,
        )


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print('\n' + '=' * width)
    print(f'  {title}')
    print('=' * width)


def print_mechanism_info(pylink_data: dict, target_joint: str, description: str):
    """Print summary of loaded mechanism."""
    print(f'\nMechanism: {description}')
    print(f'Target joint: {target_joint}')

    # Count joints based on format
    if 'pylinkage' in pylink_data and 'joints' in pylink_data['pylinkage']:
        n_joints = len(pylink_data['pylinkage']['joints'])
        print(f'Joints: {n_joints}')
    elif 'linkage' in pylink_data and 'nodes' in pylink_data['linkage']:
        n_joints = len(pylink_data['linkage']['nodes'])
        n_edges = len(pylink_data['linkage']['edges'])
        print(f'Joints: {n_joints}, Links: {n_edges}')


def print_dimensions(dim_spec, max_show: int = 6):
    """Print summary of optimizable dimensions."""
    print(f'\nOptimizable dimensions: {len(dim_spec)}')

    for i, (name, initial, bounds) in enumerate(
        zip(
            dim_spec.names, dim_spec.initial_values, dim_spec.bounds,
        ),
    ):
        if i >= max_show:
            print(f'  ... and {len(dim_spec) - max_show} more')
            break
        print(f'  - {name}: {initial:.2f} (bounds: {bounds[0]:.2f} to {bounds[1]:.2f})')
