# import pytest
from __future__ import annotations

from pylinkage.joints import Crank

from pylink_tools.kinematic import build_joint_objects
from pylink_tools.kinematic import circle_circle_intersection
from pylink_tools.kinematic import compute_joint_position
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.kinematic import find_connected_link_groups
from pylink_tools.kinematic import make_linkage
from pylink_tools.kinematic import sync_distances_from_visual
from pylink_tools.kinematic import validate_mechanism


def test_circle_circle_intersection_basic():
    """Two circles at (0,0) r=3 and (4,0) r=3 intersect at y=±2.24"""
    pt = circle_circle_intersection((0, 0), 3, (4, 0), 3)
    assert abs(pt[0] - 2.0) < 0.01
    assert abs(pt[1] - 2.24) < 0.1  # sqrt(5) ≈ 2.24


def test_circle_circle_no_intersection_fallback():
    """When circles don't intersect, returns midpoint"""
    pt = circle_circle_intersection((0, 0), 1, (10, 0), 1)
    assert pt == (5.0, 0.0)


def test_compute_joint_position_static():
    """Static joint returns its stored x,y"""
    jdata = {'name': 'A', 'type': 'Static', 'x': 5.0, 'y': 3.0}
    pos = compute_joint_position(jdata, {'A': jdata}, {})
    assert pos == (5.0, 3.0)


def test_compute_joint_position_crank():
    """Crank at distance 2 from origin at angle 0 → (2, 0)"""
    joints = {
        'A': {'name': 'A', 'type': 'Static', 'x': 0, 'y': 0},
        'B': {'name': 'B', 'type': 'Crank', 'joint0': {'ref': 'A'}, 'distance': 2.0, 'angle': 0},
    }
    pos = compute_joint_position(joints['B'], joints, {})
    assert abs(pos[0] - 2.0) < 0.01 and abs(pos[1]) < 0.01


def test_sync_distances_updates_crank():
    """sync_distances corrects crank distance to match visual position"""
    joints = [
        {'name': 'A', 'type': 'Static', 'x': 0, 'y': 0},
        {'name': 'B', 'type': 'Crank', 'joint0': {'ref': 'A'}, 'distance': 999},  # wrong
    ]
    visual = {'A': (0, 0), 'B': (3, 4)}  # actual distance = 5
    updated = sync_distances_from_visual(joints, visual)
    assert abs(updated[1]['distance'] - 5.0) < 0.01


def test_build_joint_objects_creates_crank():
    """build_joint_objects creates Crank from serialized data"""
    joints = [
        {'name': 'A', 'type': 'Static', 'x': 0, 'y': 0},
        {'name': 'B', 'type': 'Crank', 'joint0': {'ref': 'A'}, 'distance': 2.0, 'angle': 0},
    ]
    joint_info = {j['name']: j for j in joints}
    objs = build_joint_objects(joints, ['A', 'B'], joint_info, {}, angle_per_step=0.1)
    assert isinstance(objs['A'], tuple)
    assert isinstance(objs['B'], Crank)


def test_make_linkage_requires_crank():
    """make_linkage fails without a Crank"""
    linkage, err = make_linkage({'A': (0, 0)}, ['A'])
    assert linkage is None
    assert 'Crank' in err


def test_compute_trajectory_4bar():
    """Full 4-bar trajectory computation"""
    pylink_data = {
        'n_steps': 4,
        'pylinkage': {
            'name': 'test',
            'joints': [
                {'name': 'O1', 'type': 'Static', 'x': 0, 'y': 0},
                {'name': 'O2', 'type': 'Static', 'x': 4, 'y': 0},
                {'name': 'A', 'type': 'Crank', 'joint0': {'ref': 'O1'}, 'distance': 1.5, 'angle': 0},
                {'name': 'B', 'type': 'Revolute', 'joint0': {'ref': 'A'}, 'joint1': {'ref': 'O2'}, 'distance0': 3.5, 'distance1': 2.5},
            ],
            'solve_order': ['O1', 'O2', 'A', 'B'],
        },
        'meta': {'joints': {}},
    }
    result = compute_trajectory(pylink_data)
    assert result.success
    assert len(result.trajectories['A']) == 4
    assert len(result.trajectories['B']) == 4
    # Static joints stay put
    assert all(p == [0.0, 0.0] for p in result.trajectories['O1'])


def test_find_connected_link_groups_single():
    """All 4 links in a 4-bar are one connected group"""
    joints = [
        {'name': 'A', 'type': 'Static'},
        {'name': 'B', 'type': 'Static'},
        {'name': 'C', 'type': 'Crank'},
        {'name': 'D', 'type': 'Revolute'},
    ]
    links = {
        'ground': {'connects': ['A', 'B']},
        'crank': {'connects': ['A', 'C']},
        'coupler': {'connects': ['C', 'D']},
        'rocker': {'connects': ['D', 'B']},
    }
    groups = find_connected_link_groups(links, joints)
    assert len(groups) == 1
    assert groups[0].is_valid
    assert len(groups[0].links) == 4


def test_find_connected_link_groups_two_groups():
    """Two disconnected link pairs → two groups"""
    joints = [
        {'name': 'A', 'type': 'Static'},
        {'name': 'B', 'type': 'Crank'},
        {'name': 'X', 'type': 'Static'},
        {'name': 'Y', 'type': 'Static'},
    ]
    links = {
        'link1': {'connects': ['A', 'B']},
        'link2': {'connects': ['X', 'Y']},
    }
    groups = find_connected_link_groups(links, joints)
    assert len(groups) == 2
    # Both invalid (< 3 links each)
    assert not any(g.is_valid for g in groups)


def test_validate_mechanism_4bar():
    """validate_mechanism passes for valid 4-bar"""
    pylink_data = {
        'pylinkage': {
            'joints': [
                {'name': 'O1', 'type': 'Static', 'x': 0, 'y': 0},
                {'name': 'O2', 'type': 'Static', 'x': 4, 'y': 0},
                {'name': 'A', 'type': 'Crank', 'joint0': {'ref': 'O1'}, 'distance': 1.5, 'angle': 0},
                {'name': 'B', 'type': 'Revolute', 'joint0': {'ref': 'A'}, 'joint1': {'ref': 'O2'}, 'distance0': 3.5, 'distance1': 2.5},
            ],
            'solve_order': ['O1', 'O2', 'A', 'B'],
        },
        'meta': {
            'joints': {},
            'links': {
                'ground': {'connects': ['O1', 'O2']},
                'crank': {'connects': ['O1', 'A']},
                'coupler': {'connects': ['A', 'B']},
                'rocker': {'connects': ['B', 'O2']},
            },
        },
    }
    result = validate_mechanism(pylink_data)
    assert result['valid']
    assert len(result['valid_groups']) == 1
