"""
test_optimization.py - Unit tests for linkage trajectory optimization.

Tests cover:
  1. Known solution tests (inverse problem) - perturb dimensions, recover original
  2. Real mechanism tests - use saved 4-bar from user/pygraphs/
  3. Convergence history verification
  4. Error metrics computation
  5. Dimension extraction and application
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimize import analyze_convergence
from pylink_tools.optimize import apply_dimensions
from pylink_tools.optimize import apply_dimensions_from_array
from pylink_tools.optimize import compute_trajectory_error
from pylink_tools.optimize import compute_trajectory_error_detailed
from pylink_tools.optimize import ConvergenceStats
from pylink_tools.optimize import create_fitness_function
from pylink_tools.optimize import dict_to_dimensions
from pylink_tools.optimize import dimensions_to_dict
from pylink_tools.optimize import DimensionSpec
from pylink_tools.optimize import evaluate_linkage_fit
from pylink_tools.optimize import extract_dimensions
from pylink_tools.optimize import extract_dimensions_with_custom_bounds
from pylink_tools.optimize import format_convergence_report
from pylink_tools.optimize import log_optimization_progress
from pylink_tools.optimize import OptimizationResult
from pylink_tools.optimize import optimize_trajectory
from pylink_tools.optimize import run_pso_optimization
from pylink_tools.optimize import run_scipy_optimization
from pylink_tools.optimize import TargetTrajectory
from pylink_tools.optimize import validate_bounds


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_4bar_pylink():
    """A simple 4-bar linkage with known dimensions."""
    return {
        'name': 'test_4bar',
        'pylinkage': {
            'name': 'test_4bar',
            'joints': [
                {'name': 'O1', 'type': 'Static', 'x': 0, 'y': 0},
                {'name': 'O2', 'type': 'Static', 'x': 4, 'y': 0},
                {'name': 'A', 'type': 'Crank', 'joint0': {'ref': 'O1'}, 'distance': 1.5, 'angle': 0.5},
                {
                    'name': 'B', 'type': 'Revolute', 'joint0': {'ref': 'A'}, 'joint1': {'ref': 'O2'},
                    'distance0': 3.5, 'distance1': 2.5,
                },
            ],
            'solve_order': ['O1', 'O2', 'A', 'B'],
        },
        'meta': {'joints': {}, 'links': {}},
        'n_steps': 12,
    }


@pytest.fixture
def test_4bar_json():
    """Load 4-bar from tests/4bar_test.json."""
    test_file = Path(__file__).parent / '4bar_test.json'
    if test_file.exists():
        with open(test_file) as f:
            data = json.load(f)
        data['n_steps'] = 24
        return data
    return None


@pytest.fixture
def user_4bar_json():
    """Load 4-bar from user/pygraphs/ if available."""
    user_dir = Path(__file__).parent.parent / 'user' / 'pygraphs'
    # Find any 4bar JSON file in the directory
    for user_file in sorted(user_dir.glob('4bar_*.json')):
        with open(user_file) as f:
            data = json.load(f)
        data['n_steps'] = 24
        return data
    return None


# =============================================================================
# Test: Dimension Extraction
# =============================================================================

class TestDimensionExtraction:
    """Tests for extract_dimensions and related functions."""

    def test_extract_dimensions_4bar(self, simple_4bar_pylink):
        """Extract dimensions from simple 4-bar."""
        spec = extract_dimensions(simple_4bar_pylink)

        # Should find 3 dimensions: crank distance, revolute distance0, distance1
        assert len(spec) == 3
        assert 'A_distance' in spec.names
        assert 'B_distance0' in spec.names
        assert 'B_distance1' in spec.names

        # Check initial values
        assert spec.initial_values[spec.names.index('A_distance')] == 1.5
        assert spec.initial_values[spec.names.index('B_distance0')] == 3.5
        assert spec.initial_values[spec.names.index('B_distance1')] == 2.5

    def test_extract_dimensions_bounds(self, simple_4bar_pylink):
        """Bounds should be computed from initial values."""
        spec = extract_dimensions(simple_4bar_pylink, bounds_factor=2.0)

        # A_distance = 1.5, bounds should be (0.75, 3.0)
        idx = spec.names.index('A_distance')
        assert spec.bounds[idx][0] == pytest.approx(0.75, rel=0.01)
        assert spec.bounds[idx][1] == pytest.approx(3.0, rel=0.01)

    def test_extract_dimensions_custom_bounds(self, simple_4bar_pylink):
        """Custom bounds override default computation."""
        custom = {'A_distance': (0.5, 5.0)}
        spec = extract_dimensions_with_custom_bounds(simple_4bar_pylink, custom)

        idx = spec.names.index('A_distance')
        assert spec.bounds[idx] == (0.5, 5.0)

    def test_bounds_tuple_format(self, simple_4bar_pylink):
        """get_bounds_tuple returns pylinkage format."""
        spec = extract_dimensions(simple_4bar_pylink)
        lower, upper = spec.get_bounds_tuple()

        assert len(lower) == len(spec)
        assert len(upper) == len(spec)
        assert all(l < u for l, u in zip(lower, upper))


# =============================================================================
# Test: Dimension Application
# =============================================================================

class TestDimensionApplication:
    """Tests for apply_dimensions and related functions."""

    def test_apply_dimensions_updates_joints(self, simple_4bar_pylink):
        """apply_dimensions updates joint distances."""
        spec = extract_dimensions(simple_4bar_pylink)
        new_values = {'A_distance': 2.0, 'B_distance0': 4.0}

        updated = apply_dimensions(simple_4bar_pylink, new_values, spec)

        # Original unchanged
        joints_orig = {j['name']: j for j in simple_4bar_pylink['pylinkage']['joints']}
        assert joints_orig['A']['distance'] == 1.5

        # Updated has new values
        joints_new = {j['name']: j for j in updated['pylinkage']['joints']}
        assert joints_new['A']['distance'] == 2.0
        assert joints_new['B']['distance0'] == 4.0
        assert joints_new['B']['distance1'] == 2.5  # unchanged

    def test_apply_dimensions_from_array(self, simple_4bar_pylink):
        """apply_dimensions_from_array uses spec order."""
        spec = extract_dimensions(simple_4bar_pylink)
        values = (2.0, 4.0, 3.0)  # A_distance, B_distance0, B_distance1

        updated = apply_dimensions_from_array(simple_4bar_pylink, values, spec)

        joints_new = {j['name']: j for j in updated['pylinkage']['joints']}
        assert joints_new['A']['distance'] == 2.0
        assert joints_new['B']['distance0'] == 4.0
        assert joints_new['B']['distance1'] == 3.0

    def test_apply_dimensions_roundtrip(self, simple_4bar_pylink):
        """Extract → modify → apply → extract gives correct values."""
        spec = extract_dimensions(simple_4bar_pylink)
        new_values = (2.0, 4.0, 3.0)

        updated = apply_dimensions_from_array(simple_4bar_pylink, new_values, spec)
        spec2 = extract_dimensions(updated)

        for i, name in enumerate(spec.names):
            assert spec2.initial_values[i] == new_values[i]


# =============================================================================
# Test: Error Computation
# =============================================================================

class TestErrorComputation:
    """Tests for trajectory error computation."""

    def test_compute_error_perfect_match(self):
        """Perfect match gives zero error."""
        positions = [(1, 2), (3, 4), (5, 6)]
        target = TargetTrajectory(joint_name='test', positions=positions)

        error = compute_trajectory_error(positions, target, metric='mse')
        assert error == pytest.approx(0.0, abs=1e-10)

    def test_compute_error_shifted(self):
        """Shifted positions give expected error."""
        target_pos = [(0, 0), (1, 0), (2, 0)]
        computed = [(1, 0), (2, 0), (3, 0)]  # Shifted by 1 in x
        target = TargetTrajectory(joint_name='test', positions=target_pos)

        # Each point is distance 1 away, MSE = 1^2 = 1
        error = compute_trajectory_error(computed, target, metric='mse')
        assert error == pytest.approx(1.0, abs=0.01)

    def test_compute_error_diagonal_shift(self):
        """Diagonal shift gives correct error."""
        target_pos = [(0, 0)]
        computed = [(3, 4)]  # Distance = 5
        target = TargetTrajectory(joint_name='test', positions=target_pos)

        # MSE = 5^2 = 25
        error = compute_trajectory_error(computed, target, metric='mse')
        assert error == pytest.approx(25.0, abs=0.01)

        # RMSE = 5
        rmse = compute_trajectory_error(computed, target, metric='rmse')
        assert rmse == pytest.approx(5.0, abs=0.01)

    def test_compute_error_weighted(self):
        """Weights affect error computation."""
        target_pos = [(0, 0), (0, 0)]
        computed = [(1, 0), (2, 0)]  # Errors: 1, 2

        # Uniform weights: MSE = (1 + 4) / 2 = 2.5
        target_uniform = TargetTrajectory(joint_name='test', positions=target_pos)
        error_uniform = compute_trajectory_error(computed, target_uniform, metric='mse')
        assert error_uniform == pytest.approx(2.5, abs=0.01)

        # Weight first point 3x: MSE = (3*1 + 1*4) / 4 = 1.75
        target_weighted = TargetTrajectory(joint_name='test', positions=target_pos, weights=[3.0, 1.0])
        error_weighted = compute_trajectory_error(computed, target_weighted, metric='mse')
        assert error_weighted == pytest.approx(1.75, abs=0.01)

    def test_error_metrics_detailed(self):
        """compute_trajectory_error_detailed returns all metrics."""
        target_pos = [(0, 0), (0, 0), (0, 0)]
        computed = [(1, 0), (2, 0), (3, 0)]  # Distances: 1, 2, 3
        target = TargetTrajectory(joint_name='test', positions=target_pos)

        metrics = compute_trajectory_error_detailed(computed, target)

        assert metrics.max_error == pytest.approx(3.0, abs=0.01)
        assert len(metrics.per_step_errors) == 3
        assert metrics.per_step_errors[0] == pytest.approx(1.0, abs=0.01)
        assert metrics.per_step_errors[2] == pytest.approx(3.0, abs=0.01)


# =============================================================================
# Test: Known Solution (Inverse Problem)
# =============================================================================

class TestKnownSolution:
    """
    Tests where we know the answer: start with valid linkage,
    compute trajectory, perturb dimensions, verify optimizer recovers.
    """

    def test_perfect_match_stays_optimal(self, simple_4bar_pylink):
        """When target matches current trajectory, optimizer keeps dimensions."""
        result = compute_trajectory(simple_4bar_pylink, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory for test linkage')

        target_joint = 'B'
        trajectory = result.trajectories[target_joint]
        target = TargetTrajectory(joint_name=target_joint, positions=trajectory)

        # Error should already be ~0
        metrics = evaluate_linkage_fit(simple_4bar_pylink, target)
        assert metrics.rmse < 0.01, 'Initial fit should be perfect'

        # Optimization should maintain ~0 error
        opt_result = run_scipy_optimization(
            simple_4bar_pylink, target,
            max_iterations=20, verbose=False,
        )

        assert opt_result.success
        assert opt_result.final_error < 0.01

    def test_recover_perturbed_dimensions_scipy(self, test_4bar_json):
        """Perturb dimensions, verify scipy optimizer can reduce error."""
        if test_4bar_json is None:
            pytest.skip('Test 4bar JSON not found')

        # Compute original trajectory
        result = compute_trajectory(test_4bar_json, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory for test linkage')

        target_joint = 'coupler_rocker_joint'
        original_trajectory = result.trajectories[target_joint]
        target = TargetTrajectory(joint_name=target_joint, positions=original_trajectory)

        # Perturb dimensions (make linkage "wrong")
        perturbed = copy.deepcopy(test_4bar_json)
        joints = {j['name']: j for j in perturbed['pylinkage']['joints']}
        joints['crank']['distance'] *= 1.2  # 20% longer
        joints['coupler_rocker_joint']['distance0'] *= 0.9  # 10% shorter

        # Compute error of perturbed linkage
        initial_metrics = evaluate_linkage_fit(perturbed, target)

        # Run optimization
        opt_result = run_scipy_optimization(
            perturbed, target,
            max_iterations=100, verbose=False,
        )

        assert opt_result.success
        # Final error should be less than initial (optimizer improved)
        # Note: May not fully recover due to local minima
        assert opt_result.final_error <= initial_metrics.mse + 0.01

    def test_recover_perturbed_dimensions_pso(self, test_4bar_json):
        """Perturb dimensions, verify PSO optimizer can reduce error."""
        if test_4bar_json is None:
            pytest.skip('Test 4bar JSON not found')

        # Compute original trajectory
        result = compute_trajectory(test_4bar_json, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory for test linkage')

        target_joint = 'coupler_rocker_joint'
        original_trajectory = result.trajectories[target_joint]
        target = TargetTrajectory(joint_name=target_joint, positions=original_trajectory)

        # Perturb dimensions
        perturbed = copy.deepcopy(test_4bar_json)
        joints = {j['name']: j for j in perturbed['pylinkage']['joints']}
        joints['crank']['distance'] *= 1.15

        # Compute initial error
        initial_metrics = evaluate_linkage_fit(perturbed, target)

        # Run PSO optimization
        opt_result = run_pso_optimization(
            perturbed, target,
            n_particles=15, iterations=20, verbose=False,
        )

        assert opt_result.success
        assert opt_result.final_error <= initial_metrics.mse + 0.01


# =============================================================================
# Test: Real Mechanisms from user/pygraphs/
# =============================================================================

class TestRealMechanisms:
    """Tests using real saved mechanisms."""

    def test_user_4bar_trajectory(self, user_4bar_json):
        """Real 4-bar can compute trajectory."""
        if user_4bar_json is None:
            pytest.skip('User 4bar JSON not found')

        result = compute_trajectory(user_4bar_json, verbose=False)
        assert result.success
        assert 'coupler_rocker_joint' in result.trajectories
        assert len(result.trajectories['coupler_rocker_joint']) == 24

    def test_user_4bar_dimension_extraction(self, user_4bar_json):
        """Real 4-bar dimensions can be extracted."""
        if user_4bar_json is None:
            pytest.skip('User 4bar JSON not found')

        spec = extract_dimensions(user_4bar_json)

        # Should have at least 3 dimensions (simple 4-bar has 3, complex mechanisms have more)
        # Hypergraph format: names end with '_distance' (e.g., 'crank_link_distance')
        # Legacy format: names like 'crank_distance', 'coupler_rocker_joint_distance0'
        assert len(spec) >= 3
        # At least one dimension should be named with 'distance'
        assert any('distance' in name for name in spec.names)

    def test_user_4bar_optimization(self, user_4bar_json):
        """Real 4-bar can be optimized."""
        if user_4bar_json is None:
            pytest.skip('User 4bar JSON not found')

        # Compute trajectory
        result = compute_trajectory(user_4bar_json, verbose=False)
        assert result.success

        target_joint = 'coupler_rocker_joint'
        trajectory = result.trajectories[target_joint]
        target = TargetTrajectory(joint_name=target_joint, positions=trajectory)

        # Run optimization (should stay near 0 since target matches)
        opt_result = optimize_trajectory(
            user_4bar_json, target,
            method='scipy',
            max_iterations=30, verbose=False,
        )

        assert opt_result.success
        assert opt_result.final_error < 1.0  # Should be very close to 0


# =============================================================================
# Test: Convergence History
# =============================================================================

class TestConvergenceHistory:
    """Tests for convergence history tracking."""

    def test_scipy_tracks_history(self, simple_4bar_pylink):
        """scipy optimization tracks convergence history."""
        result = compute_trajectory(simple_4bar_pylink, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='B',
            positions=result.trajectories['B'],
        )

        opt_result = run_scipy_optimization(
            simple_4bar_pylink, target,
            max_iterations=30, verbose=False,
        )

        assert opt_result.convergence_history is not None
        assert len(opt_result.convergence_history) >= 1
        # First entry should be initial error
        assert opt_result.convergence_history[0] == pytest.approx(opt_result.initial_error, rel=0.1)

    def test_pso_tracks_history(self, simple_4bar_pylink):
        """PSO optimization tracks convergence history."""
        result = compute_trajectory(simple_4bar_pylink, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='B',
            positions=result.trajectories['B'],
        )

        opt_result = run_pso_optimization(
            simple_4bar_pylink, target,
            n_particles=10, iterations=15, verbose=False,
        )

        assert opt_result.convergence_history is not None
        # PSO history has iterations + 1 entries (initial + each iteration)
        assert len(opt_result.convergence_history) == 16

    def test_history_is_monotonic_or_stable(self, test_4bar_json):
        """Convergence history should generally decrease (or stay same)."""
        if test_4bar_json is None:
            pytest.skip('Test 4bar JSON not found')

        result = compute_trajectory(test_4bar_json, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='coupler_rocker_joint',
            positions=result.trajectories['coupler_rocker_joint'],
        )

        opt_result = run_pso_optimization(
            test_4bar_json, target,
            n_particles=15, iterations=20, verbose=False,
        )

        if opt_result.convergence_history:
            # Final error should be <= initial (or very close)
            assert opt_result.convergence_history[-1] <= opt_result.convergence_history[0] + 0.1


# =============================================================================
# Test: Fitness Function
# =============================================================================

class TestFitnessFunction:
    """Tests for fitness function creation."""

    def test_fitness_function_callable(self, simple_4bar_pylink):
        """create_fitness_function returns callable."""
        spec = extract_dimensions(simple_4bar_pylink)
        result = compute_trajectory(simple_4bar_pylink, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='B',
            positions=result.trajectories['B'],
        )

        fitness = create_fitness_function(simple_4bar_pylink, target, spec)

        assert callable(fitness)

        # Call with initial values
        initial_values = tuple(spec.initial_values)
        error = fitness(initial_values)

        assert isinstance(error, float)
        assert error >= 0 or error == float('inf')

    def test_fitness_function_responds_to_changes(self, test_4bar_json):
        """Fitness function returns different values for different dimensions."""
        if test_4bar_json is None:
            pytest.skip('Test 4bar JSON not found')

        spec = extract_dimensions(test_4bar_json)
        result = compute_trajectory(test_4bar_json, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='coupler_rocker_joint',
            positions=result.trajectories['coupler_rocker_joint'],
        )

        fitness = create_fitness_function(test_4bar_json, target, spec)

        # Evaluate at initial values (should be ~0)
        initial_values = tuple(spec.initial_values)
        error_initial = fitness(initial_values)

        # Evaluate at perturbed values (should be higher)
        perturbed = tuple(v * 1.3 for v in initial_values)
        error_perturbed = fitness(perturbed)

        # Perturbed should have higher error (or both inf)
        if error_initial != float('inf'):
            assert error_perturbed >= error_initial or error_perturbed == float('inf')


# =============================================================================
# Test: Utility Functions
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_dimensions_to_dict(self):
        """dimensions_to_dict converts array to named dict."""
        spec = DimensionSpec(
            names=['a', 'b', 'c'],
            initial_values=[1.0, 2.0, 3.0],
            bounds=[(0, 10), (0, 10), (0, 10)],
            joint_mapping={},
        )

        result = dimensions_to_dict((5.0, 6.0, 7.0), spec)

        assert result == {'a': 5.0, 'b': 6.0, 'c': 7.0}

    def test_dict_to_dimensions(self):
        """dict_to_dimensions converts named dict to array."""
        spec = DimensionSpec(
            names=['a', 'b', 'c'],
            initial_values=[1.0, 2.0, 3.0],
            bounds=[(0, 10), (0, 10), (0, 10)],
            joint_mapping={},
        )

        result = dict_to_dimensions({'a': 5.0, 'b': 6.0, 'c': 7.0}, spec)

        assert result == (5.0, 6.0, 7.0)

    def test_dict_to_dimensions_with_missing(self):
        """dict_to_dimensions uses initial values for missing keys."""
        spec = DimensionSpec(
            names=['a', 'b', 'c'],
            initial_values=[1.0, 2.0, 3.0],
            bounds=[(0, 10), (0, 10), (0, 10)],
            joint_mapping={},
        )

        result = dict_to_dimensions({'a': 5.0}, spec)  # b and c missing

        assert result == (5.0, 2.0, 3.0)  # Uses initial values for b, c

    def test_validate_bounds_valid(self):
        """validate_bounds returns empty for valid values."""
        spec = DimensionSpec(
            names=['a', 'b'],
            initial_values=[1.0, 2.0],
            bounds=[(0, 10), (0, 10)],
            joint_mapping={},
        )

        violations = validate_bounds((5.0, 5.0), spec)

        assert violations == []

    def test_validate_bounds_violations(self):
        """validate_bounds reports violations."""
        spec = DimensionSpec(
            names=['a', 'b'],
            initial_values=[1.0, 2.0],
            bounds=[(0, 10), (0, 10)],
            joint_mapping={},
        )

        violations = validate_bounds((-1.0, 15.0), spec)

        assert len(violations) == 2
        assert 'a' in violations[0] and 'min' in violations[0]
        assert 'b' in violations[1] and 'max' in violations[1]


# =============================================================================
# Test: OptimizationResult
# =============================================================================

class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_to_dict(self):
        """OptimizationResult.to_dict() serializes correctly."""
        result = OptimizationResult(
            success=True,
            optimized_dimensions={'a': 1.0, 'b': 2.0},
            optimized_pylink_data={'test': 'data'},
            initial_error=10.0,
            final_error=1.0,
            iterations=50,
            convergence_history=[10.0, 5.0, 1.0],
        )

        d = result.to_dict()

        assert d['success'] is True
        assert d['optimized_dimensions'] == {'a': 1.0, 'b': 2.0}
        assert d['initial_error'] == 10.0
        assert d['final_error'] == 1.0
        assert d['iterations'] == 50
        assert d['convergence_history'] == [10.0, 5.0, 1.0]


# =============================================================================
# Test: TargetTrajectory
# =============================================================================

class TestTargetTrajectory:
    """Tests for TargetTrajectory dataclass."""

    def test_target_trajectory_basic(self):
        """Basic TargetTrajectory creation."""
        target = TargetTrajectory(
            joint_name='test',
            positions=[(1, 2), (3, 4), (5, 6)],
        )

        assert target.joint_name == 'test'
        assert target.n_steps == 3
        assert target.weights == [1.0, 1.0, 1.0]  # Default uniform

    def test_target_trajectory_with_weights(self):
        """TargetTrajectory with custom weights."""
        target = TargetTrajectory(
            joint_name='test',
            positions=[(1, 2), (3, 4)],
            weights=[2.0, 3.0],
        )

        assert target.weights == [2.0, 3.0]

    def test_target_trajectory_from_dict(self):
        """TargetTrajectory.from_dict() deserializes correctly."""
        data = {
            'joint_name': 'coupler',
            'positions': [[1, 2], [3, 4]],
            'weights': [1.0, 2.0],
        }

        target = TargetTrajectory.from_dict(data)

        assert target.joint_name == 'coupler'
        assert target.n_steps == 2
        assert target.weights == [1.0, 2.0]

    def test_target_trajectory_to_dict(self):
        """TargetTrajectory.to_dict() serializes correctly."""
        target = TargetTrajectory(
            joint_name='test',
            positions=[(1, 2), (3, 4)],
        )

        d = target.to_dict()

        assert d['joint_name'] == 'test'
        assert d['positions'] == [[1, 2], [3, 4]]
        assert d['n_steps'] == 2


# =============================================================================
# Test: Convergence Logging Utilities
# =============================================================================

class TestConvergenceLogging:
    """Tests for convergence analysis and logging utilities."""

    def test_analyze_convergence_basic(self):
        """analyze_convergence computes basic stats correctly."""
        history = [10.0, 8.0, 5.0, 3.0, 2.0, 1.0]

        stats = analyze_convergence(history)

        assert stats.initial_error == 10.0
        assert stats.final_error == 1.0
        assert stats.best_error == 1.0
        assert stats.improvement_pct == pytest.approx(90.0, rel=0.01)
        assert stats.n_iterations == 5  # 6 entries, first is initial
        assert len(stats.improvement_per_iteration) == 5

    def test_analyze_convergence_no_improvement(self):
        """analyze_convergence handles flat history."""
        history = [5.0, 5.0, 5.0, 5.0]

        stats = analyze_convergence(history)

        assert stats.initial_error == 5.0
        assert stats.final_error == 5.0
        assert stats.improvement_pct == 0.0

    def test_analyze_convergence_with_inf(self):
        """analyze_convergence handles inf values."""
        history = [float('inf'), float('inf'), 10.0, 5.0, 1.0]

        stats = analyze_convergence(history)

        assert stats.best_error == 1.0
        assert stats.final_error == 1.0

    def test_analyze_convergence_empty(self):
        """analyze_convergence handles empty history."""
        stats = analyze_convergence([])

        assert stats.n_iterations == 0
        assert stats.converged is False

    def test_analyze_convergence_converged(self):
        """analyze_convergence detects convergence."""
        # Last two values are very close
        history = [10.0, 5.0, 2.0, 1.0001, 1.0000]

        stats = analyze_convergence(history, tolerance=1e-3)

        assert stats.converged is True

    def test_format_convergence_report(self):
        """format_convergence_report produces readable output."""
        result = OptimizationResult(
            success=True,
            optimized_dimensions={'a': 1.5, 'b': 2.5},
            initial_error=10.0,
            final_error=1.0,
            iterations=50,
            convergence_history=[10.0, 5.0, 1.0],
        )

        report = format_convergence_report(result)

        assert 'SUCCESS' in report
        assert 'Initial Error: 10' in report
        assert 'Final Error:   1' in report
        assert '90.0%' in report  # improvement
        assert 'Optimized Dimensions' in report

    def test_format_convergence_report_with_history(self):
        """format_convergence_report includes history when requested."""
        result = OptimizationResult(
            success=True,
            optimized_dimensions={},
            initial_error=10.0,
            final_error=1.0,
            iterations=2,
            convergence_history=[10.0, 5.0, 1.0],
        )

        report = format_convergence_report(result, include_history=True)

        assert 'Convergence History' in report
        assert '[  0] 10' in report

    def test_log_optimization_progress(self):
        """log_optimization_progress formats iteration info."""
        progress = log_optimization_progress(
            iteration=42,
            current_error=2.5,
            best_error=1.0,
            dimensions=(1.5, 2.5, 3.5),
            dimension_names=['a', 'b', 'c'],
        )

        assert '[  42]' in progress
        assert 'error=2.5' in progress
        assert 'best=1.0' in progress
        assert 'a=1.50' in progress

    def test_convergence_stats_to_dict(self):
        """ConvergenceStats.to_dict() serializes correctly."""
        stats = ConvergenceStats(
            initial_error=10.0,
            final_error=1.0,
            best_error=1.0,
            improvement_pct=90.0,
            n_iterations=50,
            n_evaluations=51,
            converged=True,
            history=[10.0, 5.0, 1.0],
            improvement_per_iteration=[5.0, 4.0],
        )

        d = stats.to_dict()

        assert d['initial_error'] == 10.0
        assert d['improvement_pct'] == 90.0
        assert d['converged'] is True


# =============================================================================
# Test: Integration with Real Optimization
# =============================================================================

class TestIntegrationConvergence:
    """Integration tests for convergence logging with real optimization."""

    def test_scipy_produces_analyzable_history(self, test_4bar_json):
        """scipy optimization history can be analyzed."""
        if test_4bar_json is None:
            pytest.skip('Test 4bar JSON not found')

        result = compute_trajectory(test_4bar_json, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='coupler_rocker_joint',
            positions=result.trajectories['coupler_rocker_joint'],
        )

        opt_result = run_scipy_optimization(
            test_4bar_json, target,
            max_iterations=30, verbose=False,
        )

        if opt_result.convergence_history:
            stats = analyze_convergence(opt_result.convergence_history)

            assert stats.initial_error >= 0
            assert stats.n_iterations >= 0

            # Report should be generatable
            report = format_convergence_report(opt_result)
            assert len(report) > 0

    def test_pso_produces_analyzable_history(self, test_4bar_json):
        """PSO optimization history can be analyzed."""
        if test_4bar_json is None:
            pytest.skip('Test 4bar JSON not found')

        result = compute_trajectory(test_4bar_json, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='coupler_rocker_joint',
            positions=result.trajectories['coupler_rocker_joint'],
        )

        opt_result = run_pso_optimization(
            test_4bar_json, target,
            n_particles=10, iterations=15, verbose=False,
        )

        if opt_result.convergence_history:
            stats = analyze_convergence(opt_result.convergence_history)

            # PSO should have iterations + 1 history entries
            assert stats.n_evaluations == 16

            report = format_convergence_report(opt_result, include_history=True)
            assert 'Convergence History' in report


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
