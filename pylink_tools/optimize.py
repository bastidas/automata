"""
optimize.py - Trajectory optimization for linkage mechanisms.

Core functionality:
  - Extract optimizable dimensions (link lengths) from a linkage spec
  - Apply dimension values back to a linkage spec
  - Compute error between computed trajectory and target trajectory
  - Run PSO optimization to fit linkage to target path

Design notes:
  - Static joints are fixed (not optimizable)
  - Crank distance and Revolute distances are optimizable
  - Crank angle is NOT optimized (it's the driver)
  - Compatible with pylinkage's optimization API

CRITICAL PARAMETERS (see trajectory_utils.py for full documentation):

  N_STEPS (Simulation Step Count):
    - Higher = better accuracy, slower optimization
    - Target trajectory MUST have same point count as N_STEPS
    - Use trajectory_utils.resample_trajectory() to match counts
    - Recommended: 24-48 for optimization, 48-96 for final results

  PHASE_ALIGNMENT:
    - Trajectories may trace same path but start at different points
    - Without alignment: identical paths can have HUGE errors
    - Use phase_invariant=True in error computation for robust scoring
    - See trajectory_utils.compute_phase_aligned_distance()
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# =============================================================================
# Schemas / Data Structures
# =============================================================================

@dataclass
class DimensionSpec:
    """
    Describes optimizable dimensions of a linkage.

    Each dimension corresponds to a link length that can be adjusted
    during optimization.

    Attributes:
        names: Human-readable names for each dimension (e.g., "B_distance", "C_distance0")
        initial_values: Current values of each dimension
        bounds: (min, max) tuple for each dimension
        joint_mapping: Maps dimension name -> (joint_name, property_name)
                       e.g., {"B_distance": ("B", "distance")}
    """
    names: list[str]
    initial_values: list[float]
    bounds: list[tuple[float, float]]
    joint_mapping: dict[str, tuple[str, str]]

    def __len__(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict:
        return {
            'names': self.names,
            'initial_values': self.initial_values,
            'bounds': self.bounds,
            'joint_mapping': {k: list(v) for k, v in self.joint_mapping.items()},
            'n_dimensions': len(self.names),
        }

    def get_bounds_tuple(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return bounds in pylinkage format: ((lower...), (upper...))"""
        lower = tuple(b[0] for b in self.bounds)
        upper = tuple(b[1] for b in self.bounds)
        return (lower, upper)


@dataclass
class TargetTrajectory:
    """
    Target positions for optimization.

    Specifies where we want a particular joint to be at each timestep.

    Attributes:
        joint_name: Which joint to optimize (match its path to target)
        positions: List of (x, y) positions, one per timestep
        weights: Optional per-timestep weights (higher = more important)
                 Defaults to uniform weights if not provided.
    """
    joint_name: str
    positions: list[tuple[float, float]]
    weights: list[float] | None = None

    def __post_init__(self):
        # Convert positions to tuples if they're lists
        self.positions = [tuple(p) for p in self.positions]

        # Set uniform weights if not provided
        if self.weights is None:
            self.weights = [1.0] * len(self.positions)

    @property
    def n_steps(self) -> int:
        return len(self.positions)

    def to_dict(self) -> dict:
        return {
            'joint_name': self.joint_name,
            'positions': [list(p) for p in self.positions],
            'weights': self.weights,
            'n_steps': self.n_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetTrajectory:
        return cls(
            joint_name=data['joint_name'],
            positions=data['positions'],
            weights=data.get('weights'),
        )


@dataclass
class OptimizationResult:
    """
    Result of an optimization run.

    Attributes:
        success: Whether optimization completed successfully
        optimized_dimensions: Final dimension values {name: value}
        optimized_pylink_data: Updated pylink_data with optimized dimensions
        initial_error: Error before optimization
        final_error: Error after optimization
        iterations: Number of iterations/evaluations performed
        convergence_history: Optional list of error values over iterations
        error: Error message if success=False
    """
    success: bool
    optimized_dimensions: dict[str, float]
    optimized_pylink_data: dict | None = None
    initial_error: float = 0.0
    final_error: float = 0.0
    iterations: int = 0
    convergence_history: list[float] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'optimized_dimensions': self.optimized_dimensions,
            'optimized_pylink_data': self.optimized_pylink_data,
            'initial_error': self.initial_error,
            'final_error': self.final_error,
            'iterations': self.iterations,
            'convergence_history': self.convergence_history,
            'error': self.error,
        }


@dataclass
class ErrorMetrics:
    """
    Detailed error metrics between computed and target trajectory.

    Attributes:
        total_error: Sum of squared distances
        mse: Mean squared error
        rmse: Root mean squared error
        max_error: Maximum distance at any timestep
        per_step_errors: Distance error at each timestep
    """
    total_error: float
    mse: float
    rmse: float
    max_error: float
    per_step_errors: list[float]

    def to_dict(self) -> dict:
        return {
            'total_error': self.total_error,
            'mse': self.mse,
            'rmse': self.rmse,
            'max_error': self.max_error,
            'per_step_errors': self.per_step_errors,
        }


# =============================================================================
# Dimension Extraction
# =============================================================================

def extract_dimensions(
    pylink_data: dict,
    bounds_factor: float = 2.0,
    min_length: float = 0.1,
) -> DimensionSpec:
    """
    Extract optimizable dimensions (link lengths) from pylink_data.

    Identifies all link lengths that can be adjusted:
      - Crank: distance (length from parent to crank point)
      - Revolute: distance0, distance1 (lengths to each parent)

    Static joints are NOT included (they are fixed ground points).

    Args:
        pylink_data: Full pylink document with 'pylinkage' section
        bounds_factor: Multiplier for suggested bounds (e.g., 2.0 means
                       bounds are [value/2, value*2])
        min_length: Minimum allowed link length

    Returns:
        DimensionSpec with names, initial values, bounds, and joint mapping

    Example:
        >>> spec = extract_dimensions(pylink_data)
        >>> print(spec.names)
        ['B_distance', 'C_distance0', 'C_distance1']
        >>> print(spec.initial_values)
        [1.0, 3.0, 1.0]
    """
    pylinkage_data = pylink_data.get('pylinkage', {})
    joints_data = pylinkage_data.get('joints', [])

    names: list[str] = []
    initial_values: list[float] = []
    bounds: list[tuple[float, float]] = []
    joint_mapping: dict[str, tuple[str, str]] = {}

    for jdata in joints_data:
        jtype = jdata.get('type')
        jname = jdata.get('name')

        if jtype == 'Crank':
            # Crank has a single 'distance' parameter
            distance = jdata.get('distance', 1.0)
            dim_name = f'{jname}_distance'

            names.append(dim_name)
            initial_values.append(distance)
            bounds.append(_compute_bounds(distance, bounds_factor, min_length))
            joint_mapping[dim_name] = (jname, 'distance')

        elif jtype == 'Revolute':
            # Revolute has distance0 and distance1
            distance0 = jdata.get('distance0', 1.0)
            distance1 = jdata.get('distance1', 1.0)

            dim_name0 = f'{jname}_distance0'
            dim_name1 = f'{jname}_distance1'

            names.append(dim_name0)
            initial_values.append(distance0)
            bounds.append(_compute_bounds(distance0, bounds_factor, min_length))
            joint_mapping[dim_name0] = (jname, 'distance0')

            names.append(dim_name1)
            initial_values.append(distance1)
            bounds.append(_compute_bounds(distance1, bounds_factor, min_length))
            joint_mapping[dim_name1] = (jname, 'distance1')

        # Static joints are skipped (fixed ground points)

    return DimensionSpec(
        names=names,
        initial_values=initial_values,
        bounds=bounds,
        joint_mapping=joint_mapping,
    )


def _compute_bounds(
    value: float,
    factor: float,
    min_length: float,
) -> tuple[float, float]:
    """Compute bounds for a dimension value."""
    lower = max(min_length, value / factor)
    upper = value * factor
    return (lower, upper)


def extract_dimensions_with_custom_bounds(
    pylink_data: dict,
    custom_bounds: dict[str, tuple[float, float]],
) -> DimensionSpec:
    """
    Extract dimensions with user-specified bounds.

    Args:
        pylink_data: Full pylink document
        custom_bounds: Dict of {dimension_name: (min, max)}
                       Dimensions not in this dict use default bounds.

    Returns:
        DimensionSpec with custom bounds applied
    """
    spec = extract_dimensions(pylink_data)

    # Override bounds where specified
    for i, name in enumerate(spec.names):
        if name in custom_bounds:
            spec.bounds[i] = custom_bounds[name]

    return spec


# =============================================================================
# Dimension Application
# =============================================================================

def apply_dimensions(
    pylink_data: dict,
    dimension_values: dict[str, float],
    dimension_spec: DimensionSpec | None = None,
) -> dict:
    """
    Apply dimension values to pylink_data, returning a new copy.

    Updates the link lengths in the pylinkage.joints list based on
    the provided dimension values.

    Args:
        pylink_data: Original pylink document (not modified)
        dimension_values: Dict of {dimension_name: new_value}
        dimension_spec: Optional DimensionSpec for validation/mapping.
                        If not provided, uses naming convention to infer mapping.

    Returns:
        New pylink_data dict with updated dimensions

    Example:
        >>> updated = apply_dimensions(pylink_data, {"B_distance": 2.5})
        >>> # Joint B now has distance=2.5
    """
    # Deep copy to avoid mutation
    updated = copy.deepcopy(pylink_data)

    # Build joint lookup
    joints_data = updated.get('pylinkage', {}).get('joints', [])
    joint_by_name = {j['name']: j for j in joints_data}

    # Get mapping from dimension_spec or infer from naming convention
    if dimension_spec is not None:
        mapping = dimension_spec.joint_mapping
    else:
        mapping = _infer_mapping_from_names(dimension_values.keys())

    # Apply each dimension
    for dim_name, new_value in dimension_values.items():
        if dim_name in mapping:
            joint_name, prop_name = mapping[dim_name]
            if joint_name in joint_by_name:
                joint_by_name[joint_name][prop_name] = new_value
        else:
            # Try to infer from naming convention
            parts = dim_name.rsplit('_', 1)
            if len(parts) == 2:
                joint_name, prop_name = parts
                if joint_name in joint_by_name:
                    joint_by_name[joint_name][prop_name] = new_value

    return updated


def apply_dimensions_from_array(
    pylink_data: dict,
    values: tuple[float, ...],
    dimension_spec: DimensionSpec,
) -> dict:
    """
    Apply dimension values from a numeric array (for optimizer callbacks).

    Args:
        pylink_data: Original pylink document
        values: Tuple/list of values in same order as dimension_spec.names
        dimension_spec: Spec defining the mapping

    Returns:
        Updated pylink_data with new dimensions
    """
    if len(values) != len(dimension_spec.names):
        raise ValueError(
            f'Expected {len(dimension_spec.names)} values, got {len(values)}',
        )

    dimension_values = dict(zip(dimension_spec.names, values))
    return apply_dimensions(pylink_data, dimension_values, dimension_spec)


def _infer_mapping_from_names(
    dim_names: list[str],
) -> dict[str, tuple[str, str]]:
    """
    Infer joint mapping from dimension names using naming convention.

    Expected format: "{joint_name}_{property_name}"
    e.g., "B_distance" -> ("B", "distance")
          "C_distance0" -> ("C", "distance0")
    """
    mapping = {}
    for name in dim_names:
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            joint_name, prop_name = parts
            mapping[name] = (joint_name, prop_name)
    return mapping


# =============================================================================
# Error Computation
# =============================================================================

def compute_trajectory_error(
    computed_trajectory: list[tuple[float, float]],
    target: TargetTrajectory,
    metric: str = 'mse',
) -> float:
    """
    Compute error between computed trajectory and target.

    Args:
        computed_trajectory: List of (x, y) positions from simulation
        target: TargetTrajectory with target positions and optional weights
        metric: Error metric to return:
                - "mse": Mean squared error (default)
                - "rmse": Root mean squared error
                - "total": Total squared error (sum)
                - "max": Maximum error at any timestep

    Returns:
        Error value (lower is better, 0 is perfect match)

    Raises:
        ValueError: If trajectories have different lengths
    """
    if len(computed_trajectory) != len(target.positions):
        raise ValueError(
            f'Trajectory length mismatch: computed={len(computed_trajectory)}, '
            f'target={len(target.positions)}',
        )

    metrics = compute_trajectory_error_detailed(computed_trajectory, target)

    if metric == 'mse':
        return metrics.mse
    elif metric == 'rmse':
        return metrics.rmse
    elif metric == 'total':
        return metrics.total_error
    elif metric == 'max':
        return metrics.max_error
    else:
        raise ValueError(f'Unknown metric: {metric}')


def compute_trajectory_error_detailed(
    computed_trajectory: list[tuple[float, float]],
    target: TargetTrajectory,
    phase_invariant: bool = False,
) -> ErrorMetrics:
    """
    Compute detailed error metrics between computed and target trajectory.

    Args:
        computed_trajectory: List of (x, y) positions from simulation
        target: TargetTrajectory with target positions and weights
        phase_invariant: If True, find optimal phase alignment before scoring.
            CRITICAL: Use this when trajectories may start at different points!
            Without this, identical paths can have huge errors if out of phase.

    Returns:
        ErrorMetrics with total, mse, rmse, max, and per-step errors

    Note:
        If trajectories have different lengths, use
        trajectory_utils.resample_trajectory() first.
    """
    n = len(computed_trajectory)
    if n != len(target.positions):
        raise ValueError(
            f'Trajectory length mismatch: computed={n}, target={len(target.positions)}. '
            f'Use trajectory_utils.resample_trajectory() to match lengths.',
        )

    if n == 0:
        return ErrorMetrics(
            total_error=0.0,
            mse=0.0,
            rmse=0.0,
            max_error=0.0,
            per_step_errors=[],
        )

    # Apply phase alignment if requested
    if phase_invariant:
        from pylink_tools.trajectory_utils import compute_phase_aligned_distance
        result = compute_phase_aligned_distance(
            target.positions,
            computed_trajectory,
            method='rotation',
            return_aligned=True,
        )
        # Use aligned trajectory for detailed error computation
        computed_trajectory = result.aligned_trajectory

    weights = target.weights or [1.0] * n

    per_step_errors = []
    weighted_squared_sum = 0.0
    total_weight = 0.0

    for i, (computed, target_pos) in enumerate(zip(computed_trajectory, target.positions)):
        # Euclidean distance
        dx = computed[0] - target_pos[0]
        dy = computed[1] - target_pos[1]
        dist = np.sqrt(dx * dx + dy * dy)

        per_step_errors.append(dist)
        weighted_squared_sum += weights[i] * (dist ** 2)
        total_weight += weights[i]

    # Compute metrics
    total_error = weighted_squared_sum
    mse = weighted_squared_sum / total_weight if total_weight > 0 else 0.0
    rmse = np.sqrt(mse)
    max_error = max(per_step_errors) if per_step_errors else 0.0

    return ErrorMetrics(
        total_error=total_error,
        mse=mse,
        rmse=rmse,
        max_error=max_error,
        per_step_errors=per_step_errors,
    )


def evaluate_linkage_fit(
    pylink_data: dict,
    target: TargetTrajectory,
    n_steps: int | None = None,
    phase_invariant: bool = False,
) -> ErrorMetrics:
    """
    Evaluate how well a linkage fits a target trajectory.

    This is a convenience function that:
      1. Computes the trajectory for the linkage
      2. Extracts the target joint's path
      3. Computes error metrics (optionally phase-aligned)

    Args:
        pylink_data: Full pylink document
        target: Target trajectory to match
        n_steps: Number of simulation steps (uses target.n_steps if not provided)
        phase_invariant: If True, find optimal phase alignment before scoring.
            RECOMMENDED for real-world targets where starting point may differ.

    Returns:
        ErrorMetrics describing the fit quality

    Note:
        n_steps is CRITICAL - it determines simulation resolution and MUST
        match target.n_steps for accurate scoring. Use a higher value (48-96)
        for better accuracy at the cost of slower optimization.
    """
    from pylink_tools.kinematic import compute_trajectory

    # Set n_steps if not provided
    if n_steps is None:
        n_steps = target.n_steps

    # Update n_steps in pylink_data
    eval_data = copy.deepcopy(pylink_data)
    eval_data['n_steps'] = n_steps

    # CRITICAL: Use skip_sync=True to use the dimension values we set,
    # not overwrite them from stale meta.joints visual positions!
    result = compute_trajectory(eval_data, verbose=False, skip_sync=True)

    if not result.success:
        # Return infinite error if simulation fails
        return ErrorMetrics(
            total_error=float('inf'),
            mse=float('inf'),
            rmse=float('inf'),
            max_error=float('inf'),
            per_step_errors=[float('inf')] * n_steps,
        )

    # Get target joint trajectory
    joint_name = target.joint_name
    if joint_name not in result.trajectories:
        raise ValueError(f"Target joint '{joint_name}' not found in trajectories")

    computed = result.trajectories[joint_name]
    # Convert from [[x,y], ...] to [(x,y), ...]
    computed_tuples = [tuple(pos) for pos in computed]

    return compute_trajectory_error_detailed(computed_tuples, target, phase_invariant=phase_invariant)


# =============================================================================
# Utility Functions
# =============================================================================

def dimensions_to_dict(
    values: tuple[float, ...],
    spec: DimensionSpec,
) -> dict[str, float]:
    """Convert dimension array to named dict."""
    return dict(zip(spec.names, values))


def dict_to_dimensions(
    values_dict: dict[str, float],
    spec: DimensionSpec,
) -> tuple[float, ...]:
    """Convert named dict to dimension array (in spec order)."""
    return tuple(
        values_dict.get(name, spec.initial_values[i])
        for i, name in enumerate(spec.names)
    )


def validate_bounds(
    values: tuple[float, ...],
    spec: DimensionSpec,
) -> list[str]:
    """
    Check if values are within bounds.

    Returns:
        List of violation messages (empty if all valid)
    """
    violations = []
    for i, (value, (lower, upper)) in enumerate(zip(values, spec.bounds)):
        if value < lower:
            violations.append(f'{spec.names[i]}: {value} < {lower} (min)')
        elif value > upper:
            violations.append(f'{spec.names[i]}: {value} > {upper} (max)')
    return violations


# =============================================================================
# Convergence Logging / Analysis
# =============================================================================

@dataclass
class ConvergenceStats:
    """
    Statistics about optimization convergence.

    Useful for analyzing and debugging optimization runs.
    """
    initial_error: float
    final_error: float
    best_error: float
    improvement_pct: float
    n_iterations: int
    n_evaluations: int
    converged: bool
    history: list[float]
    improvement_per_iteration: list[float]

    def to_dict(self) -> dict:
        return {
            'initial_error': self.initial_error,
            'final_error': self.final_error,
            'best_error': self.best_error,
            'improvement_pct': self.improvement_pct,
            'n_iterations': self.n_iterations,
            'n_evaluations': self.n_evaluations,
            'converged': self.converged,
            'history': self.history,
            'improvement_per_iteration': self.improvement_per_iteration,
        }


def analyze_convergence(
    history: list[float],
    tolerance: float = 1e-6,
) -> ConvergenceStats:
    """
    Analyze convergence history from an optimization run.

    Args:
        history: List of error values (one per iteration)
        tolerance: Convergence tolerance

    Returns:
        ConvergenceStats with analysis results
    """
    if not history:
        return ConvergenceStats(
            initial_error=0.0,
            final_error=0.0,
            best_error=0.0,
            improvement_pct=0.0,
            n_iterations=0,
            n_evaluations=0,
            converged=False,
            history=[],
            improvement_per_iteration=[],
        )

    initial = history[0]
    final = history[-1]
    best = min(history)

    # Handle inf/nan
    if initial == float('inf') or np.isnan(initial):
        improvement_pct = 0.0
    elif initial == 0:
        improvement_pct = 0.0 if final == 0 else -100.0
    else:
        improvement_pct = (1 - final / initial) * 100

    # Compute per-iteration improvement
    improvements = []
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        if prev == float('inf') or curr == float('inf'):
            improvements.append(0.0)
        elif prev == 0:
            improvements.append(0.0)
        else:
            improvements.append(prev - curr)

    # Check if converged (change < tolerance)
    converged = False
    if len(history) >= 2:
        recent_change = abs(history[-1] - history[-2])
        converged = recent_change < tolerance

    return ConvergenceStats(
        initial_error=initial,
        final_error=final,
        best_error=best,
        improvement_pct=improvement_pct,
        n_iterations=len(history) - 1,  # First entry is initial state
        n_evaluations=len(history),
        converged=converged,
        history=history,
        improvement_per_iteration=improvements,
    )


def format_convergence_report(
    result: OptimizationResult,
    include_history: bool = False,
) -> str:
    """
    Format a human-readable convergence report.

    Args:
        result: OptimizationResult from an optimization run
        include_history: Include full iteration history

    Returns:
        Formatted string report
    """
    lines = [
        '=' * 50,
        'OPTIMIZATION CONVERGENCE REPORT',
        '=' * 50,
        f"Status: {'SUCCESS' if result.success else 'FAILED'}",
        f'Initial Error: {result.initial_error:.6f}',
        f'Final Error:   {result.final_error:.6f}',
    ]

    if result.initial_error > 0 and result.initial_error != float('inf'):
        improvement = (1 - result.final_error / result.initial_error) * 100
        lines.append(f'Improvement:   {improvement:.1f}%')

    lines.append(f'Iterations:    {result.iterations}')

    if result.optimized_dimensions:
        lines.append('')
        lines.append('Optimized Dimensions:')
        for name, value in result.optimized_dimensions.items():
            lines.append(f'  {name}: {value:.4f}')

    if result.convergence_history and include_history:
        lines.append('')
        lines.append('Convergence History:')
        for i, err in enumerate(result.convergence_history):
            lines.append(f'  [{i:3d}] {err:.6f}')

    if result.error:
        lines.append('')
        lines.append(f'Error: {result.error}')

    lines.append('=' * 50)
    return '\n'.join(lines)


def log_optimization_progress(
    iteration: int,
    current_error: float,
    best_error: float,
    dimensions: tuple[float, ...] | None = None,
    dimension_names: list[str] | None = None,
) -> str:
    """
    Format a single iteration's progress for logging.

    Args:
        iteration: Current iteration number
        current_error: Error at this iteration
        best_error: Best error seen so far
        dimensions: Current dimension values (optional)
        dimension_names: Names for dimensions (optional)

    Returns:
        Formatted progress string
    """
    parts = [f'[{iteration:4d}]', f'error={current_error:.6f}', f'best={best_error:.6f}']

    if dimensions is not None and dimension_names is not None:
        dim_str = ', '.join(f'{n}={v:.2f}' for n, v in zip(dimension_names[:3], dimensions[:3]))
        if len(dimensions) > 3:
            dim_str += '...'
        parts.append(f'dims=({dim_str})')

    return ' | '.join(parts)


# =============================================================================
# Optimization Wrappers
# =============================================================================

def create_fitness_function(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec,
    metric: str = 'mse',
    verbose: bool = False,
    phase_invariant: bool = False,
) -> Callable[[tuple[float, ...]], float]:
    """
    Create a fitness function for optimization.

    Returns a callable that takes dimension values and returns error score.
    This is the core objective function used by all optimizers.

    Args:
        pylink_data: Base pylink document (will be modified with new dimensions)
        target: Target trajectory to match
        dimension_spec: Specification of optimizable dimensions
        metric: Error metric ("mse", "rmse", "total", "max")
        verbose: If True, print evaluation info
        phase_invariant: If True, find optimal phase alignment before scoring.
            RECOMMENDED when target may have different starting point than
            simulation. Adds O(n) overhead per evaluation but prevents
            false errors from phase mismatch.

    Returns:
        Callable that takes dimension tuple and returns float error

    Example:
        >>> fitness = create_fitness_function(pylink_data, target, spec)
        >>> error = fitness((20.0, 50.0, 40.0))  # Evaluate with these dimensions

    Note on phase_invariant:
        Without phase alignment, two identical paths starting at different
        points can have HUGE errors (e.g., MSE=200 for a 90° phase shift).
        Use phase_invariant=True when:
        - Target comes from external source (captured data, hand-drawn)
        - Target was generated with different initial conditions
        - You're not sure about phase alignment
    """
    eval_count = [0]  # Mutable counter for tracking evaluations

    def fitness(dimensions: tuple[float, ...]) -> float:
        """
        Evaluate linkage fitness with given dimensions.

        Args:
            dimensions: Tuple of dimension values in spec order

        Returns:
            Error score (lower is better, inf for invalid configurations)
        """
        eval_count[0] += 1

        try:
            # Apply dimensions to get updated pylink_data
            updated_data = apply_dimensions_from_array(
                pylink_data, dimensions, dimension_spec,
            )

            # Evaluate fit (with optional phase alignment)
            metrics = evaluate_linkage_fit(
                updated_data, target, phase_invariant=phase_invariant,
            )

            # Get requested metric
            if metric == 'mse':
                error = metrics.mse
            elif metric == 'rmse':
                error = metrics.rmse
            elif metric == 'total':
                error = metrics.total_error
            elif metric == 'max':
                error = metrics.max_error
            else:
                error = metrics.mse

            if verbose and eval_count[0] % 50 == 0:
                print(f'  Eval #{eval_count[0]}: error={error:.4f}, dims={dimensions[:3]}...')

            return error

        except Exception as e:
            if verbose:
                print(f'  Eval #{eval_count[0]} failed: {e}')
            return float('inf')

    # Attach metadata to function
    fitness.eval_count = eval_count
    fitness.dimension_spec = dimension_spec
    fitness.target = target
    fitness.phase_invariant = phase_invariant

    return fitness


def create_pylinkage_fitness_function(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec,
    metric: str = 'mse',
) -> Callable:
    """
    Create a fitness function compatible with pylinkage's optimization API.

    This wraps our fitness function with pylinkage's @kinematic_minimization
    decorator pattern for use with pl.particle_swarm_optimization.

    Note: This requires building a Linkage object and is more complex.
    For simpler use, prefer run_scipy_optimization().

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Specification of optimizable dimensions
        metric: Error metric

    Returns:
        Fitness function compatible with pylinkage PSO
    """
    import pylinkage as pl

    @pl.kinematic_minimization
    def fitness_func(loci, **_kwargs):
        """
        Compute error between loci and target trajectory.

        loci: List of joint trajectories, each is list of (x, y) positions
        """
        # Find target joint in loci
        # loci is ordered by linkage.joints order (non-static joints only)
        # We need to figure out which index corresponds to our target joint

        # Get trajectory for last joint (typically the output)
        # In a 4-bar, this is usually the coupler point
        target_locus = []
        for step_positions in loci:
            # step_positions is ((x1,y1), (x2,y2), ...) for all joints at this step
            # We need the position of our target joint
            # For now, use the last non-static joint
            if step_positions and len(step_positions) > 0:
                target_locus.append(step_positions[-1])

        if len(target_locus) != len(target.positions):
            return float('inf')

        # Compute error
        total_error = 0.0
        weights = target.weights or [1.0] * len(target.positions)

        for i, (computed, target_pos) in enumerate(zip(target_locus, target.positions)):
            if computed is None or computed[0] is None:
                return float('inf')
            dx = computed[0] - target_pos[0]
            dy = computed[1] - target_pos[1]
            total_error += weights[i] * (dx * dx + dy * dy)

        if metric == 'mse':
            return total_error / len(target.positions)
        elif metric == 'total':
            return total_error
        else:
            return total_error / len(target.positions)

    return fitness_func


def run_scipy_optimization(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    method: str = 'L-BFGS-B',
    metric: str = 'mse',
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Run optimization using scipy.optimize.minimize.

    This is often faster than PSO for well-behaved problems and supports
    gradient-based methods like L-BFGS-B.

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Dimensions to optimize (extracted if not provided)
        method: Scipy optimizer method ("L-BFGS-B", "SLSQP", "Powell", "Nelder-Mead")
        metric: Error metric for fitness function
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        verbose: Print progress

    Returns:
        OptimizationResult with optimized dimensions

    Example:
        >>> result = run_scipy_optimization(pylink_data, target)
        >>> if result.success:
        ...     print(f"Final error: {result.final_error}")
        ...     print(f"Optimized: {result.optimized_dimensions}")
    """
    from scipy.optimize import minimize

    # Extract dimensions if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    # Create fitness function
    fitness = create_fitness_function(
        pylink_data, target, dimension_spec, metric=metric, verbose=verbose,
    )

    # Compute initial error
    initial_values = tuple(dimension_spec.initial_values)
    initial_error = fitness(initial_values)

    if verbose:
        print(f'Starting scipy optimization ({method})')
        print(f'  Dimensions: {len(dimension_spec)}')
        print(f'  Initial error: {initial_error:.4f}')

    # Convert bounds to scipy format: [(low, high), ...]
    scipy_bounds = dimension_spec.bounds

    # Track convergence history
    history = [initial_error]

    def callback(xk):
        error = fitness(tuple(xk))
        history.append(error)
        if verbose and len(history) % 10 == 0:
            print(f'  Iteration {len(history)}: error={error:.4f}')

    # Run optimization
    try:
        # Build options dict based on method
        options = {'maxiter': max_iterations}

        # Different methods use different tolerance parameter names
        if method in ('L-BFGS-B', 'SLSQP'):
            options['ftol'] = tolerance
        elif method == 'Powell':
            options['ftol'] = tolerance
        elif method == 'Nelder-Mead':
            options['xatol'] = tolerance
            options['fatol'] = tolerance

        result = minimize(
            fun=fitness,
            x0=initial_values,
            method=method,
            bounds=scipy_bounds if method in ('L-BFGS-B', 'SLSQP') else None,
            options=options,
            callback=callback,
        )

        # Extract results
        optimized_values = tuple(result.x)
        final_error = result.fun

        # Build optimized pylink_data
        optimized_pylink_data = apply_dimensions_from_array(
            pylink_data, optimized_values, dimension_spec,
        )

        # Create dimension dict
        optimized_dims = dimensions_to_dict(optimized_values, dimension_spec)

        if verbose:
            print(f'  Converged: {result.success}')
            print(f'  Final error: {final_error:.4f}')
            print(f'  Iterations: {result.nit}')
            print(f'  Function evals: {result.nfev}')

        return OptimizationResult(
            success=result.success,
            optimized_dimensions=optimized_dims,
            optimized_pylink_data=optimized_pylink_data,
            initial_error=initial_error,
            final_error=final_error,
            iterations=result.nit,
            convergence_history=history,
        )

    except Exception as e:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'Optimization failed: {str(e)}',
        )


def run_pso_optimization(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    n_particles: int = 32,
    iterations: int = 512,
    metric: str = 'mse',
    verbose: bool = False,
    phase_invariant: bool = False,
) -> OptimizationResult:
    """
    Run Particle Swarm Optimization to fit linkage to target trajectory.

    PSO is a population-based optimizer that's good for:
      - Non-convex problems
      - Escaping local minima
      - Problems where gradients are unavailable

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Dimensions to optimize (extracted if not provided)
        n_particles: Number of particles in swarm
        iterations: Number of PSO iterations
        metric: Error metric for fitness function
        verbose: Print progress
        phase_invariant: If True, use phase-aligned scoring.
            RECOMMENDED when target may start at different phase than simulation.
            Prevents false errors from phase mismatch at cost of O(n) per eval.

    Returns:
        OptimizationResult with optimized dimensions

    Example:
        >>> result = run_pso_optimization(pylink_data, target, n_particles=50, iterations=100)
        >>> if result.success:
        ...     updated_linkage = result.optimized_pylink_data

    Note on phase_invariant:
        Set to True when working with:
        - External/captured target trajectories
        - Hand-drawn or digitized paths
        - Any case where starting point alignment is uncertain
    """
    # Extract dimensions if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    # Create fitness function
    fitness = create_fitness_function(
        pylink_data, target, dimension_spec,
        metric=metric, verbose=verbose, phase_invariant=phase_invariant,
    )

    # Compute initial error
    initial_values = tuple(dimension_spec.initial_values)
    initial_error = fitness(initial_values)

    if verbose:
        print('Starting PSO optimization')
        print(f'  Dimensions: {len(dimension_spec)}')
        print(f'  Particles: {n_particles}')
        print(f'  Iterations: {iterations}')
        print(f'  Initial error: {initial_error:.4f}')

    # Get bounds
    bounds = dimension_spec.get_bounds_tuple()

    # Run PSO
    try:
        best_score, best_dims, history = _run_pso_internal(
            fitness_func=fitness,
            initial_values=initial_values,
            bounds=bounds,
            n_particles=n_particles,
            iterations=iterations,
            verbose=verbose,
        )

        # Build optimized pylink_data
        optimized_pylink_data = apply_dimensions_from_array(
            pylink_data, best_dims, dimension_spec,
        )

        # Create dimension dict
        optimized_dims = dimensions_to_dict(best_dims, dimension_spec)

        if verbose:
            print(f'  Final error: {best_score:.4f}')
            print(f'  Improvement: {(1 - best_score/initial_error)*100:.1f}%')

        return OptimizationResult(
            success=True,
            optimized_dimensions=optimized_dims,
            optimized_pylink_data=optimized_pylink_data,
            initial_error=initial_error,
            final_error=best_score,
            iterations=iterations,
            convergence_history=history,
        )

    except Exception as e:
        import traceback
        if verbose:
            traceback.print_exc()
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'PSO optimization failed: {str(e)}',
        )


def run_pylinkage_pso(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    n_particles: int = 32,
    iterations: int = 512,
    metric: str = 'mse',
    verbose: bool = False,
) -> OptimizationResult:
    """
    Run Particle Swarm Optimization using pylinkage's native PSO implementation.

    This uses pylinkage.particle_swarm_optimization with the
    @kinematic_minimization decorator, which handles:
      - Building and validating the linkage at each iteration
      - Rejecting invalid configurations (returns inf penalty)
      - Efficient loci computation

    **EXPERIMENTAL**: This method explores the bounds space broadly, which means
    many random dimension combinations may produce invalid linkages (returning inf).
    For most use cases, prefer:
      - run_pso_optimization(): Custom PSO that starts from valid config
      - run_scipy_optimization(): Gradient-based, faster convergence

    Note: pylinkage uses pyswarms with ring topology which requires
    n_particles >= 18 (neighborhood size k=17). Smaller values will
    be automatically increased.

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Dimensions to optimize (extracted if not provided)
        n_particles: Number of particles in swarm (minimum 20 for pylinkage)
        iterations: Number of PSO iterations
        metric: Error metric for fitness function
        verbose: Print progress

    Returns:
        OptimizationResult with optimized dimensions
    """
    # Pylinkage/pyswarms requires minimum particles for ring topology
    MIN_PARTICLES = 20
    if n_particles < MIN_PARTICLES:
        if verbose:
            print(f'  Note: Increasing n_particles from {n_particles} to {MIN_PARTICLES} (pylinkage minimum)')
        n_particles = MIN_PARTICLES
    import pylinkage as pl
    from pylink_tools.kinematic import (
        build_joint_objects, make_linkage, compute_proper_solve_order,
    )

    # Extract dimensions if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    # Build the base linkage
    pylinkage_data = pylink_data.get('pylinkage', {})
    joints_data = pylinkage_data.get('joints', [])
    meta = pylink_data.get('meta', {})
    meta_joints = meta.get('joints', {})
    n_steps = target.n_steps

    solve_order = compute_proper_solve_order(joints_data, verbose=False)
    joint_info = {j['name']: j for j in joints_data}
    angle_per_step = 2 * np.pi / n_steps

    joint_objects = build_joint_objects(
        joints_data, solve_order, joint_info, meta_joints, angle_per_step, verbose=False,
    )

    linkage, error = make_linkage(joint_objects, solve_order, pylinkage_data.get('name', 'opt'))

    if error:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=f'Failed to build linkage: {error}',
        )

    # Save initial state
    init_pos = linkage.get_coords()
    init_constraints = linkage.get_num_constraints()

    if verbose:
        print('Starting pylinkage PSO optimization')
        print(f'  Dimensions: {len(dimension_spec)}')
        print(f'  Initial constraints: {init_constraints}')
        print(f'  Particles: {n_particles}')
        print(f'  Iterations: {iterations}')

    # Find target joint index in linkage.joints
    target_joint_name = target.joint_name
    target_joint_idx = None
    for idx, joint in enumerate(linkage.joints):
        if joint.name == target_joint_name:
            target_joint_idx = idx
            break

    if target_joint_idx is None:
        # Default to last joint
        target_joint_idx = len(linkage.joints) - 1
        if verbose:
            print(f"  Warning: Target joint '{target_joint_name}' not found, using last joint")

    # Create fitness function with @kinematic_minimization decorator
    @pl.kinematic_minimization
    def fitness_func(loci, **_kwargs):
        """
        Compute error between linkage trajectory and target.

        loci: List of step positions, each step is tuple of (x,y) for each joint
        """
        # Extract target joint trajectory from loci
        # loci[step_idx] = ((x1,y1), (x2,y2), ...) positions of all joints at this step
        try:
            computed_trajectory = []
            for step_positions in loci:
                if target_joint_idx < len(step_positions):
                    pos = step_positions[target_joint_idx]
                    computed_trajectory.append(pos)
                else:
                    return float('inf')

            if len(computed_trajectory) != len(target.positions):
                return float('inf')

            # Compute weighted sum of squared errors
            total_error = 0.0
            weights = target.weights or [1.0] * len(target.positions)

            for i, (computed, target_pos) in enumerate(zip(computed_trajectory, target.positions)):
                if computed is None or computed[0] is None:
                    return float('inf')
                dx = computed[0] - target_pos[0]
                dy = computed[1] - target_pos[1]
                total_error += weights[i] * (dx * dx + dy * dy)

            if metric == 'mse':
                return total_error / len(target.positions)
            else:
                return total_error

        except Exception:
            return float('inf')

    # Get bounds in pylinkage format
    bounds = dimension_spec.get_bounds_tuple()

    # Compute initial error
    initial_error = float('inf')
    try:
        linkage.rebuild()
        loci_init = list(linkage.step(iterations=n_steps))
        if loci_init:
            computed = [
                step[target_joint_idx] if target_joint_idx < len(step) else (0, 0)
                for step in loci_init
            ]
            metrics = compute_trajectory_error_detailed(computed, target)
            initial_error = metrics.mse if metric == 'mse' else metrics.total_error
    except Exception:
        pass

    if verbose:
        print(f'  Initial error: {initial_error:.4f}')

    # Run pylinkage PSO
    # CRITICAL: Use 'init_pos' to set initial particle positions near known valid config
    # Without this, particles initialize randomly in bounds and never find valid configs
    # Shape: (n_particles, dimensions) - cluster particles around valid starting point
    init_constraints_array = np.array(init_constraints)
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])
    bound_range = upper_bounds - lower_bounds

    # Initialize particles with small perturbations around the known valid position
    # Use ~10% of bounds range as initial spread (much tighter than random initialization)
    init_spread = 0.10
    init_pos = np.zeros((n_particles, len(init_constraints)))
    for i in range(n_particles):
        if i == 0:
            # First particle is exactly at the known valid position
            init_pos[i] = init_constraints_array
        else:
            # Other particles: small random perturbations around valid position
            perturbation = (np.random.random(len(init_constraints)) - 0.5) * 2 * init_spread * bound_range
            init_pos[i] = np.clip(
                init_constraints_array + perturbation,
                lower_bounds,
                upper_bounds,
            )

    if verbose:
        print(f'  Init spread: {init_spread*100:.0f}% of bounds range')

    try:
        results = pl.particle_swarm_optimization(
            eval_func=fitness_func,
            linkage=linkage,
            bounds=bounds,
            n_particles=n_particles,
            iters=iterations,
            order_relation=min,
            verbose=verbose,
            init_pos=init_pos,  # Pass initial positions through kwargs to pyswarms
        )

        # Extract best result
        # Results is list of (score, dimensions, coords)
        if results and len(results) > 0:
            best_score, best_dims_tuple, best_coords = results[0]

            # Convert to our format
            best_dims = tuple(best_dims_tuple) if not isinstance(best_dims_tuple, tuple) else best_dims_tuple

            # Build optimized pylink_data
            optimized_pylink_data = apply_dimensions_from_array(
                pylink_data, best_dims, dimension_spec,
            )

            optimized_dims = dimensions_to_dict(best_dims, dimension_spec)

            if verbose:
                print(f'  Final error: {best_score:.4f}')
                if initial_error != float('inf'):
                    print(f'  Improvement: {(1 - best_score/initial_error)*100:.1f}%')

            return OptimizationResult(
                success=True,
                optimized_dimensions=optimized_dims,
                optimized_pylink_data=optimized_pylink_data,
                initial_error=initial_error,
                final_error=best_score,
                iterations=iterations,
                convergence_history=None,  # pylinkage doesn't provide this easily
            )
        else:
            return OptimizationResult(
                success=False,
                optimized_dimensions={},
                initial_error=initial_error,
                error='PSO returned no results',
            )

    except Exception as e:
        import traceback
        if verbose:
            traceback.print_exc()
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'pylinkage PSO failed: {str(e)}',
        )
    finally:
        # Reset linkage to initial state
        try:
            linkage.set_num_constraints(init_constraints)
            linkage.set_coords(init_pos)
        except Exception:
            pass


def _run_pso_internal(
    fitness_func: Callable[[tuple[float, ...]], float],
    initial_values: tuple[float, ...],
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
    n_particles: int = 32,
    iterations: int = 512,
    w: float = 0.7,      # Inertia weight
    c1: float = 1.5,     # Cognitive parameter
    c2: float = 1.5,     # Social parameter
    verbose: bool = False,
) -> tuple[float, tuple[float, ...], list[float]]:
    """
    Internal PSO implementation.

    Standard PSO algorithm with velocity clamping and boundary handling.

    Args:
        fitness_func: Objective function to minimize
        initial_values: Starting point (used to seed one particle)
        bounds: ((lower...), (upper...)) bounds for each dimension
        n_particles: Swarm size
        iterations: Number of iterations
        w: Inertia weight (momentum)
        c1: Cognitive parameter (personal best attraction)
        c2: Social parameter (global best attraction)
        verbose: Print progress

    Returns:
        (best_score, best_position, convergence_history)
    """
    n_dims = len(initial_values)
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])

    # Initialize particles
    # First particle starts at initial values, rest are random
    positions = np.random.uniform(
        lower_bounds, upper_bounds, (n_particles, n_dims),
    )
    positions[0] = np.array(initial_values)

    # Initialize velocities (small random values)
    max_velocity = (upper_bounds - lower_bounds) * 0.2
    velocities = np.random.uniform(-max_velocity, max_velocity, (n_particles, n_dims))

    # Initialize personal bests
    personal_best_positions = positions.copy()
    personal_best_scores = np.full(n_particles, float('inf'))

    # Evaluate initial positions
    for i in range(n_particles):
        score = fitness_func(tuple(positions[i]))
        personal_best_scores[i] = score

    # Initialize global best
    best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[best_idx].copy()
    global_best_score = personal_best_scores[best_idx]

    # Track convergence
    history = [global_best_score]

    # Main PSO loop
    for iteration in range(iterations):
        for i in range(n_particles):
            # Generate random factors
            r1, r2 = np.random.random(n_dims), np.random.random(n_dims)

            # Update velocity
            cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
            social = c2 * r2 * (global_best_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Clamp velocity
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            # Update position
            positions[i] = positions[i] + velocities[i]

            # Handle boundary violations (reflect)
            for d in range(n_dims):
                if positions[i, d] < lower_bounds[d]:
                    positions[i, d] = lower_bounds[d]
                    velocities[i, d] *= -0.5
                elif positions[i, d] > upper_bounds[d]:
                    positions[i, d] = upper_bounds[d]
                    velocities[i, d] *= -0.5

            # Evaluate new position
            score = fitness_func(tuple(positions[i]))

            # Update personal best
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

        history.append(global_best_score)

        if verbose and (iteration + 1) % 10 == 0:
            print(f'  Iteration {iteration + 1}/{iterations}: best_error={global_best_score:.4f}')

    return global_best_score, tuple(global_best_position), history


def optimize_trajectory(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    custom_bounds: dict[str, tuple[float, float]] | None = None,
    method: str = 'pylinkage',
    metric: str = 'mse',
    verbose: bool = False,
    **kwargs,
) -> OptimizationResult:
    """
    Main entry point for trajectory optimization.

    Finds link lengths that make the mechanism's trajectory best match
    the target positions.

    Args:
        pylink_data: Full pylink document
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Pre-computed dimension spec (optional)
        custom_bounds: Override bounds for specific dimensions
        method: Optimization method:
                - "pylinkage": Particle Swarm Optimization (default, robust)
                - "pso": Alias for pylinkage PSO
                - "scipy": scipy.optimize.minimize with L-BFGS-B (faster)
                - "powell": scipy Powell method (gradient-free)
                - "nelder-mead": Nelder-Mead simplex (gradient-free)
        metric: Error metric ("mse", "rmse", "total", "max")
        verbose: Print progress
        **kwargs: Method-specific arguments:
                  PSO: n_particles (32), iterations (512, max 10000)
                  scipy: max_iterations (100), tolerance (1e-6)

    Returns:
        OptimizationResult with optimized dimensions and updated pylink_data

    Example:
        >>> target = TargetTrajectory(
        ...     joint_name="coupler",
        ...     positions=[(100, 120), (105, 125), ...]
        ... )
        >>> result = optimize_trajectory(pylink_data, target, method="pylinkage")
        >>> if result.success:
        ...     print(f"Reduced error from {result.initial_error:.2f} to {result.final_error:.2f}")
        ...     optimized_linkage = result.optimized_pylink_data
    """
    # Extract/prepare dimensions
    if dimension_spec is None:
        if custom_bounds:
            dimension_spec = extract_dimensions_with_custom_bounds(pylink_data, custom_bounds)
        else:
            dimension_spec = extract_dimensions(pylink_data)

    if verbose:
        print('Optimizing trajectory fit')
        print(f'  Target joint: {target.joint_name}')
        print(f'  Target steps: {target.n_steps}')
        print(f'  Dimensions: {dimension_spec.names}')
        print(f'  Method: {method}')

    # Route to appropriate optimizer
    method_lower = method.lower()

    if method_lower == 'pso':
        return run_pso_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            metric=metric,
            verbose=verbose,
            n_particles=kwargs.get('n_particles', 32),
            iterations=kwargs.get('iterations', 512),
        )

    elif method_lower == 'pylinkage':
        # NOTE: pylinkage's native PSO (run_pylinkage_pso) has issues with particle initialization
        # where random exploration almost never finds valid linkage configurations.
        # Until fixed, route to our custom PSO which works reliably.
        if verbose:
            print('  Note: Using custom PSO (native pylinkage PSO has initialization issues)')
        return run_pso_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            metric=metric,
            verbose=verbose,
            n_particles=kwargs.get('n_particles', 32),
            iterations=kwargs.get('iterations', 512),
        )

    elif method_lower in ('scipy', 'l-bfgs-b'):
        return run_scipy_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            method='L-BFGS-B',
            metric=metric,
            verbose=verbose,
            max_iterations=kwargs.get('max_iterations', 100),
            tolerance=kwargs.get('tolerance', 1e-6),
        )

    elif method_lower == 'powell':
        return run_scipy_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            method='Powell',
            metric=metric,
            verbose=verbose,
            max_iterations=kwargs.get('max_iterations', 100),
            tolerance=kwargs.get('tolerance', 1e-6),
        )

    elif method_lower == 'nelder-mead':
        return run_scipy_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            method='Nelder-Mead',
            metric=metric,
            verbose=verbose,
            max_iterations=kwargs.get('max_iterations', 100),
            tolerance=kwargs.get('tolerance', 1e-6),
        )

    else:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=f'Unknown optimization method: {method}',
        )
