"""
Pylinkage-native PSO optimizer for linkage trajectory optimization.

Uses pylinkage.particle_swarm_optimization with the @kinematic_minimization
decorator, which handles:
  - Building and validating the linkage at each iteration
  - Rejecting invalid configurations (returns inf penalty)
  - Efficient loci computation

Note: pylinkage uses pyswarms with ring topology which requires
n_particles >= 18 (neighborhood size k=17).

Reference: https://github.com/HugoFara/pylinkage

License: pylinkage is GPL-3.0 (note: may have commercial restrictions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylink_tools.optimization_types import (
        DimensionSpec,
        OptimizationResult,
        TargetTrajectory,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PylinkagePSOConfig:
    """
    Configuration for pylinkage PSO optimizer.

    Attributes:
        n_particles: Number of particles (minimum 20 for pylinkage)
        iterations: Number of PSO iterations
        init_mode: Particle initialization strategy:
            - 'random': Small perturbations around initial config
            - 'sobol': Pre-sample using Sobol sequence, filter valid, use best
            - 'behnken': Pre-sample using Box-Behnken design (requires 3+ dims)
        init_samples: Number of samples for presampling modes
        init_spread: Spread factor for random initialization (fraction of bounds)
    """

    n_particles: int = 32
    iterations: int = 512
    init_mode: str = "random"
    init_samples: int = 128
    init_spread: float = 0.25


# =============================================================================
# Main Interface
# =============================================================================


def run_pylinkage_pso(
    pylink_data: dict,
    target: "TargetTrajectory",
    dimension_spec: "DimensionSpec | None" = None,
    config: PylinkagePSOConfig | None = None,
    metric: str = "mse",
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal["rotation", "fft", "frechet"] = "rotation",
    **kwargs,
) -> "OptimizationResult":
    """
    Run PSO using pylinkage's native implementation.

    This uses pylinkage.particle_swarm_optimization which handles:
      - Building and validating the linkage at each iteration
      - Rejecting invalid configurations (returns inf penalty)
      - Efficient loci computation

    Note: pylinkage uses pyswarms with ring topology which requires
    n_particles >= 18 (neighborhood size k=17). Smaller values will
    be automatically increased to 20.

    Args:
        pylink_data: Base pylink document with linkage configuration
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Dimensions to optimize (extracted if not provided)
        config: Pylinkage PSO configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring.
            RECOMMENDED when target may start at different phase than simulation.
            Prevents false errors from phase mismatch at cost of O(n) per eval.
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        **kwargs: Additional arguments for interface compatibility:
            - n_particles, iterations, init_mode, init_samples: Override config

    Returns:
        OptimizationResult with:
            - success: True if optimization completed
            - optimized_dimensions: Best found dimension values
            - optimized_pylink_data: Updated linkage with optimized dimensions
            - initial_error: Error before optimization
            - final_error: Best achieved error
            - iterations: Number of iterations

    Example:
        >>> from optimizers import run_pylinkage_pso, PylinkagePSOConfig
        >>> config = PylinkagePSOConfig(n_particles=50, init_mode='sobol')
        >>> result = run_pylinkage_pso(pylink_data, target, config=config)

    Note on init_mode:
        Using 'sobol' or 'behnken' presampling can dramatically improve
        convergence for pylinkage PSO, which historically struggled with
        random initialization landing in invalid regions.
    """
    import pylinkage as pl

    from pylink_tools.kinematic import (
        build_joint_objects,
        compute_proper_solve_order,
        make_linkage,
    )
    from pylink_tools.optimization_helpers import (
        apply_dimensions_from_array,
        dimensions_to_dict,
        extract_dimensions,
        presample_valid_positions,
    )
    from pylink_tools.optimization_types import OptimizationResult
    from pylink_tools.optimize import (
        compute_trajectory_error_detailed,
        compute_trajectory_error_fast,
    )

    # Use default config if not provided
    if config is None:
        config = PylinkagePSOConfig()

    # Allow kwargs to override config
    n_particles = kwargs.get("n_particles", config.n_particles)
    iterations = kwargs.get("iterations", config.iterations)
    init_mode = kwargs.get("init_mode", config.init_mode)
    init_samples = kwargs.get("init_samples", config.init_samples)
    init_spread = kwargs.get("init_spread", config.init_spread)

    # Pylinkage/pyswarms requires minimum particles for ring topology
    MIN_PARTICLES = 20
    if n_particles < MIN_PARTICLES:
        if verbose:
            logger.info(f"  Note: Increasing n_particles from {n_particles} to {MIN_PARTICLES} " "(pylinkage minimum)")
        n_particles = MIN_PARTICLES

    # Extract dimensions if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error="No optimizable dimensions found",
        )

    # Build the base linkage
    pylinkage_data = pylink_data.get("pylinkage", {})
    joints_data = pylinkage_data.get("joints", [])
    meta = pylink_data.get("meta", {})
    meta_joints = meta.get("joints", {})
    n_steps = target.n_steps

    solve_order = compute_proper_solve_order(joints_data, verbose=False)
    joint_info = {j["name"]: j for j in joints_data}
    angle_per_step = 2 * np.pi / n_steps

    joint_objects = build_joint_objects(
        joints_data,
        solve_order,
        joint_info,
        meta_joints,
        angle_per_step,
        verbose=False,
    )

    linkage, error = make_linkage(joint_objects, solve_order, pylinkage_data.get("name", "opt"))

    if error or linkage is None:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=f"Failed to build linkage: {error or 'Unknown error'}",
        )

    # Save initial state
    init_coords = linkage.get_coords()
    init_constraints = linkage.get_num_constraints()

    if verbose:
        logger.info("Starting pylinkage PSO optimization")
        logger.info(f"  Dimensions: {len(dimension_spec)}")
        logger.info(f"  Initial constraints: {init_constraints}")
        logger.info(f"  Particles: {n_particles}")
        logger.info(f"  Iterations: {iterations}")
        logger.info(f"  Phase invariant: {phase_invariant}")
        logger.info(f"  Init mode: {init_mode}")
        logger.info(f"  n_steps for simulation: {n_steps}")
        logger.info(f"  target trajectory points: {len(target.positions)}")

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
            logger.warning(f"Target joint '{target_joint_name}' not found, using last joint")

    # Track fitness function calls for debugging
    fitness_call_count = [0]

    # Create fitness function WITHOUT @kinematic_minimization decorator
    # The decorator hardcodes 96 iterations which doesn't match our n_steps
    def fitness_func(linkage_obj, params, init_pos_arg=None):
        """
        Fitness function for pylinkage PSO.

        Args:
            linkage_obj: The linkage to evaluate
            params: Dimension values to set
            init_pos_arg: Initial joint positions (optional)
        """
        fitness_call_count[0] += 1

        try:
            # Set initial coordinates if provided
            if init_pos_arg is not None:
                linkage_obj.set_coords(init_pos_arg)

            # Set the constraint parameters
            linkage_obj.set_num_constraints(params)

            # Run simulation with OUR n_steps (not pylinkage's hardcoded 96)
            try:
                linkage_obj.rebuild()
                loci = list(linkage_obj.step(iterations=n_steps))
            except Exception as e:
                if fitness_call_count[0] <= 3:
                    logger.debug(f"    [fitness #{fitness_call_count[0]}] rebuild/step failed: {e}")
                return float("inf")

            if not loci or len(loci) != n_steps:
                if fitness_call_count[0] <= 3:
                    logger.debug(
                        f"    [fitness #{fitness_call_count[0]}] loci len mismatch: "
                        f"{len(loci) if loci else 0} vs {n_steps}"
                    )
                return float("inf")

            # Extract target joint trajectory from loci
            computed_trajectory = []
            for step_positions in loci:
                if target_joint_idx < len(step_positions):
                    pos = step_positions[target_joint_idx]
                    computed_trajectory.append(pos)
                else:
                    if fitness_call_count[0] <= 3:
                        logger.debug(f"    [fitness #{fitness_call_count[0]}] " "target_joint_idx out of range")
                    return float("inf")

            if len(computed_trajectory) != len(target.positions):
                if fitness_call_count[0] <= 3:
                    logger.debug(
                        f"    [fitness #{fitness_call_count[0]}] len mismatch: "
                        f"{len(computed_trajectory)} vs {len(target.positions)}"
                    )
                return float("inf")

            # Check for invalid positions
            for idx, pos in enumerate(computed_trajectory):
                if pos is None or pos[0] is None:
                    if fitness_call_count[0] <= 3:
                        logger.debug(f"    [fitness #{fitness_call_count[0]}] " f"invalid position at idx={idx}")
                    return float("inf")

            # Use phase-invariant scoring if requested
            if phase_invariant:
                result = compute_trajectory_error_fast(
                    computed_trajectory,
                    target,
                    metric=metric,
                    phase_invariant=True,
                    phase_align_method=phase_align_method,
                )
            else:
                # Direct point-to-point comparison (faster but phase-sensitive)
                total_error = 0.0
                weights = target.weights or [1.0] * len(target.positions)

                for i, (computed, target_pos) in enumerate(zip(computed_trajectory, target.positions)):
                    dx = computed[0] - target_pos[0]
                    dy = computed[1] - target_pos[1]
                    total_error += weights[i] * (dx * dx + dy * dy)

                if metric == "mse":
                    result = total_error / len(target.positions)
                else:
                    result = total_error

            return result

        except Exception as e:
            if fitness_call_count[0] <= 3:
                logger.debug(f"    [fitness #{fitness_call_count[0]}] EXCEPTION: {e}")
            return float("inf")

    # Get bounds in pylinkage format
    bounds = dimension_spec.get_bounds_tuple()

    # Compute initial error
    initial_error = float("inf")
    try:
        linkage.rebuild()
        loci_init = list(linkage.step(iterations=n_steps))
        if loci_init:
            computed = [step[target_joint_idx] if target_joint_idx < len(step) else (0, 0) for step in loci_init]
            metrics = compute_trajectory_error_detailed(
                computed,
                target,
                phase_invariant=phase_invariant,
            )
            initial_error = metrics.mse if metric == "mse" else metrics.total_error
    except Exception:
        pass

    if verbose:
        logger.info(f"  Initial error: {initial_error:.4f}")

    # Get bounds arrays
    init_constraints_array = np.array(init_constraints)
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])
    bound_range = upper_bounds - lower_bounds

    # Log detailed bounds information
    if verbose:
        logger.info("  Dimension bounds:")
        for i, name in enumerate(dimension_spec.names):
            logger.info(
                f"    {name}: [{lower_bounds[i]:.2f}, {upper_bounds[i]:.2f}] "
                f"(initial: {init_constraints_array[i]:.2f})"
            )

    # Verify initial constraints are within bounds
    for i, name in enumerate(dimension_spec.names):
        if init_constraints_array[i] < lower_bounds[i] or init_constraints_array[i] > upper_bounds[i]:
            logger.warning(f"    WARNING: {name} initial value {init_constraints_array[i]:.2f} " "is OUTSIDE bounds!")

    # Initialize particle positions based on init_mode
    init_pos = np.zeros((n_particles, len(init_constraints)))

    if init_mode in ("sobol", "behnken"):
        # Pre-sample using DOE methods to find valid configurations
        try:
            presampled_positions, presampled_scores = presample_valid_positions(
                pylink_data=pylink_data,
                target=target,
                dimension_spec=dimension_spec,
                n_samples=init_samples,
                n_best=n_particles,
                mode=init_mode,
                metric=metric,
                phase_invariant=phase_invariant,
            )

            if len(presampled_positions) > 0:
                n_presampled = min(len(presampled_positions), n_particles)
                init_pos[:n_presampled] = presampled_positions[:n_presampled]

                # Fill remaining with perturbations around best presampled position
                if n_presampled < n_particles:
                    best_pos = presampled_positions[0]
                    fill_spread = 0.15
                    for i in range(n_presampled, n_particles):
                        perturbation = (np.random.random(len(init_constraints)) - 0.5) * 2 * fill_spread * bound_range
                        init_pos[i] = np.clip(best_pos + perturbation, lower_bounds, upper_bounds)

                if verbose:
                    logger.info(
                        f"  Using {n_presampled} pre-sampled positions " f"(best score: {presampled_scores[0]:.4f})"
                    )
            else:
                # No valid presampled positions, fall back to random
                logger.warning("Presampling found no valid positions, falling back to random init")
                init_mode = "random"
        except Exception as e:
            logger.warning(f"Presampling failed: {e}. Falling back to random init.")
            init_mode = "random"

    if init_mode == "random":
        # Original behavior: small perturbations around known valid position
        for i in range(n_particles):
            if i == 0:
                init_pos[i] = init_constraints_array
            else:
                perturbation = (np.random.random(len(init_constraints)) - 0.5) * 2 * init_spread * bound_range
                init_pos[i] = np.clip(
                    init_constraints_array + perturbation,
                    lower_bounds,
                    upper_bounds,
                )
        if verbose:
            logger.info(f"  Init spread: {init_spread * 100:.0f}% of bounds range")

    # Reset linkage to initial state before optimization
    linkage.set_num_constraints(init_constraints)
    linkage.set_coords(init_coords)

    logger.debug("  Calling pylinkage particle_swarm_optimization...")
    logger.debug(f"    bounds: lower={bounds[0]}, upper={bounds[1]}")

    try:
        results = pl.particle_swarm_optimization(
            eval_func=fitness_func,
            linkage=linkage,
            bounds=bounds,
            n_particles=n_particles,
            iters=iterations,
            order_relation=min,
            verbose=verbose,
            init_pos=init_pos,
        )

        # Extract best result
        # Results is list of (score, dimensions, coords)
        if results and len(results) > 0:
            best_score, best_dims_tuple, best_coords = results[0]

            # Convert to our format
            best_dims = tuple(best_dims_tuple) if not isinstance(best_dims_tuple, tuple) else best_dims_tuple

            # Build optimized pylink_data
            optimized_pylink_data = apply_dimensions_from_array(
                pylink_data,
                best_dims,
                dimension_spec,
            )

            optimized_dims = dimensions_to_dict(best_dims, dimension_spec)

            if verbose:
                logger.info(f"  Final error: {best_score:.4f}")
                if initial_error != float("inf") and initial_error > 0:
                    improvement = (1 - best_score / initial_error) * 100
                    logger.info(f"  Improvement: {improvement:.1f}%")

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
                error="PSO returned no results",
            )

    except Exception as e:
        if verbose:
            logger.error(f"pylinkage PSO failed: {e}", exc_info=True)
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f"pylinkage PSO failed: {str(e)}",
        )
    finally:
        # Reset linkage to initial state
        try:
            linkage.set_num_constraints(init_constraints)
            linkage.set_coords(init_coords)
        except Exception:
            pass
