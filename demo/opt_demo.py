#!/usr/bin/env python3
"""
Optimization Demo - Basic linkage trajectory optimization.

WHAT THIS DEMO DOES:
====================
1. Loads a 4-bar linkage mechanism
2. Creates a target trajectory by randomizing the link dimensions
3. Runs PSO optimization to recover the original dimensions
4. Visualizes the results

This demonstrates the core optimization workflow: given a target trajectory,
find the mechanism dimensions that produce it.

WHY RANDOMIZE TO CREATE THE TARGET:
===================================
By randomizing dimensions to create the target, we guarantee:
- The target trajectory is physically achievable (not impossible)
- We know the "ground truth" dimensions for validation
- We can measure how well the optimizer recovered them

RUN THIS DEMO:
==============
    python demo/opt_demo.py

Output saved to: user/demo/
"""
from __future__ import annotations

import json
from datetime import datetime

from configs.appconfig import USER_DIR
from demo.helpers import get_dimension_spec
from demo.helpers import load_mechanism
from demo.helpers import print_dimensions
from demo.helpers import print_mechanism_info
from demo.helpers import print_section
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimize import run_pso_optimization
from target_gen import AchievableTargetConfig
from target_gen import create_achievable_target
from target_gen import DimensionVariationConfig
from viz_tools.demo_viz import plot_convergence_comparison
from viz_tools.demo_viz import plot_dimension_bounds
from viz_tools.demo_viz import variation_plot


# =============================================================================
# CONFIGURATION - Edit these to change demo behavior
# =============================================================================

# Which mechanism to optimize (simple is easiest to understand)
MECHANISM = 'simple'  # Options: 'simple', 'intermediate', 'complex', 'leg'

# How much to randomize dimensions for the target (0.3 = ±30%)
# Smaller = easier optimization, larger = harder
VARIATION_RANGE = 0.35

# PSO optimizer settings
N_PARTICLES = 64      # More particles = better exploration, slower
N_ITERATIONS = 150    # More iterations = better convergence, slower

# Bounds factor: how much optimizer can vary dimensions
# 2.0 means dimensions can be anywhere from value/2 to value*2
BOUNDS_FACTOR = 2.0
MIN_LENGTH = 5.0      # Minimum link length (prevents degenerate mechanisms)

# Reproducibility
RANDOM_SEED = 42      # Set to None for random results each run

# Output
OUTPUT_DIR = USER_DIR / 'demo' / 'optimization'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def main():
    print_section('OPTIMIZATION DEMO')
    print(f'Mechanism: {MECHANISM}')
    print(f'Output: {OUTPUT_DIR}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load mechanism
    # -------------------------------------------------------------------------
    print_section('Step 1: Load Mechanism')

    pylink_data, target_joint, description = load_mechanism(MECHANISM)
    dim_spec = get_dimension_spec(pylink_data, MECHANISM, BOUNDS_FACTOR, MIN_LENGTH)
    initial_dims = dict(zip(dim_spec.names, dim_spec.initial_values))

    print_mechanism_info(pylink_data, target_joint, description)
    print_dimensions(dim_spec)

    # -------------------------------------------------------------------------
    # Step 2: Create target by randomizing dimensions
    # -------------------------------------------------------------------------
    print_section('Step 2: Create Target Trajectory')

    print(f'\nRandomizing dimensions by ±{VARIATION_RANGE*100:.0f}%...')

    target_config = AchievableTargetConfig(
        dimension_variation=DimensionVariationConfig(
            default_variation_range=VARIATION_RANGE,
        ),
        random_seed=RANDOM_SEED,
    )

    target_result = create_achievable_target(
        pylink_data, target_joint, dim_spec, config=target_config,
    )

    target = target_result.target
    target_dims = target_result.target_dimensions

    print(f'Target trajectory: {target.n_steps} points')
    print('\nTarget dimensions (what optimizer should find):')
    for name in dim_spec.names:
        initial = initial_dims[name]
        target_val = target_dims[name]
        change = ((target_val - initial) / initial * 100)
        print(f'  {name}: {initial:.2f} → {target_val:.2f} ({change:+.1f}%)')

    # Compute initial trajectory for comparison
    initial_traj = compute_trajectory(pylink_data, verbose=False).trajectories[target_joint]

    # -------------------------------------------------------------------------
    # Step 3: Run optimization
    # -------------------------------------------------------------------------
    print_section('Step 3: Run PSO Optimization')

    print('\nPSO Settings:')
    print(f'  Particles: {N_PARTICLES}')
    print(f'  Iterations: {N_ITERATIONS}')
    print(f'  Bounds factor: {BOUNDS_FACTOR}')
    print('\nOptimizing...\n')

    result = run_pso_optimization(
        pylink_data=pylink_data,
        target=target,
        dimension_spec=dim_spec,
        n_particles=N_PARTICLES,
        iterations=N_ITERATIONS,
        metric='mse',
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Step 4: Analyze results
    # -------------------------------------------------------------------------
    print_section('Step 4: Results')

    print(f'\nOptimization {"succeeded" if result.success else "failed"}!')
    print(f'  Initial error: {result.initial_error:.6f}')
    print(f'  Final error:   {result.final_error:.6f}')

    if result.initial_error > 0:
        improvement = (1 - result.final_error / result.initial_error) * 100
        print(f'  Improvement:   {improvement:.1f}%')

    print('\nDimension recovery:')
    print(f'  {"Dimension":<30} {"Target":>10} {"Found":>10} {"Error":>10}')
    print(f'  {"-"*30} {"-"*10} {"-"*10} {"-"*10}')

    total_error = 0
    for name in dim_spec.names:
        target_val = target_dims[name]
        found_val = result.optimized_dimensions.get(name, initial_dims[name])
        error = abs(found_val - target_val)
        total_error += error
        print(f'  {name:<30} {target_val:>10.2f} {found_val:>10.2f} {error:>10.4f}')

    print(f'  {"-"*30} {"-"*10} {"-"*10} {"-"*10}')
    print(f'  {"TOTAL ERROR":<30} {"":>10} {"":>10} {total_error:>10.4f}')

    # -------------------------------------------------------------------------
    # Step 5: Visualize
    # -------------------------------------------------------------------------
    print_section('Step 5: Save Visualizations')

    # Get optimized trajectory
    opt_result = compute_trajectory(result.optimized_pylink_data, verbose=False, skip_sync=True)
    optimized_traj = opt_result.trajectories[target_joint] if opt_result.success else None

    # Plot trajectory comparison
    import numpy as np
    variation_plot(
        target_joint=target_joint,
        out_path=OUTPUT_DIR / f'trajectory_{TIMESTAMP}.png',
        base_trajectory=np.array(initial_traj),
        target_trajectory=np.array(target.positions),
        variation_trajectories=[np.array(optimized_traj)] if optimized_traj else None,
        title='Optimization Result',
        subtitle='Initial (black) vs Target (red) vs Optimized (blue)',
    )

    # Plot convergence
    if result.convergence_history:
        plot_convergence_comparison(
            {'PSO': result.convergence_history},
            out_path=OUTPUT_DIR / f'convergence_{TIMESTAMP}.png',
            title='Convergence History',
            layout='overlay',
        )

    # Plot dimension bounds
    plot_dimension_bounds(
        dim_spec,
        out_path=OUTPUT_DIR / f'bounds_{TIMESTAMP}.png',
        initial_values=initial_dims,
        target_values=target_dims,
        optimized_values=result.optimized_dimensions,
        title='Dimension Recovery',
    )

    # Save results JSON
    results_data = {
        'timestamp': TIMESTAMP,
        'mechanism': MECHANISM,
        'success': result.success,
        'initial_error': result.initial_error,
        'final_error': result.final_error,
        'target_dimensions': target_dims,
        'found_dimensions': result.optimized_dimensions,
    }

    with open(OUTPUT_DIR / f'results_{TIMESTAMP}.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section('COMPLETE')

    print(f'\nOutput files saved to: {OUTPUT_DIR}')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')

    # Quality assessment
    if result.final_error < 0.1:
        print('\n✓ EXCELLENT fit (error < 0.1)')
    elif result.final_error < 1.0:
        print('\n✓ GOOD fit (error < 1.0)')
    elif result.final_error < 10.0:
        print('\n⚠ MODERATE fit - try more iterations')
    else:
        print('\n✗ POOR fit - check bounds or increase iterations')


if __name__ == '__main__':
    main()
