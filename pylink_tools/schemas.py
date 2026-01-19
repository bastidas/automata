"""
schemas.py - Data structures for linkage kinematics.

Dataclasses and models used across pylink_tools modules.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrajectoryResult:
    """Result of a trajectory computation."""
    success: bool
    trajectories: dict[str, list[list[float]]]  # joint_name -> [[x,y], ...]
    n_steps: int
    joint_types: dict[str, str]
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'trajectories': self.trajectories,
            'n_steps': self.n_steps,
            'joint_types': self.joint_types,
            'error': self.error,
        }


@dataclass
class LinkageSpec:
    """Specification for building a linkage from parameters.

    Used for optimization - defines the structure and what dimensions are variable.
    """
    joints_data: list[dict]
    solve_order: list[str]
    dimension_names: tuple[str, ...]  # Names of optimizable parameters
    dimension_bounds: tuple[tuple[float, ...], tuple[float, ...]]  # (lower, upper)

    # TODO: Add mapping from dimension indices to joint properties


@dataclass
class MechanismGroup:
    """A connected group of links that may form a valid mechanism."""
    links: list[str]           # Link names in this group
    joints: list[str]          # Joint names touched by these links
    has_crank: bool
    has_ground: bool           # Has at least one Static joint
    is_valid: bool
    error: str | None = None
