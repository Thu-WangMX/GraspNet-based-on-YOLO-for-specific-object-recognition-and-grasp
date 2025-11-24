"""Trajectory planning framework for Flexiv robots.

This module provides a lightweight scaffold that exposes two planning
approaches for Cartesian 6D pose goals:

* ``SCurvePlanner`` – analytic S-curve interpolation (jerk-limited
  polynomial profile) for quick prototyping.
* ``SQPTrajectoryPlanner`` – numerical optimisation based on SLSQP that
  can enforce smoothness costs and simple boundary constraints.

The planners return trajectories as lists of :class:`Pose` instances
containing position (XYZ) and orientation (roll, pitch, yaw in degrees).

The implementation intentionally focuses on clarity and extensibility:
obstacle avoidance is out of scope, but interfaces are ready for future
constraints (joint limits, velocity bounds, etc.).
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def _smooth_step(t: np.ndarray) -> np.ndarray:
    """Quintic smooth step (S-curve) profile.

    Produces a trajectory with zero velocity/acceleration at the
    boundaries: ``10 t^3 - 15 t^4 + 6 t^5``.
    """

    return 10 * t**3 - 15 * t**4 + 6 * t**5

def _pose_to_state(pose: "Pose") -> np.ndarray:
    """Convert Pose to 6D state [x, y, z, rx, ry, rz] where r* is a rotation vector."""
    rotvec = pose.to_rotation().as_rotvec()
    return np.concatenate([pose.position, rotvec])

def _state_to_pose(state: np.ndarray) -> "Pose":
    """Convert 6D state to Pose (simple version without continuity)."""
    position = state[:3]
    r = R.from_rotvec(state[3:])
    rpy_deg = r.as_euler("xyz", degrees=True)
    return Pose(position, rpy_deg)

def _state_to_pose_continuous(state: np.ndarray, ref_rpy: np.ndarray) -> "Pose":
    """Convert 6D state to Pose with Euler-branch continuity (preferred in trajectories)."""
    position = state[:3]
    r = R.from_rotvec(state[3:])
    rpy_deg = _as_euler_xyz_continuous(r, ref_rpy)
    return Pose(position, rpy_deg)

def _wrap180(deg: np.ndarray) -> np.ndarray:
    """Wrap angles (degrees) into the range (-180, 180].

    This ensures a canonical representation of any angle by applying a modulo
    operation. For exact -180°, it normalizes to +180° for consistency.

    Args:
        deg: Array of angles in degrees.

    Returns:
        Array of wrapped angles in degrees within (-180, 180].
    """
    out = ((deg + 180.0) % 360.0) - 180.0
    # Normalize -180 to +180 to avoid duplicated representations
    out[np.isclose(out, -180.0, atol=1e-9)] = 180.0
    return out

def _unwrap_to_ref(curr_deg: np.ndarray, ref_deg: np.ndarray) -> np.ndarray:
    """Unwrap angles (degrees) to be as close as possible to a reference.

    This adjusts each axis independently by adding/subtracting integer multiples
    of 360°, minimizing the difference to the reference angles.

    Args:
        curr_deg: Current angles in degrees.
        ref_deg: Reference angles in degrees.

    Returns:
        Unwrapped angles (degrees) close to the reference branch.
    """
    out = curr_deg.astype(float).copy()
    diff = out - ref_deg
    out -= np.round(diff / 360.0) * 360.0
    return out

def _euler_xyz_complement(rpy_deg: np.ndarray) -> np.ndarray:
    """Return the coupled equivalent Euler solution for XYZ convention.

    For Euler XYZ, the following equivalence holds:
        (α, β, γ) ↔ (α±180°, 180°−β, γ±180°)
    This function returns the minus-180° variant (any ± variant is equivalent),
    then wraps to (-180, 180].

    Args:
        rpy_deg: Euler angles (roll, pitch, yaw) in degrees, XYZ order.

    Returns:
        Complementary coupled Euler solution in degrees, wrapped to (-180, 180].
    """
    a, b, c = rpy_deg
    comp = np.array([a - 180.0, 180.0 - b, c - 180.0], dtype=float)
    return _wrap180(comp)

def _as_euler_xyz_continuous(rot: R, ref_deg: np.ndarray) -> np.ndarray:
    """Map a Rotation to Euler XYZ (deg) while preserving branch continuity.

    The function evaluates both the "raw" Euler solution and its coupled
    equivalent branch, unwraps each to be close to a given reference, and then
    selects the candidate with the smaller Euclidean distance to the reference.

    Args:
        rot: A scipy Rotation instance.
        ref_deg: Reference Euler angles (deg) to keep continuity.

    Returns:
        Euler angles (deg) in XYZ convention, continuous w.r.t. the reference.
    """
    rpy1 = rot.as_euler("xyz", degrees=True)   # raw solution
    rpy2 = _euler_xyz_complement(rpy1)         # coupled equivalent

    # Unwrap each candidate towards the reference branch
    ref = _wrap180(ref_deg)
    cand1 = _unwrap_to_ref(_wrap180(rpy1), ref)
    cand2 = _unwrap_to_ref(_wrap180(rpy2), ref)

    # Choose the branch closer to the reference
    # return cand2 if np.linalg.norm(cand2 - ref_deg) < np.linalg.norm(cand1 - ref_deg) else cand1
    return cand2 if np.linalg.norm(cand2 - ref) < np.linalg.norm(cand1 - ref) else cand1


@dataclasses.dataclass
class Pose:
    """Simple Cartesian pose representation using XYZ + RPY (deg)."""

    position: np.ndarray
    rpy_deg: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.rpy_deg = np.asarray(self.rpy_deg, dtype=np.float64).reshape(3)

    def as_vector(self) -> np.ndarray:
        """Return the pose as ``[x, y, z, rx, ry, rz]`` vector (degrees)."""

        return np.concatenate([self.position, self.rpy_deg])

    def to_rotation(self) -> R:
        """Return orientation as :class:`~scipy.spatial.transform.Rotation`."""

        return R.from_euler("xyz", np.deg2rad(self.rpy_deg))

    def quaternion_xyzw(self) -> np.ndarray:
        x, y, z, w = self.to_rotation().as_quat()
        return np.array([x, y, z, w], dtype=np.float64)

    def quaternion_wxyz(self) -> np.ndarray:
        x, y, z, w = self.to_rotation().as_quat()
        return np.array([w, x, y, z], dtype=np.float64)

    @classmethod
    def from_xyz_quat(
        cls, position: Sequence[float], orientation_wxyz: Sequence[float]
    ) -> "Pose":
        w, x, y, z = np.asarray(orientation_wxyz, dtype=np.float64)
        r = R.from_quat([x, y, z, w])
        rpy_deg = np.rad2deg(r.as_euler("xyz"))
        return cls(np.array(position, dtype=np.float64), rpy_deg)

    @classmethod
    def from_xyz_rpy(
        cls, position: Sequence[float], rpy_deg: Sequence[float]
    ) -> "Pose":
        return cls(np.array(position, dtype=np.float64), np.array(rpy_deg, dtype=np.float64))

    def copy(self) -> "Pose":
        return Pose(np.copy(self.position), np.copy(self.rpy_deg))

@dataclasses.dataclass
class Trajectory:
    """Trajectory samples containing poses and associated velocities."""

    poses: List[Pose]
    linear_velocities: List[np.ndarray]
    angular_velocities: List[np.ndarray]

    def __post_init__(self) -> None:
        count = len(self.poses)
        if len(self.linear_velocities) != count or len(self.angular_velocities) != count:
            raise ValueError("Velocity lists must match pose count")
        self.linear_velocities = [np.asarray(v, dtype=np.float64).reshape(3) for v in self.linear_velocities]
        self.angular_velocities = [np.asarray(v, dtype=np.float64).reshape(3) for v in self.angular_velocities]

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx]

    def __iter__(self):
        return iter(self.poses)

class BaseTrajectoryPlanner(ABC):
    """Abstract base class for Cartesian trajectory planners."""

    def __init__(self, num_samples: int = 50, duration: float = 5.0) -> None:
        self.num_samples = num_samples
        self.duration = duration

    @abstractmethod
    def plan(self, start: Pose, goal: Pose) -> Trajectory:
        """Compute a sequence of poses from ``start`` to ``goal``."""

class SCurvePlanner(BaseTrajectoryPlanner):
    """Analytic S-curve planner for quick prototyping."""

    def plan(self, start: Pose, goal: Pose) -> Trajectory:
        t = np.linspace(0.0, 1.0, self.num_samples)
        s = _smooth_step(t)
        s_dot = 30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4
        duration = max(self.duration, 1e-9)
        alpha_dot = s_dot / duration

        # 位置插值
        p0 = start.position.astype(float); p1 = goal.position.astype(float)
        dp = p1 - p0

        # 姿态 geodesic：R = R0 * exp(α log(R0^T R1))
        R0 = start.to_rotation()
        R1 = goal.to_rotation()
        R_rel = R1 * R0.inv()
        rotvec_rel = R_rel.as_rotvec()

        poses: List[Pose] = []
        linear_velocities: List[np.ndarray] = []
        angular_velocities: List[np.ndarray] = []
        prev_rpy = start.rpy_deg.astype(float)

        for idx, alpha in enumerate(s):
            p = p0 + alpha * dp
            r = R0 * R.from_rotvec(alpha * rotvec_rel)
            state = np.concatenate([p, r.as_rotvec()])
            pose = _state_to_pose_continuous(state, prev_rpy)
            poses.append(pose)
            prev_rpy = pose.rpy_deg
            linear_velocities.append(alpha_dot[idx] * dp)
            angular_velocities.append(alpha_dot[idx] * rotvec_rel)

        return Trajectory(poses, linear_velocities, angular_velocities)

class SQPTrajectoryPlanner(BaseTrajectoryPlanner):
    """SQP planner (SLSQP) minimizing a jerk-like objective on SE(3).

    The state is a sequence of 6D vectors [x, y, z, rx, ry, rz], where the last
    three entries form a rotation vector. The objective penalizes both position
    and rotational velocities/accelerations using group-aware differences:

      - Position velocity:    (p_{k+1} - p_k) / dt
      - Rotational velocity:  log( R_k^T R_{k+1} ) / dt
      - Second differences are used for accelerations.

    A single L2 velocity constraint can be optionally enforced by combining
    position and rotational velocities with a tunable scale factor.
    """

    def __init__(
        self,
        num_samples: int = 50,
        duration: float = 5.0,
        max_velocity: Optional[float] = None,
        weight_velocity: float = 1e-2,
        weight_acc: float = 1.0,
        rot_scale_v: float = 1.0,
        rot_scale_a: float = 1.0,
    ) -> None:
        super().__init__(num_samples=num_samples, duration=duration)
        self.max_velocity = max_velocity
        self.weight_velocity = weight_velocity
        self.weight_acc = weight_acc
        # Scale factors to balance position (m, m/s, m/s^2) and rotation (rad, rad/s, rad/s^2)
        self.rot_scale_v = rot_scale_v
        self.rot_scale_a = rot_scale_a

    # ---------- small internal helpers ----------

    def _initial_guess(self, start: "Pose", goal: "Pose") -> np.ndarray:
        """Build a good initial guess via SCurvePlanner states."""
        baseline = SCurvePlanner(num_samples=self.num_samples, duration=self.duration).plan(start, goal)
        return np.stack([_pose_to_state(p) for p in baseline])

    @staticmethod
    def _states_to_PR(states: np.ndarray) -> Tuple[np.ndarray, R]:
        """Split batch states into positions (N,3) and rotations (Rotation over N)."""
        P = states[:, :3]
        # SciPy supports batch conversion from rotvec array (N,3)
        Rm = R.from_rotvec(states[:, 3:])
        return P, Rm

    def _vel_acc_cost(self, states: np.ndarray, dt: float) -> float:
        """Compute the jerk-like objective composed of velocity and acceleration terms."""
        P, Rm = self._states_to_PR(states)

        # Position velocity and acceleration (finite differences)
        vP = np.diff(P, axis=0) / dt                      # shape: (N-1, 3)
        aP = np.diff(vP, axis=0) / dt                     # shape: (N-2, 3)

        # Rotational velocity on the group: vr_k = log(R_k^T R_{k+1}) / dt
        R_rel = Rm[:-1].inv() * Rm[1:]                    # Rotation batch of length (N-1)
        vr = R_rel.as_rotvec() / dt                       # shape: (N-1, 3)
        ar = np.diff(vr, axis=0) / dt                     # shape: (N-2, 3)

        # Weighted quadratic costs; rot terms scaled to commensurate with position
        cost_vel = np.sum(vP**2) + (self.rot_scale_v**2) * np.sum(vr**2)
        cost_acc = np.sum(aP**2) + (self.rot_scale_a**2) * np.sum(ar**2)
        return self.weight_velocity * cost_vel + self.weight_acc * cost_acc

    def _velocity_constraint(self, states: np.ndarray, dt: float) -> np.ndarray:
        """Optional per-segment velocity constraint combining pos and rot velocities.

        Returns positive values when within the limit (SLSQP inequality format).
        """
        P, Rm = self._states_to_PR(states)
        vP = np.diff(P, axis=0) / dt                      # (N-1, 3)
        vr = (Rm[:-1].inv() * Rm[1:]).as_rotvec() / dt    # (N-1, 3)

        # Combine pos and rot velocities into one norm with scaling
        v_combined = np.linalg.norm(np.hstack([vP, self.rot_scale_v * vr]), axis=1)
        return self.max_velocity - v_combined

    # ---------- main API ----------

    def plan(self, start: "Pose", goal: "Pose") -> Trajectory:
        n = self.num_samples
        if n < 3:
            raise ValueError("num_samples must be >= 3 for SQP planner")

        # Boundary states
        start_state = _pose_to_state(start)
        goal_state = _pose_to_state(goal)

        # Initial guess as baseline (good for convergence)
        baseline_states = self._initial_guess(start, goal)

        # Optimize only interior states: indices [1 .. n-2]
        x0 = baseline_states[1:-1].ravel()
        dt = self.duration / (n - 1)

        def unpack(x: np.ndarray) -> np.ndarray:
            """Rebuild the full (n,6) states with fixed boundary conditions."""
            states = baseline_states.copy()
            states[1:-1] = x.reshape(-1, 6)
            states[0] = start_state
            states[-1] = goal_state
            return states

        def objective(x: np.ndarray) -> float:
            states = unpack(x)
            return self._vel_acc_cost(states, dt)
        
        def euler_continuous_from_rotations(Rm: R, order='xyz', degrees=True):
            q = Rm.as_quat()  # [x,y,z,w]
            for i in range(1, len(q)):
                if np.dot(q[i], q[i-1]) < 0.0:
                    q[i] = -q[i]
            R_cont = R.from_quat(q)
            eul = R_cont.as_euler(order, degrees=degrees)
            # 仅用于显示时再 wrap，避免中途影响连续性
            eul = ((eul + 180.0) % 360.0) - 180.0
            return eul

        constraints = []
        if self.max_velocity is not None:
            def vel_cons(x: np.ndarray) -> np.ndarray:
                states = unpack(x)
                return self._velocity_constraint(states, dt)
            constraints.append({"type": "ineq", "fun": vel_cons})

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"SQP planning failed: {result.message}")

        optimal_states = unpack(result.x)

        # Convert to poses with continuous Euler branch selection
        poses: List[Pose] = []
        prev_rpy = _wrap180(start.rpy_deg.astype(float))
        for state in optimal_states:
            pose = _state_to_pose_continuous(state, prev_rpy)
            poses.append(pose)
            prev_rpy = pose.rpy_deg

        # Velocity interpolation (central differences for interior points)
        P, Rm = self._states_to_PR(optimal_states)
        ################
        eul_deg = euler_continuous_from_rotations(Rm, order='xyz', degrees=True)
        poses = [Pose(position=P[i], rpy_deg=eul_deg[i]) for i in range(len(P))]
        ################
        linear_velocities: List[np.ndarray] = [np.zeros(3, dtype=np.float64) for _ in range(n)]
        angular_velocities: List[np.ndarray] = [np.zeros(3, dtype=np.float64) for _ in range(n)]

        if n >= 2:
            linear_velocities[0] = (P[1] - P[0]) / dt
            angular_velocities[0] = (Rm[0].inv() * Rm[1]).as_rotvec() / dt
            linear_velocities[-1] = (P[-1] - P[-2]) / dt
            angular_velocities[-1] = (Rm[-2].inv() * Rm[-1]).as_rotvec() / dt

        for i in range(1, n - 1):
            linear_velocities[i] = (P[i + 1] - P[i - 1]) / (2.0 * dt)
            omega_fwd = (Rm[i].inv() * Rm[i + 1]).as_rotvec() / dt
            omega_back = (Rm[i - 1].inv() * Rm[i]).as_rotvec() / dt
            angular_velocities[i] = 0.5 * (omega_fwd + omega_back)

        return Trajectory(poses, linear_velocities, angular_velocities)

class TrajectoryPlanner():
    def __init__(self):
        pass

    def plan_with_s_curve(
        self,
        start_pose: Pose,
        goal_pose: Pose,
        duration: float = 5.0,
        num_samples: int = 50,
    ) -> Trajectory:
        planner = SCurvePlanner(num_samples=num_samples, duration=duration)
        return planner.plan(start_pose, goal_pose)

    def plan_with_sqp(
        self,
        start_pose: Pose,
        goal_pose: Pose,
        duration: float = 5.0,
        num_samples: int = 50,
        max_velocity: Optional[float] = None,
    ) -> Trajectory:
        planner = SQPTrajectoryPlanner(
            num_samples=num_samples,
            duration=duration,
            max_velocity=max_velocity,
        )
        return planner.plan(start_pose, goal_pose)

    def execute_cartesian_trajectory(
            self,
            robot=None,
            start_pose: Pose = None,
            goal_pose: Pose = None,
            planner_name: str = "s_curve",
            duration: float = 5.0,
            num_samples: int = 10,
            speed: float = 0.4,
            acc: float = 0.1,
            zoneRadius: str = "Z100",
    ) -> None:
        if planner_name == "s_curve":
            trajectory = self.plan_with_s_curve(start_pose, goal_pose, duration, num_samples)
        elif planner_name == "sqp":
            trajectory = self.plan_with_sqp(start_pose, goal_pose, duration, num_samples)
        else:
            raise ValueError(f"Unknown planner name: {planner_name}")

        pose_list = trajectory.poses
        if not pose_list:
            print("Received empty trajectory; nothing to execute.")
            return

        if robot is None:
            print("No robot provided; skipping execution.")
            return

        point_list = []
        linear_velocity_list = []
        angular_velocity_list = []
        for pose, vel_lin, vel_ang in zip(
            trajectory.poses,
            trajectory.linear_velocities,
            trajectory.angular_velocities,
        ):
            print("pose:", pose)
            point_list.append(pose.position.tolist() + pose.rpy_deg.tolist())
            linear_velocity_list.append(vel_lin.tolist())
            angular_velocity_list.append(vel_ang.tolist())

        if hasattr(robot, "MoveL_multi_points_with_velocity"):
            robot.MoveL_multi_points_with_velocity(point_list, linear_velocity_list, speed, acc, zoneRadius)
        else:
            robot.MoveL_multi_points(point_list, speed, acc, zoneRadius)

__all__ = [
    "Pose",
    "Trajectory",
    "TrajectoryPlanner",
    "SCurvePlanner",
    "SQPTrajectoryPlanner",
]
