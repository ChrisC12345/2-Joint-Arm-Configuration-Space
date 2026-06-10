"""Trajectory generation for robotic arms"""

from dataclasses import dataclass
from logger import Logger
import math

import numpy as np
from torus import torus_diff, torus_tuple_wrap, torus_wrap, torus_tuple_diff


@dataclass
class Trajectory:
    positions: np.ndarray  # shape (N, 2)
    velocities: np.ndarray  # shape (N, 2)
    accelerations: np.ndarray  # shape (N, 2)
    states: np.ndarray

    def __len__(self):
        return len(self.positions)

    def __add__(self, other):
        """concatenate two trajectories together,
        assuming they are contiguous in time and space"""
        return Trajectory(
            positions=np.concatenate((self.positions, other.positions)),
            velocities=np.concatenate((self.velocities, other.velocities)),
            accelerations=np.concatenate((self.accelerations, other.accelerations)),
            states=np.concatenate((self.states, other.states)),
        )


def linear_traj(path, dt=0.02):
    """follow a path with constant speed in configuration space
    generally not a good trajectory, but simple to implement
    returns two np arrays for positions and velocities at each time step"""
    positions = np.array(path)
    velocities = []
    for i in range(len(path) - 1):
        current = np.array(path[i])
        next = np.array(path[i + 1])
        velocities.append(torus_diff(next, current) / dt)
    velocities.append(np.zeros(2))

    velocities_arr = np.array(velocities)
    return Trajectory(
        positions=positions,
        velocities=velocities_arr,
        accelerations=np.zeros_like(velocities_arr),
        states=np.array(["linear"] * len(positions)),
    )


def trapezoidal_arc_traj(path, v_max, a_max, radius, v_turn, dt=0.02):
    """follow a path with trapezoidal velocity profile in configuration space
    accelerates at max_accel until max_vel, cruises at max_vel,
    then decelerates at max_accel to stop at the end of the path

    Parameters
    ---
    waypoints : list of np arrays representing points in configuration space to follow
    v_max : tuple of max velocity in rad/s for each joint
    a_max : tuple of max acceleration in rad/s^2 for each joint
    dt : time step for the trajectory"""
    traj = Trajectory(
        positions=np.empty((0, 2)),
        velocities=np.empty((0, 2)),
        accelerations=np.empty((0, 2)),
        states=np.empty(0, dtype=object),
    )
    waypoints = path.points
    steps = path.steps
    if len(waypoints) == 2:
        return trap_traj_endpts(waypoints[0], steps[0], v_max, a_max, dt=dt)
    elif len(waypoints) > 2:
        start = waypoints[0]
        start_v = 0
        prev_end_vec = np.zeros(2)
        for i in range(len(waypoints) - 2):
            p1 = waypoints[i]
            u1, u3 = steps[i], steps[i + 1]
            _, arc_end, _, vec, end_vec, *_ = calc_arc_params(p1, u1, u3, radius)

            # vector from start (prev arc end) to c1: u1 minus both arc offsets
            traj += trap_traj_endpts(
                start, u1 - vec - prev_end_vec, v_max, a_max, start_v, v_turn, dt=dt
            )
            traj += arc_traj(p1, u1, u3, v_turn, v_turn, radius, dt=dt)
            start = arc_end
            start_v = v_turn
            prev_end_vec = end_vec
        traj += trap_traj_endpts(start, u3 - end_vec, v_max, a_max, v_turn, 0, dt=dt)
        return traj


def trap_traj_endpts(p1, u1, v_max, a_max, v1=0, v2=0, dt=0.02):
    """helper function to generate a trapezoidal trajectory between two points
    returns three np arrays for position, velocity, acceleration at each time step

    Parameters
    -
    p1 : starting point in configuration space tuple of joint angles
    vec: vector from p1 to endpoint
    v_max : tuple of max velocity in rad/s for each joint
    a_max : tuple of max acceleration in rad/s^2 for each joint"""
    accelerations = []
    velocities = []
    positions = []
    states = []

    max_accel = 0
    max_vel = 0

    diff = u1

    # now consider it as 1D
    distance = np.linalg.norm(diff)

    if distance < 1e-3:
        p = torus_tuple_wrap(tuple(np.asarray(p1, dtype=float)))
        return Trajectory(
            positions=np.array([p]),
            velocities=np.array([(0.0, 0.0)]),
            accelerations=np.array([(0.0, 0.0)]),
            states=np.array(["hold"]),
        )

    slope = (diff[1]) / (diff[0]) if diff[0] != 0 else float("inf")

    # choose the more restrictive constraint between the two joints to ensure we don't
    # exceed limits on either joint
    if a_max[1] / a_max[0] > math.fabs(slope):
        ratio = 1 / math.sqrt(1 + slope**2)  # ratio between x and total components
        max_accel = a_max[0] * ratio
    else:
        ratio = 1 / math.sqrt(
            1 + (1 / slope) ** 2
        )  # ratio between y and total components
        max_accel = a_max[1] * ratio

    if v_max[1] / v_max[0] > math.fabs(slope):
        ratio = 1 / math.sqrt(1 + slope**2)  # ratio between x and total components
        max_vel = v_max[0] * ratio
    else:
        ratio = 1 / math.sqrt(
            1 + (1 / slope) ** 2
        )  # ratio between y and total components
        max_vel = v_max[1] * ratio

    # derived from vf^2-v0^2 = 2ax with v1-peak_v and v_peak-v2
    peak_vel = min(max_vel, math.sqrt((2 * max_accel * distance + v1**2 + v2**2) / 2))
    accel_steps = int((peak_vel - v1) / max_accel / dt)
    decel_steps = int((peak_vel - v2) / max_accel / dt)
    cruise_steps = int(
        (
            distance
            - accel_steps * dt * (v1 + peak_vel) / 2
            - decel_steps * dt * (v2 + peak_vel) / 2
        )
        / peak_vel
        / dt
    )

    x, v = 0, v1

    for i in range(accel_steps):
        v = v1 + max_accel * i * dt
        x = v1 * i * dt + 0.5 * max_accel * (i * dt) ** 2
        accelerations.append(decompose_scalar_vec(u1, max_accel))
        velocities.append(decompose_scalar_vec(u1, v))
        positions.append(torus_tuple_wrap(decompose_scalar_vec(u1, x, p1)))
        states.append("accel")

    v = peak_vel
    accel_dist = x + v * dt

    for i in range(cruise_steps):
        x = accel_dist + v * i * dt
        accelerations.append(decompose_scalar_vec(u1, 0))
        velocities.append(decompose_scalar_vec(u1, v))
        positions.append(torus_tuple_wrap(decompose_scalar_vec(u1, x, p1)))
        states.append("cruise")

    dist_after_cruise = x + v * dt

    for i in range(decel_steps):
        v = peak_vel - max_accel * i * dt
        x = dist_after_cruise + peak_vel * i * dt - 0.5 * max_accel * (i * dt) ** 2
        accelerations.append(decompose_scalar_vec(u1, -max_accel))
        velocities.append(decompose_scalar_vec(u1, v))
        positions.append(torus_tuple_wrap(decompose_scalar_vec(u1, x, p1)))
        states.append("decel")

    return Trajectory(
        positions=np.array(positions),
        velocities=np.array(velocities),
        accelerations=np.array(accelerations),
        states=np.array(states),
    )


def calc_arc_params(p1, u1, u3, r):
    """helper function to calculate the center and start/end points of a circular arc
    going from p1 to p3 with p2 as an intermediate point

    Parameters
    ---
    p1 : starting point in configuration space np array of joint angles
    u1 : vector from p1 to p2
    u3 : vector from p2 to p3
    r : desired radius of the circular arc

    Returns
    ---
    start, end, center, angle subtended at center by the arc"""
    p2 = np.array(p1) + u1
    angle = math.acos(np.dot(-u1, u3) / (np.linalg.norm(-u1) * np.linalg.norm(u3))) / 2

    safe_r = min(
        r, np.linalg.norm(u1) * math.tan(angle), np.linalg.norm(u3) * math.tan(angle)
    )

    vec = safe_r / math.tan(angle) * (u1 / np.linalg.norm(u1))
    end_vec = safe_r / math.tan(angle) * (u3 / np.linalg.norm(u3))
    start = p2 - vec  # arc start sits between p1 and p2
    end = p2 + end_vec
    bisector = -u1 / np.linalg.norm(u1) + u3 / np.linalg.norm(
        u3
    )  # both point away from p2
    center = p2 + safe_r / math.sin(angle) * bisector / np.linalg.norm(bisector)

    return start, end, center, vec, end_vec, safe_r, math.pi - 2 * angle


def arc_traj(p1, u1, u3, v1, v3, r, dt=0.02):
    """generate a circular arc trajectory going from p1 to p3 with p2 as a waypoint
    with constant acceleration along the arc

    Parameters
    ---
    p1: direction that trajectory is coming from at beginning
    u1: vector from p1 to p2
    u3: vector from p2 to p3
    r: radius of the circular arc
    v1: starting linear velocity along the arc
    v3: ending linear velocity along the arc
    """
    start, end, center, vec, _end_vec, r, angle = calc_arc_params(p1, u1, u3, r)
    arc_length = r * angle
    accel = (v3**2 - v1**2) / (2 * arc_length)
    num_steps = int(arc_length / ((v1 + v3) / 2) / dt)

    # determine CW (-1) or CCW (+1) from cross product of incoming  and centripetal
    incoming = u1
    turn_sign = np.sign(
        incoming[0] * (center[1] - start[1]) - incoming[1] * (center[0] - start[0])
    )

    if num_steps <= 0:
        return Trajectory(
            positions=np.empty((0, 2)),
            velocities=np.empty((0, 2)),
            accelerations=np.empty((0, 2)),
            states=np.empty(0, dtype=object),
        )

    start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    arc_swept = 0.0
    v = v1

    positions = []
    velocities = []
    accelerations = []
    states = []

    for i in range(num_steps):
        t = i * dt
        arc_swept = v1 * t + 0.5 * accel * t**2
        v = v1 + accel * t

        theta = start_angle + turn_sign * arc_swept / r
        x = center + r * np.array([math.cos(theta), math.sin(theta)])

        radial = (x - center) / r
        tangent_vec = turn_sign * np.array([-radial[1], radial[0]])

        a_c = v**2 / r
        a_total = -a_c * radial + accel * tangent_vec

        positions.append(torus_tuple_wrap(tuple(x)))
        velocities.append(tuple(tangent_vec * v))
        accelerations.append(tuple(a_total))
        states.append("arc")

    return Trajectory(
        positions=np.array(positions),
        velocities=np.array(velocities),
        accelerations=np.array(accelerations),
        states=np.array(states),
    )


def decompose_scalar_vec(vec, scalar, start=(0.0, 0.0)):
    """decomposes a scalar value into x and y components from start along vec

    Parameters
    ----
    start : starting point in configuration space np array of joint angles
    vec : vector from start to end point in configuration space np array of joint angles
    scalar: scalar value to decompose
    returns a tuple of (x_component, y_component)"""
    distance = np.linalg.norm(vec)
    x_comp = start[0] + scalar * vec[0] / distance
    y_comp = start[1] + scalar * vec[1] / distance
    return (x_comp, y_comp)
