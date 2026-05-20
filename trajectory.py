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


def trapezoidal_arc_traj(waypoints, v_max, a_max, radius, v_turn, dt=0.02):
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
    if len(waypoints) == 2:
        return trap_traj_endpts(waypoints[0], waypoints[1], v_max, a_max, dt=dt)
    elif len(waypoints) > 2:
        start = waypoints[0]  # where to start from every loop iteration
        start_v = 0
        for i in range(1, len(waypoints) - 1):
            p1, p2, p3 = waypoints[i - 1], waypoints[i], waypoints[i + 1]
            c1, c2 = calc_circular_params(p1, p2, p3, radius)[:2]
            traj += trap_traj_endpts(start, c1, v_max, a_max, start_v, v_turn, dt=dt)
            traj += arc_traj(c1, p2, c2, v_turn, v_turn, radius, dt=dt)
            start = c2
            start_v = v_turn
        traj += trap_traj_endpts(start, waypoints[-1], v_max, a_max, v_turn, 0, dt=dt)
        return traj


def trap_traj_endpts(p1, p2, v_max, a_max, v1=0, v2=0, dt=0.02):
    """helper function to generate a trapezoidal trajectory between two points
    returns three np arrays for position, velocity, acceleration at each time step

    Parameters
    -
    p1 : starting point in configuration space tuple of joint angles
    p2 : ending point in configuration space tuple of joint angles
    v_max : tuple of max velocity in rad/s for each joint
    a_max : tuple of max acceleration in rad/s^2 for each joint"""
    accelerations = []
    velocities = []
    positions = []
    states = []

    max_accel = 0
    max_vel = 0

    diff = torus_tuple_diff(p2, p1)

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

    slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else math.inf

    # choose the more restrictive constraint between the two joints to ensure we don't
    # exceed limits on either joint
    if a_max[1] / a_max[0] > slope:
        ratio = 1 / math.sqrt(1 + slope**2)  # ratio between x and total components
        max_accel = a_max[0] * ratio
        max_vel = v_max[0] * ratio
    else:
        ratio = 1 / math.sqrt(
            1 + (1 / slope) ** 2
        )  # ratio between y and total components
        max_accel = a_max[1] * ratio
        max_vel = v_max[1] * ratio

    # derived from vf^2-v0^2 = 2ax with v1-peak_v and v_peak-v2
    peak_vel = min(max_vel, math.sqrt(2 * max_accel * distance + v1**2 + v2**2))
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
        accelerations.append(decompose_scalar(p1, p2, max_accel))
        velocities.append(decompose_scalar(p1, p2, v))
        positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, x)))
        v = v1 + max_accel * i * dt
        x = v1 * i * dt + 0.5 * max_accel * (i * dt) ** 2
        states.append("accel")

    v = peak_vel
    accel_dist = x

    for i in range(cruise_steps):
        accelerations.append(decompose_scalar(p1, p2, 0))
        velocities.append(decompose_scalar(p1, p2, v))
        positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, x)))
        x = accel_dist + v * i * dt
        states.append("cruise")

    dist_after_cruise = x

    for i in range(decel_steps):
        accelerations.append(decompose_scalar(p1, p2, -max_accel))
        velocities.append(decompose_scalar(p1, p2, v))
        positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, x)))
        v = peak_vel - max_accel * i * dt
        x = dist_after_cruise + peak_vel * i * dt - 0.5 * max_accel * (i * dt) ** 2
        states.append("decel")

    return Trajectory(
        positions=np.array(positions),
        velocities=np.array(velocities),
        accelerations=np.array(accelerations),
        states=np.array(states),
    )


def calc_circular_params(p1, p2, p3, r):
    """helper function to calculate the center and start/end points of a circular arc
    going from p1 to p3 with p2 as an intermediate point

    Returns
    ---
    start, end, center, angle subtended at center by the arc"""
    p2 = np.array(p2)
    u1 = np.array(torus_tuple_diff(p1, p2))
    u3 = np.array(torus_tuple_diff(p3, p2))
    angle = math.acos(np.dot(u1, u3) / (np.linalg.norm(u1) * np.linalg.norm(u3))) / 2

    safe_r = min(
        r, np.linalg.norm(u1) * math.tan(angle), np.linalg.norm(u3) * math.tan(angle)
    )

    start = p2 + safe_r / math.tan(angle) * (u1 / np.linalg.norm(u1))
    end = p2 + safe_r / math.tan(angle) * (u3 / np.linalg.norm(u3))
    bisector = u1 / np.linalg.norm(u1) + u3 / np.linalg.norm(u3)
    center = p2 + safe_r / math.sin(angle) * bisector / np.linalg.norm(bisector)

    return start, end, center, safe_r, math.pi - 2 * angle


def arc_traj(p1, p2, p3, v1, v3, r, dt=0.02):
    """generate a circular arc trajectory going from p1 to p3 with p2 as a waypoint
    with constant acceleration along the arc

    Parameters
    ---
    p1: direction that trajectory is coming from at beginning
    p2: point trajectory heads in at beginning
    p3: ending point direction
    r: radius of the circular arc
    v1: starting linear velocity along the arc
    v3: ending linear velocity along the arc
    """
    start, end, center, r, angle = calc_circular_params(p1, p2, p3, r)
    arc_length = r * angle
    accel = (v3**2 - v1**2) / (2 * arc_length)
    num_steps = int(arc_length / ((v1 + v3) / 2) / dt)

    # determine CW (-1) or CCW (+1) from cross product of incoming direction × centripetal
    incoming = np.array(p2) - np.array(p1)
    turn_sign = np.sign(
        incoming[0] * (center[1] - start[1]) - incoming[1] * (center[0] - start[0])
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


def decompose_scalar(p1, p2, scalar):
    """decomposes a scalar value into x and y components along p1-p2

    Parameters
    ----
    p1 : starting point in configuration space np array of joint angles
    p2 : ending point in configuration space np array of joint angles
    scalar: scalar value to decompose
    returns a tuple of (x_component, y_component)"""
    diff = torus_tuple_diff(p2, p1)
    distance = np.linalg.norm(diff)
    x_comp = p1[0] + scalar * diff[0] / distance
    y_comp = p1[1] + scalar * diff[1] / distance
    return (x_comp, y_comp)


# def trap_traj_endpts(p1, p2, max_v, max_a, dt=0.02):
#     """helper function to generate a trapezoidal trajectory between two points
#     returns three np arrays for position, velocity, acceleration at each time step

#     Parameters
#     -
#     p1 : starting point in configuration space tuple of joint angles
#     p2 : ending point in configuration space tuple of joint angles
#     max_vel : tuple of max velocity in rad/s for each joint
#     max_accel : tuple of max acceleration in rad/s^2 for each joint"""
#     accelerations = []
#     velocities = []
#     positions = []

#     max_accel = 0
#     max_vel = 0

#     slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p2[0] else math.inf

#     # choose the more restrictive constraint between the two joints to ensure we don't
#     # exceed limits on either joint
#     if max_a[1] / max_a[0] > slope:
#         ratio = 1 / math.sqrt(1 + slope**2)  # ratio between x and total components
#         max_accel = max_a[0] * ratio
#         max_vel = max_v[0] * ratio
#     else:
#         ratio = 1 / math.sqrt(
#             1 + (1 / slope) ** 2
#         )  # ratio between y and total components
#         max_accel = max_a[1] * ratio
#         max_vel = max_v[1] * ratio

#     diff = torus_tuple_diff(p2, p1)

#     # now consider it as 1D
#     distance = np.linalg.norm(diff)
#     accel_time = max_vel / max_accel
#     accel_distance = 0.5 * max_accel * accel_time**2

#     if distance < 2 * accel_distance:
#         print("triangle profile")
#         # triangle profile
#         num_steps = math.ceil(2 * math.sqrt(distance / max_accel) / dt)
#         half_num_steps = num_steps // 2
#         for i in range(half_num_steps):
#             accelerations.append(decompose_scalar(p1, p2, max_accel))
#             velocity = max_accel * i * dt
#             velocities.append(decompose_scalar(p1, p2, velocity))
#             position = 0.5 * max_accel * (i * dt) ** 2
#             raw = decompose_scalar(p1, p2, position)
#             positions.append((torus_wrap(raw[0]), torus_wrap(raw[1])))
#         for i in range(half_num_steps):
#             accelerations.append(decompose_scalar(p1, p2, -max_accel))
#             velocity = max_accel * (half_num_steps - i) * dt
#             velocities.append(decompose_scalar(p1, p2, velocity))
#             position = distance - 0.5 * max_accel * ((half_num_steps - i) * dt) ** 2
#             raw = decompose_scalar(p1, p2, position)
#             positions.append((torus_wrap(raw[0]), torus_wrap(raw[1])))
#     else:
#         print("trapezoid profile")
#         # trapezoid profile
#         accel_steps = int(accel_time / dt)
#         cruise_distance = distance - 2 * accel_distance
#         cruise_steps = int((cruise_distance / max_vel) / dt)
#         # acceleration steps
#         for i in range(accel_steps):
#             accelerations.append(decompose_scalar(p1, p2, max_accel))
#             velocity = max_accel * i * dt
#             velocities.append(decompose_scalar(p1, p2, velocity))
#             position = 0.5 * max_accel * (i * dt) ** 2
#             positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, position)))
#         # cruise steps
#         for i in range(cruise_steps):
#             accelerations.append(decompose_scalar(p1, p2, 0))
#             velocity = max_vel
#             velocities.append(decompose_scalar(p1, p2, velocity))
#             position = accel_distance + max_vel * i * dt
#             positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, position)))
#         # deceleration steps
#         for i in range(accel_steps):
#             accelerations.append(decompose_scalar(p1, p2, -max_accel))
#             velocity = max_vel - max_accel * i * dt
#             velocities.append(decompose_scalar(p1, p2, velocity))
#             position = distance - 0.5 * max_accel * ((accel_steps - i) * dt) ** 2
#             positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, position)))

#     return (positions, velocities, accelerations)
