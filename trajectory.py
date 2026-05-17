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

    def __len__(self):
        return len(self.positions)


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
    )


def trapezoidal_traj(waypoints, max_vel, max_accel, dt=0.02):
    """follow a path with trapezoidal velocity profile in configuration space
    accelerates at max_accel until max_vel, cruises at max_vel, 
    then decelerates at max_accel to stop at the end of the path

    Parameters
    ---
    waypoints : list of np arrays representing points in configuration space to follow
    max_vel : tuple of max velocity in rad/s for each joint
    max_accel : tuple of max acceleration in rad/s^2 for each joint
    dt : time step for the trajectory"""
    positions = []
    velocities = []
    accelerations = []

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        pos, vel, accel = trap_traj_endpts(p1, p2, max_vel, max_accel, dt)
        positions.extend(pos)
        velocities.extend(vel)
        accelerations.extend(accel)

    return Trajectory(
        positions=np.array(positions),
        velocities=np.array(velocities),
        accelerations=np.array(accelerations),
    )


def trap_traj_endpts(p1, p2, max_velocities, max_accelerations, dt=0.02):
    """helper function to generate a trapezoidal trajectory between two points
    returns three np arrays for position, velocity, acceleration at each time step

    Parameters
    -
    p1 : starting point in configuration space tuple of joint angles
    p2 : ending point in configuration space tuple of joint angles
    max_vel : tuple of max velocity in rad/s for each joint
    max_accel : tuple of max acceleration in rad/s^2 for each joint"""
    accelerations = []
    velocities = []
    positions = []

    max_accel = 0
    max_vel = 0

    slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[1] != p2[0] else math.inf

    # choose the more restrictive constraint between the two joints to ensure we don't 
    # exceed limits on either joint
    if max_accelerations[1] / max_accelerations[0] > slope:
        ratio = 1 / math.sqrt(1 + slope**2)  # ratio between x and total components
        max_accel = max_accelerations[0] * ratio
        max_vel = max_velocities[0] * ratio
    else:
        ratio = 1 / math.sqrt(
            1 + (1 / slope) ** 2
        )  # ratio between y and total components
        max_accel = max_accelerations[1] * ratio
        max_vel = max_velocities[1] * ratio

    diff = torus_tuple_diff(p2, p1)

    # now consider it as 1D
    distance = np.linalg.norm(diff)
    accel_time = max_vel / max_accel
    accel_distance = 0.5 * max_accel * accel_time**2

    if distance < 2 * accel_distance:
        print("triangle profile")
        # triangle profile
        num_steps = math.ceil(2 * math.sqrt(distance / max_accel) / dt)
        half_num_steps = num_steps // 2
        for i in range(half_num_steps):
            accelerations.append(decompose_scalar(p1, p2, max_accel))
            velocity = max_accel * i * dt
            velocities.append(decompose_scalar(p1, p2, velocity))
            position = 0.5 * max_accel * (i * dt) ** 2
            raw = decompose_scalar(p1, p2, position)
            positions.append((torus_wrap(raw[0]), torus_wrap(raw[1])))
        for i in range(half_num_steps):
            accelerations.append(decompose_scalar(p1, p2, -max_accel))
            velocity = max_accel * (half_num_steps - i) * dt
            velocities.append(decompose_scalar(p1, p2, velocity))
            position = distance - 0.5 * max_accel * ((half_num_steps - i) * dt) ** 2
            raw = decompose_scalar(p1, p2, position)
            positions.append((torus_wrap(raw[0]), torus_wrap(raw[1])))
    else:
        print("trapezoid profile")
        # trapezoid profile
        accel_steps = int(accel_time / dt)
        cruise_distance = distance - 2 * accel_distance
        cruise_steps = int((cruise_distance / max_vel) / dt)
        # acceleration steps
        for i in range(accel_steps):
            accelerations.append(decompose_scalar(p1, p2, max_accel))
            velocity = max_accel * i * dt
            velocities.append(decompose_scalar(p1, p2, velocity))
            position = 0.5 * max_accel * (i * dt) ** 2
            positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, position)))
        # cruise steps
        for i in range(cruise_steps):
            accelerations.append(decompose_scalar(p1, p2, 0))
            velocity = max_vel
            velocities.append(decompose_scalar(p1, p2, velocity))
            position = accel_distance + max_vel * i * dt
            positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, position)))
        # deceleration steps
        for i in range(accel_steps):
            accelerations.append(decompose_scalar(p1, p2, -max_accel))
            velocity = max_vel - max_accel * i * dt
            velocities.append(decompose_scalar(p1, p2, velocity))
            position = distance - 0.5 * max_accel * ((accel_steps - i) * dt) ** 2
            positions.append(torus_tuple_wrap(decompose_scalar(p1, p2, position)))

    return (positions, velocities, accelerations)


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
