"""Trajectory generation for robotic arms"""

import math

import numpy as np
from torus import torus_diff

def linear_traj(path, dt = 0.02):
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

    return positions, np.array(velocities)

def trapezoidal_traj(waypoints, max_vel, max_accel, dt = 0.02):
    """follow a path with trapezoidal velocity profile in configuration space
    accelerates at max_accel until max_vel, cruises at max_vel, then decelerates at max_accel to stop at the end of the path"""
    
def trap_traj_endpts(p1, p2, max_velocities, max_accelerations, dt = 0.02):
    """helper function to generate a trapezoidal trajectory between two points
    returns three np arrays for position, velocity, acceleration at each time step
    
    Parameters
    -
    p1: starting point in configuration space (tuple of joint angles)
    p2: ending point in configuration space (tuple of joint angles)
    max_vel: tuple of max velocity in rad/s for each joint
    max_accel: tuple of max acceleration in rad/s^2 for each joint"""
    accelerations = np.array([])
    velocities = np.array([])
    positions = np.array([])

    max_accel = 0
    max_vel = 0

    slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[1] != p2[0] else math.inf

    # choose the more restrictive constraint between the two joints to ensure we don't exceed limits on either joint
    if max_accelerations[1]/max_accelerations[0] > slope:
        ratio = 1 / math.sqrt(1 + slope**2) # ratio between x and total components
        max_accel = max_accelerations[0]*ratio
        max_vel = max_velocities[0]*ratio
    else:
        ratio = 1 / math.sqrt(1 + (1/slope)**2) # ratio between y and total components
        max_accel = max_accelerations[1]*ratio
        max_vel = max_velocities[1]*ratio

    diff = torus_diff(p2, p1)

    # now consider it as 1D
    distance = np.linalg.norm(diff)
    accel_time = max_vel / max_accel
    accel_distance = 0.5 * max_accel * accel_time**2
    if distance < 2 * accel_distance:
        # triangle profile
        time_steps = 2 * int(math.ceil(math.sqrt(2 * distance / max_accel) / dt))
        # trapezoidal profile


