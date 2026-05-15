"""Trajectory generation for robotic arms"""

import numpy as np
from torus import torus_diff

def linear_traj(path, dt = 0.02):
    """follow a path with constant speed in configuration space
    generally not a good trajectory, but simple to implement"""
    positions = np.array(path)
    velocities = []
    for i in range(len(path) - 1):
        current = np.array(path[i])
        next = np.array(path[i + 1])
        velocities.append(torus_diff(next, current) / dt)
    velocities.append(np.zeros(2))

    return positions, np.array(velocities)

