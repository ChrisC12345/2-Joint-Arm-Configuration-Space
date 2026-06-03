"""kinematics and collision checking for a 2-link arm in 2D with circular
and polygonal obstacles"""

import math
import numpy as np
from enum import Enum
from obstacles import ObstacleType
from obstacles import (
    is_collision_circle,
    is_collision_polygon,
    seg_circle_vec,
    seg_seg_vec,
)

L1 = 40
L2 = 30


def set_arm_lengths(l1, l2):
    global L1, L2
    L1 = l1
    L2 = l2


def forward_kinematics(t1, t2):
    """returns (elbow, tip) as (x,y) tuples"""
    elbow = np.array([L1 * math.cos(t1), L1 * math.sin(t1)])
    angle = t1 + t2
    elbowToTip = np.array([L2 * math.cos(angle), L2 * math.sin(angle)])
    return elbow, elbow + elbowToTip


def is_collision_batch(t1, t2, obstacles):
    """Vectorized is_collision for arrays of configs.
    Returns bool array of same shape as t1/t2."""
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    ex = L1 * np.cos(t1)
    ey = L1 * np.sin(t1)
    angle = t1 + t2
    tx = ex + L2 * np.cos(angle)
    ty = ey + L2 * np.sin(angle)
    ox = np.zeros_like(ex)
    oy = np.zeros_like(ey)
    result = np.zeros_like(t1, dtype=bool)
    for obs in obstacles:
        if obs.getType() == ObstacleType.CIRCLE:
            center, radius = obs.getParams()
            cx, cy, r2 = float(center[0]), float(center[1]), float(radius) ** 2
            result |= seg_circle_vec(ox, oy, ex, ey, cx, cy, r2)
            result |= seg_circle_vec(ex, ey, tx, ty, cx, cy, r2)
        elif obs.getType() == ObstacleType.POLYGON:
            verts = obs.getParams()
            for k in range(len(verts)):
                p3x, p3y = float(verts[k][0]), float(verts[k][1])
                p4x, p4y = (
                    float(verts[(k + 1) % len(verts)][0]),
                    float(verts[(k + 1) % len(verts)][1]),
                )
                result |= seg_seg_vec(ox, oy, ex, ey, p3x, p3y, p4x, p4y)
                result |= seg_seg_vec(ex, ey, tx, ty, p3x, p3y, p4x, p4y)
    return result


def is_collision(t1, t2, obstacles):
    # checks both links against all obstacles
    # obstacle is a tuple (obstacle type, parameters)
    origin = np.array((0, 0))
    elbow, tip = forward_kinematics(t1, t2)
    for obstacle in obstacles:
        if obstacle.getType() == ObstacleType.POLYGON:
            vertices = obstacle.getParams()
            if is_collision_polygon(origin, elbow, vertices) or is_collision_polygon(
                elbow, tip, vertices
            ):
                return True
        elif obstacle.getType() == ObstacleType.CIRCLE:
            center, radius = obstacle.getParams()
            if is_collision_circle(
                origin, elbow, center, radius
            ) or is_collision_circle(elbow, tip, center, radius):
                return True
    return False
