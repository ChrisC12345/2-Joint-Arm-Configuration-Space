"""kinematics and collision checking for a 2-link arm in 2D with circular
and polygonal obstacles"""

import math
import numpy as np
from obstacles import ObstacleType
from obstacles import (
    is_collision_circle,
    is_collision_polygon,
    seg_circle_vec,
    seg_seg_vec,
)


class Arm:
    L1 = 40
    L2 = 30

    @classmethod
    def set_lengths(cls, l1, l2):
        cls.L1 = l1
        cls.L2 = l2

    @classmethod
    def forward_kinematics(cls, t1, t2):
        """Returns (elbow, tip) as (x, y) numpy arrays."""
        elbow = np.array([cls.L1 * math.cos(t1), cls.L1 * math.sin(t1)])
        angle = t1 + t2
        elbow_to_tip = np.array([cls.L2 * math.cos(angle), cls.L2 * math.sin(angle)])
        return elbow, elbow + elbow_to_tip

    @classmethod
    def is_collision_batch(cls, t1, t2, obstacles):
        """Vectorized collision check for arrays of configs.
        Returns bool array of same shape as t1/t2."""
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        ex = cls.L1 * np.cos(t1)
        ey = cls.L1 * np.sin(t1)
        angle = t1 + t2
        tx = ex + cls.L2 * np.cos(angle)
        ty = ey + cls.L2 * np.sin(angle)
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

    @classmethod
    def is_collision(cls, t1, t2, obstacles):
        origin = np.array((0, 0))
        elbow, tip = cls.forward_kinematics(t1, t2)
        for obstacle in obstacles:
            if obstacle.getType() == ObstacleType.POLYGON:
                vertices = obstacle.getParams()
                if is_collision_polygon(
                    origin, elbow, vertices
                ) or is_collision_polygon(elbow, tip, vertices):
                    return True
            elif obstacle.getType() == ObstacleType.CIRCLE:
                center, radius = obstacle.getParams()
                if is_collision_circle(
                    origin, elbow, center, radius
                ) or is_collision_circle(elbow, tip, center, radius):
                    return True
        return False
