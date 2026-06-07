"""defines obstacle classes and collision checking functions for a 2D arm simulation"""

from enum import Enum
import numpy as np


class ObstacleType(Enum):
    CIRCLE = "circle"
    POLYGON = "polygon"


class Obstacle:
    def __init__(self, type, params):
        self.type = type
        self.params = params

    def getType(self):
        return self.type

    def getParams(self):
        return self.params


class CircleObstacle(Obstacle):
    def __init__(self, center, radius):
        super().__init__(ObstacleType.CIRCLE, (center, radius))


class PolygonObstacle(Obstacle):
    """vertices should be a list of numpy arrays (x,y) in cw or ccw order,
    at least 1 vertex"""

    def __init__(self, vertices):
        super().__init__(ObstacleType.POLYGON, vertices)


def is_collision_circle(p1, p2, center, radius):
    """returns True if segment collides with circle"""
    d = p2 - p1
    t = np.dot(center - p1, d) / np.dot(d, d)
    if t < 0:
        return np.linalg.norm(center - p1) < radius
    elif t > 1:
        return np.linalg.norm(center - p2) < radius
    else:
        closest = p1 + t * d
        return np.linalg.norm(center - closest) < radius


def is_collision_segment(p1, p2, p3, p4):
    """returns True if the segments p1-p2 and p3-p4 intersect including endpoints"""
    # checks if p1-p3-p2-p4 is a convex quadrilateral
    e1 = p3 - p1
    e2 = p2 - p3
    e3 = p4 - p2
    e4 = p1 - p4
    c1, c2, c3, c4 = (
        np.cross(e1, e2),
        np.cross(e2, e3),
        np.cross(e3, e4),
        np.cross(e4, e1),
    )

    # sign of cross product determines cw/ccw orientation so all must have same sign to be convex
    if c1 > 0 and c2 > 0 and c3 > 0 and c4 > 0:
        return True
    elif c1 < 0 and c2 < 0 and c3 < 0 and c4 < 0:
        return True
    elif c1 != 0 and c2 != 0 and c3 != 0 and c4 != 0:
        return False

    # deal with pesky edge case of collinearity, won't happen 99% of the time but we want to be robust
    def on_segment(p1, p2, p3):
        """given three collinear points,
        returns True if p3 is on the line segment p1-p2 including endpoints"""
        return min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(
            p1[1], p2[1]
        ) <= p3[1] <= max(p1[1], p2[1])

    if c1 == 0 and on_segment(p1, p2, p3):
        return True
    if c2 == 0 and on_segment(p3, p4, p2):
        return True
    if c3 == 0 and on_segment(p1, p2, p4):
        return True
    if c4 == 0 and on_segment(p3, p4, p1):
        return True

    return False


def is_collision_polygon(p1, p2, polygon):
    """returns True if segment p1-p2 collides with polygon,
    treating endpoints as not part of the segment"""
    for i in range(len(polygon)):
        p3 = polygon[i]
        p4 = polygon[(i + 1) % len(polygon)]
        if is_collision_segment(p1, p2, p3, p4):
            return True
    return False


def seg_circle_vec(p1x, p1y, p2x, p2y, cx, cy, r2):
    """Vectorized segment-circle collision; p1/p2 are arrays, cx/cy/r2 are scalars."""
    dx, dy = p2x - p1x, p2y - p1y
    ls = np.where(dx * dx + dy * dy == 0, 1e-30, dx * dx + dy * dy)  # norm squared
    # parameter of closest point from seg p1-p2 to c using vector projection
    t = np.clip(((cx - p1x) * dx + (cy - p1y) * dy) / ls, 0.0, 1.0)
    qx, qy = p1x + t * dx - cx, p1y + t * dy - cy
    return qx * qx + qy * qy < r2


def seg_seg_vec(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """Vectorized segment-segment intersection; p1/p2 are arrays, p3/p4 are scalars."""
    e1x, e1y = p3x - p1x, p3y - p1y
    e2x, e2y = p2x - p3x, p2y - p3y
    e3x, e3y = p4x - p2x, p4y - p2y
    e4x, e4y = p1x - p4x, p1y - p4y

    # use cross product to determine if consecutive edges are all in cw or ccw
    # positive cross product means second vector is ccw of first, negative means cw
    c1 = e1x * e2y - e1y * e2x
    c2 = e2x * e3y - e2y * e3x
    c3 = e3x * e4y - e3y * e4x
    c4 = e4x * e1y - e4y * e1x
    r = ((c1 > 0) & (c2 > 0) & (c3 > 0) & (c4 > 0)) | (
        (c1 < 0) & (c2 < 0) & (c3 < 0) & (c4 < 0)
    )

    # skip unecessary compute
    return r

    # handle collinear edge case
    # this is actually pretty unessecary since the probability of a random segment
    # being exactly collinear with an edge is basically 0, but we want to be robust
    r |= (
        (c1 == 0)
        & (np.minimum(p1x, p2x) <= p3x)
        & (p3x <= np.maximum(p1x, p2x))
        & (np.minimum(p1y, p2y) <= p3y)
        & (p3y <= np.maximum(p1y, p2y))
    )
    r |= (
        (c2 == 0)
        & (min(p3x, p4x) <= p2x)
        & (p2x <= max(p3x, p4x))
        & (min(p3y, p4y) <= p2y)
        & (p2y <= max(p3y, p4y))
    )
    r |= (
        (c3 == 0)
        & (np.minimum(p1x, p2x) <= p4x)
        & (p4x <= np.maximum(p1x, p2x))
        & (np.minimum(p1y, p2y) <= p4y)
        & (p4y <= np.maximum(p1y, p2y))
    )
    r |= (
        (c4 == 0)
        & (min(p3x, p4x) <= p1x)
        & (p1x <= max(p3x, p4x))
        & (min(p3y, p4y) <= p1y)
        & (p1y <= max(p3y, p4y))
    )
    return r
