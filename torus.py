"""utility functions for working with angles in a toroidal space,
used for the C-space of a 2-link arm where each joint angle wraps around at 2*pi."""

import math
import numpy as np


def torus_point(point):
    """Wraps a single point to the range [-pi, pi]."""
    return ((point + math.pi) % (2 * math.pi)) - math.pi


def torus_diff(a, b):
    """Returns the difference between two angles a and b, wrapped to [-pi, pi]."""
    diff = ((a - b + math.pi) % (2 * math.pi)) - math.pi
    return diff


def torus_point_diff(a, b):
    """Returns the torus difference between two points a and b, applied elementwise."""
    diff = []
    for i in range(len(a)):
        diff_i = torus_diff(a[i], b[i])
        diff.append(diff_i)
    return diff


def torus_dist_sq(a, nodes):
    """Vectorized squared torus distance from point a to every row in nodes array."""
    diff = np.abs(nodes - a)
    diff = np.minimum(diff, 2 * math.pi - diff)
    return diff[:, 0] ** 2 + diff[:, 1] ** 2
