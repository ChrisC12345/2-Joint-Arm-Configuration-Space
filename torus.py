# file for doing torus calculations
import math
import numpy as np

def torus_point(point):
    '''Wraps a single point to the range [-pi, pi].'''
    return ((point + math.pi) % (2 * math.pi)) - math.pi

def torus_diff(a, b):
    '''Returns the difference between two angles a and b, wrapped to [-pi, pi].'''
    diff = ((a - b + math.pi) % (2 * math.pi)) - math.pi
    return diff

def torus_dist_sq(a, nodes):
    """Vectorized squared torus distance from point a to every row in nodes array."""
    diff = np.abs(nodes - a)
    diff = np.minimum(diff, 2 * math.pi - diff)
    return diff[:, 0] ** 2 + diff[:, 1] ** 2