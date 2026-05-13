"""RRT path planning and path smoothing for a 2-link arm in C-space, treating the space as a torus."""

import numpy as np
import math
from arm import is_collision_batch
from torus import torus_diff, torus_point, torus_dist_sq


def _line_free(a, b, obstacles):
    """Return True if the straight arc a→b in C-space is collision-free."""
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    n = max(4, int(np.linalg.norm(diff) / 0.05))
    ts = np.linspace(0, 1, n, endpoint=False)
    configs = ((a + ts[:, None] * diff) + math.pi) % (2 * math.pi) - math.pi
    return not np.any(is_collision_batch(configs[:, 0], configs[:, 1], obstacles))


def rrt(start, goal, obstacles, max_iter=5000, step_size=0.05):
    start = np.asarray(start, float)
    goal  = np.asarray(goal,  float)

    # pre-allocate storage — avoids repeated list reallocation
    nodes  = np.empty((max_iter + 2, 2))
    nodes[0] = start
    n_nodes  = 1
    parent   = [-1]   # parent[i] = index of parent, -1 for root

    GOAL_BIAS = 0.1

    for _ in range(max_iter):
        # sample: bias toward goal 10 % of the time
        point = goal if np.random.random() < GOAL_BIAS \
                     else np.random.uniform(-math.pi, math.pi, 2)

        # nearest neighbour — one vectorised numpy call instead of a Python loop
        nearest_idx = int(np.argmin(torus_dist_sq(point, nodes[:n_nodes])))
        nearest = nodes[nearest_idx]

        # steer toward sample by step_size
        direction = torus_diff(point, nearest)  # alternative using torus_diff function
        norm = math.hypot(direction[0], direction[1])
        if norm < 1e-10:
            continue
        new_node = torus_point(nearest + direction * (step_size / norm))

        if _line_free(nearest, new_node, obstacles):
            nodes[n_nodes] = new_node
            parent.append(nearest_idx)
            n_nodes += 1

            # check if we reached the goal
            if torus_dist_sq(goal, nodes[n_nodes-1:n_nodes])[0] < step_size ** 2:
                path = []
                idx = n_nodes - 1
                while idx >= 0:
                    path.append(tuple(nodes[idx]))
                    idx = parent[idx]
                path.reverse()
                return path

    return None


def smooth_path(path, obstacles):
    path = [np.asarray(p, float) for p in path]
    i = 0
    while i < len(path) - 2:
        if _line_free(path[i], path[i + 2], obstacles):
            path.pop(i + 1)
        else:
            i += 1
    return [tuple(p) for p in path]

def interpolate_path(path, resolution=0.05):
    dense_path = []
    for i in range(len(path) - 1):
        a = np.array(path[i])
        b = np.array(path[i+1])
        diff = torus_diff(b, a) 
        length = np.linalg.norm(diff)
        steps = max(2, int(length / resolution))
        for t in range(steps):
            config = a + (t / steps) * diff
            # wrap back into [-pi, pi]
            config = torus_point(config)
            dense_path.append(tuple(config))
    dense_path.append(path[-1])
    return dense_path

def is_reachable(grid, start, goal, N=200):
    '''Returns True if there is a path from start to goal through free cells in the grid, treating the grid as a torus.
    grid is a 2D numpy array where 0 = free and 1 = occupied. start and goal are (t1, t2) configs.'''
    # convert configs to grid indices
    def to_idx(config):
        i = int((config[0] + math.pi) / (2 * math.pi) * N)
        j = int((config[1] + math.pi) / (2 * math.pi) * N)
        return np.clip(i, 0, N-1), np.clip(j, 0, N-1)
    
    si, sj = to_idx(start)
    gi, gj = to_idx(goal)
    
    # BFS flood fill through free cells
    from collections import deque
    visited = np.zeros((N, N), dtype=bool)
    queue = deque([(si, sj)])
    visited[si, sj] = True
    
    while queue:
        i, j = queue.popleft()
        if i == gi and j == gj:
            return True
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = (i+di) % N, (j+dj) % N  # toroidal wraparound
            if not visited[ni, nj] and grid[nj, ni] == 0:
                visited[ni, nj] = True
                queue.append((ni, nj))
    return False