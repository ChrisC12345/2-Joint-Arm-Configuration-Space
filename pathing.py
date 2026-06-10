"""RRT path planning and path smoothing for a 2-link arm in C-space,
treating the space as a torus."""

import heapq

import numpy as np
import math
from arm import is_collision_batch
from torus import torus_diff, torus_tuple_diff, torus_wrap, torus_dist_sq


class Path:
    def __init__(self, points, steps):
        """Represents a trajectory path with given points and
        vectors steps between points."""
        self.points = points  # list of tuples
        self.steps = steps  # list of np arrays

    def __len__(self):
        return len(self.points)


def _line_free(a, b, obstacles, resolution=0.05):
    """Return True if the straight arc a→b in C-space is collision-free."""
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    n = max(4, int(np.linalg.norm(diff) / resolution))
    ts = np.linspace(0, 1, n, endpoint=False)
    configs = ((a + ts[:, None] * diff) + math.pi) % (2 * math.pi) - math.pi
    return not np.any(is_collision_batch(configs[:, 0], configs[:, 1], obstacles))


def _line_free_vec(start, vec, obstacles, resolution=0.01):
    """Return True if the straight arc start -> start + vec in C-space
    is collision-free."""
    length = np.linalg.norm(vec)
    num_pts = max(2, int(length / resolution))
    points = torus_wrap(start + np.linspace(0, 1, num_pts)[:, None] * vec)
    return not np.any(is_collision_batch(points[:, 0], points[:, 1], obstacles))


def choose_new_node(nodes, n_nodes, bias, goal, step_size):
    """Choose a new node to add to the RRT. Returns (nearest_idx, new_node)"""
    # sample: bias toward goal 10 % of the time
    point = (
        goal if np.random.random() < bias else np.random.uniform(-math.pi, math.pi, 2)
    )

    # nearest neighbour — one vectorised numpy call instead of a Python loop
    nearest_idx = int(np.argmin(torus_dist_sq(point, nodes[:n_nodes])))
    nearest = nodes[nearest_idx]

    # steer toward sample by step_size
    direction = torus_diff(point, nearest)  # alternative using torus_diff function
    norm = math.hypot(direction[0], direction[1])
    if norm < 1e-10:
        return None, None
    new_node = torus_wrap(nearest + direction * (step_size / norm))
    return nearest_idx, new_node


def retrace_path(nodes, parent, idx):
    """Retrace path from start to node at index idx. Returns a Path object."""
    path = []
    steps = []
    idx = idx
    while idx > 0:
        path.append(tuple(nodes[idx]))
        steps.append(np.array(torus_tuple_diff(nodes[idx], nodes[parent[idx]])))
        idx = parent[idx]
    path.append(tuple(nodes[0]))  # add start
    path.reverse()
    steps.reverse()
    return Path(path, steps)


def rrt(start, goal, obstacles, max_iter=5000, step_size=0.05):
    """Basic RRT algorithm to find a path from start to goal in C-space."""
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)

    # pre-allocate storage — avoids repeated list reallocation
    nodes = np.empty((max_iter + 2, 2))
    nodes[0] = start
    n_nodes = 1
    parent = [-1]  # parent[i] = index of parent, -1 for root

    GOAL_BIAS = 0.1

    for i in range(max_iter):
        nearest_idx, new_node = choose_new_node(
            nodes, n_nodes, GOAL_BIAS, goal, step_size
        )

        if new_node is not None and _line_free(nodes[nearest_idx], new_node, obstacles):
            nodes[n_nodes] = new_node
            parent.append(nearest_idx)
            n_nodes += 1

            # check if we reached the goal
            if torus_dist_sq(goal, nodes[n_nodes - 1 : n_nodes])[0] < step_size**2:
                print(f"number of iterations: {i}")
                return retrace_path(nodes, parent, n_nodes - 1)

    return None


def choose_parent(new_node, nodes, parent, cost, nearest_idx, in_radius, obstacles):
    """Choose parent for new_node from nodes in in_radius, and update cost."""
    least_cost_idx = nearest_idx  # index of least cost node in radius
    nearest = nodes[nearest_idx]
    least_cost = cost[nearest_idx] + np.linalg.norm(torus_diff(new_node, nearest))
    for i in in_radius:
        node = nodes[i]
        if (
            _line_free(node, new_node, obstacles)
            and cost[i] + np.linalg.norm(torus_diff(new_node, nodes[i])) < least_cost
        ):
            least_cost_idx = i
            least_cost = cost[i] + np.linalg.norm(torus_diff(new_node, nodes[i]))
            parent[-1] = least_cost_idx
            cost[-1] = least_cost


def propagate_cost(i, nodes, parent, cost, n_nodes):
    """Recursively update cost of all descendants of node i."""
    for j in range(n_nodes):
        if parent[j] == i:
            cost[j] = cost[i] + np.linalg.norm(torus_diff(nodes[j], nodes[i]))
            propagate_cost(j, nodes, parent, cost, n_nodes)


def rewire(new_node, nodes, n_nodes, parent, cost, in_radius, obstacles):
    """Rewire nodes in in_radius to new_node if it reduces their cost."""
    for i in in_radius:
        node = nodes[i]
        if (
            _line_free(node, new_node, obstacles)
            and cost[-1] + np.linalg.norm(torus_diff(new_node, nodes[i])) < cost[i]
        ):
            parent[i] = n_nodes
            cost[i] = cost[-1] + np.linalg.norm(torus_diff(new_node, nodes[i]))
            propagate_cost(i, nodes, parent, cost, n_nodes + 1)


def rrt_star(start, goal, obstacles, max_iter=5000, step_size=0.05, radius=0.2):
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)

    nodes = np.empty((max_iter + 2, 2))
    nodes[0] = start
    n_nodes = 1  # number of nodes currently in the tree
    parent = [-1]  # index of parent node
    cost = [0.0]  # cost[i] = cost to reach node i
    best_goal_node = None  # index of best node

    GOAL_BIAS = 0.1

    for _ in range(max_iter):
        nearest_idx, new_node = choose_new_node(
            nodes, n_nodes, GOAL_BIAS, goal, step_size
        )
        if new_node is None:
            continue
        nearest = nodes[nearest_idx]

        if _line_free(nearest, new_node, obstacles):
            nodes[n_nodes] = new_node
            parent.append(nearest_idx)
            cost.append(
                cost[nearest_idx] + np.linalg.norm(torus_diff(new_node, nearest))
            )

            # choose parent from nodes within radius
            in_radius = np.where(torus_dist_sq(nodes[:n_nodes], new_node) < radius**2)[
                0
            ]  # indices of nodes within radius
            choose_parent(
                new_node, nodes, parent, cost, nearest_idx, in_radius, obstacles
            )

            # rewire
            rewire(new_node, nodes, n_nodes, parent, cost, in_radius, obstacles)

            if torus_dist_sq(goal, new_node.reshape(1, 2))[0] < step_size**2:
                if best_goal_node is None or cost[-1] < cost[best_goal_node]:
                    best_goal_node = n_nodes

            n_nodes += 1

    # get best path
    return (
        retrace_path(nodes, parent, best_goal_node)
        if best_goal_node is not None
        else None
    )


def smooth_greedy(rrt_path, obstacles):
    """Given a path from RRT, return a smoothed version by removing unnecessary
    waypoints using a greedy approach. Returns a new Path object."""
    path = [np.asarray(p, float) for p in rrt_path.points]
    vectors = [v.copy() for v in rrt_path.steps]
    i = 0
    vec = vectors[0]
    while i < len(path) - 2:
        vec += vectors[i + 1]
        if _line_free_vec(path[i], vec, obstacles):
            path.pop(i + 1)
            vectors.pop(i)
        else:
            vectors[i] = vec - vectors[i + 1]  # save accumulated displacement
            i += 1
            vec = vectors[i]
    if i < len(vectors) and np.linalg.norm(vec) > 0:
        vectors[i] = vec  # save last segment's accumulated displacement
    return Path([tuple(p) for p in path], vectors)


def smooth_dijkstra(rrt_path, obstacles, node_cost=0):
    """Given a path from RRT, return a smoothed version using dijkstra's algorithm.
    The cost is total distance + (n_nodes - 1) * node_cost"""
    cost = [float("inf")] * len(rrt_path)
    pq = []  # priority queue of (distance, index)
    cost[0] = 0
    parent = [-1] * len(rrt_path)
    heapq.heappush(pq, (0, 0))  # (distance, index)

    while pq:
        d, u = heapq.heappop(pq)
        if d > cost[u]:
            continue

        vec = np.zeros(2)
        for v in range(u + 1, len(rrt_path)):
            vec += rrt_path.steps[v - 1]
            if (
                _line_free_vec(rrt_path.points[u], vec, obstacles)
                and cost[u] + np.linalg.norm(vec) + node_cost < cost[v]
            ):
                cost[v] = cost[u] + np.linalg.norm(vec) + node_cost
                parent[v] = u
                heapq.heappush(pq, (cost[v], v))

    # retrace
    path = []
    steps = []
    idx = len(rrt_path) - 1
    while idx > 0:
        path.append(tuple(rrt_path.points[idx]))
        steps.append(sum(rrt_path.steps[parent[idx] : idx]))
        idx = parent[idx]
    path.append(tuple(rrt_path.points[0]))  # add start
    path.reverse()
    steps.reverse()
    return Path(path, steps)


# this isnt really used
def interpolate_path(path, resolution=0.05):
    dense_path = []
    for i in range(len(path) - 1):
        a = np.array(path[i])
        b = np.array(path[i + 1])
        diff = torus_diff(b, a)
        length = np.linalg.norm(diff)
        steps = max(2, int(length / resolution))
        for t in range(steps):
            config = a + (t / steps) * diff
            # wrap back into [-pi, pi]
            config = torus_wrap(config)
            dense_path.append(tuple(config))
    dense_path.append(path[-1])
    return dense_path


def is_reachable(grid, start, goal, N=200):
    """Returns True if there is a path from start to goal through free cells in the
    grid, treating the grid as a torus.
    grid is a 2D numpy array where 0 = free and 1 = occupied.
    start and goal are (t1, t2) configs."""

    # convert configs to grid indices
    def to_idx(config):
        i = int((config[0] + math.pi) / (2 * math.pi) * N)
        j = int((config[1] + math.pi) / (2 * math.pi) * N)
        return np.clip(i, 0, N - 1), np.clip(j, 0, N - 1)

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
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % N, (j + dj) % N  # toroidal wraparound
            if not visited[ni, nj] and grid[nj, ni] == 0:
                visited[ni, nj] = True
                queue.append((ni, nj))
    return False
