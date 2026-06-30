# Double Jointed Arm Configuration Space Motion Planner

Motion planning — finding a collision-free path for a robot arm — is a fundamental problem in robotics. This project implements it from scratch using configuration space theory and the RRT algorithm, then closes the loop with a PID + feedforward controller running on a full physics simulation of the arm.

The configuration space (C-space) is a 2D plane whose axes represent the two joint angles. Each point in it represents a unique arm pose, and obstacles in the real world become regions to avoid in this space. A path through free C-space is a sequence of arm poses that avoids all collisions.

## Architecture

```
robot.py          ← entry point: wires everything together in an FRC-style 20 ms loop
├── animation.py  ← interactive UI: obstacle placement, C-space display, path animation
├── pathing.py    ← RRT planner, path smoothing, interpolation, reachability check
├── arm.py        ← forward kinematics and collision detection
├── obstacles.py  ← obstacle classes (circle, polygon) and vectorized collision math
├── control.py    ← PID controller and trajectory follower with feedforward
├── simulation.py ← physics simulation using Euler-Lagrange dynamics
├── trajectory.py ← trajectory generation (linear, trapezoidal)
├── torus.py      ← toroidal geometry utilities for angle wraparound
└── logger.py     ← real-time matplotlib data logger
```

## Files

### robot.py
Entry point. Runs an FRC-style periodic loop (20 ms timestep) that ties together path planning, the physics simulation, and PID control. Interactively places obstacles, generates a C-space, lets the user pick start and goal, runs RRT, smooths and interpolates the path, then animates the arm following it under PID control.

### arm.py
Forward kinematics and collision detection for the 2-link arm. Computes elbow and tip coordinates from joint angles, and checks both arm segments against all obstacles. Includes a vectorized `is_collision_batch` for fast C-space rasterization.

### obstacles.py
Defines `CircleObstacle` and `PolygonObstacle` classes, plus the underlying collision geometry:
- Segment–circle via dot-product projection
- Segment–segment via cross-product convex-quadrilateral test (with collinearity edge cases)
- Fully vectorized numpy versions of both for batch collision queries

### pathing.py
- **RRT** — rapidly-exploring random tree in toroidal C-space with 10% goal bias and pre-allocated node storage
- **`smooth_path`** — greedy shortcutting that removes unnecessary waypoints
- **`interpolate_path`** — densifies the path to a fixed angular resolution
- **`is_reachable`** — BFS flood-fill on the C-space grid to confirm a path exists before running RRT

### simulation.py
Physics simulation using the Euler-Lagrange equations of motion for a 2-link planar arm:
- `SingleJointArmSim` — one joint with a DC motor model (back-EMF, gear ratio, resistance, nominal voltage)
- `DoubleJointArmSim` — assembles two joints, computes the full mass matrix, Coriolis/centrifugal coupling, and gravity torques, then integrates acceleration at each timestep
- `animateFreeFall` — free-fall simulation with no motor power, for visualizing the raw dynamics

### control.py
- **`PIDController`** — standard PID with toroidal error (handles angle wraparound correctly)
- **`TrajectoryFollower`** — combines PID with three feedforward terms computed from the motor model:
  - **kV** (velocity feedforward): `gear_ratio × kE`
  - **kA** (acceleration feedforward): derived from arm inertia, motor resistance, and gear ratio
  - **kG** (gravity feedforward): `cos(θ)` term scaled by motor back-EMF constant

### trajectory.py
Trajectory generation between C-space waypoints:
- **`linear_traj`** — constant-speed traversal through each waypoint
- **`trapezoidal_traj`** / **`trap_traj_endpts`** — trapezoidal velocity profile (accelerate → cruise → decelerate), respecting per-joint velocity and acceleration limits

### torus.py
Utilities for working with angles in C-space, which wraps at ±π (a torus topology):
- `torus_diff` — shortest angular difference, wrapped to [−π, π]
- `torus_point` — wrap an angle back into [−π, π]
- `torus_dist_sq` — vectorized squared toroidal distance for nearest-neighbor lookup in RRT

### animation.py
Interactive matplotlib UI (requires `QtAgg` backend):
- **Obstacle placement** — click to add circles (center + edge click) or polygons/lines (vertex clicks + Enter); undo and configurable arm lengths
- **C-space viewer** — displays the rasterized collision grid; click start then goal
- **`animate_path`** — side-by-side real-world and C-space animation with:
  - PID mode: shows actual vs. setpoint arm pose, traces actual tip and C-space trajectory
  - Kinematic mode: pure path playback without physics
  - Spacebar to pause/resume, arrow keys to step frame-by-frame (with state rewind in PID mode)
  - Reset button to replay from the start

### logger.py
A lightweight real-time dashboard that opens a small matplotlib window and displays key-value pairs (voltages, feedforward terms, etc.) as the simulation runs. Updates at up to 10 fps and auto-expands to two columns for larger datasets.

It also supports live scrolling graphs via `Logger.graphData(key, value, group=None)`, drawn in a separate window. Each value is appended to a rolling buffer and plotted over time; series sharing a `group` are overlaid on the same axes (e.g. actual vs. setpoint), while ungrouped series each get their own subplot. Non-numeric and non-finite values are ignored. `Logger.update()` refreshes both the text panel and the graphs.

## Math

- **Forward kinematics**: standard 2-link planar arm via trigonometry and vector addition
- **Collision detection**: segment–circle via dot-product projection; segment–segment via cross-product orientation test
- **Lagrangian dynamics**: full 2×2 mass matrix with Coriolis/centrifugal coupling and gravity; solved each timestep for joint accelerations
- **C-space sampling**: toroidal distance metric so the planner handles angle wraparound correctly
- **RRT**: probabilistically complete; with goal bias converges faster in practice
- **Feedforward**: kV/kA derived from motor electrical model; kG from the static gravity balance condition

## Usage

```
pip install numpy matplotlib pyqt6
python robot.py
```

1. Place obstacles in the workspace window (circles or polygons), then click **Done**
2. Click a start point, then a goal point in the C-space window
3. The arm animates following the planned path under PID + feedforward control


Full transparency: a significant amount of ai was used in the making of this project. Although, all the essential math and algorithms were created by me. I also wrote the pathing, tajectory, control, torus and various other files myself (with a bit of ai in-line suggestions, but they are my ideas). The logging and animation is mostly written by ai because graphics were not the focus of this project. The focus was mainly on understanding the alogorithms, math, and physics.


## Future Work
1. Add obstacle avoidance for circular trajectories