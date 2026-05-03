# visualize the c-space of a 2-link arm and animate a path through it

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.widgets as mwidgets
import math
import arm
from arm import forward_kinematics, is_collision, is_collision_batch
from obstacles import Obstacle, ObstacleType
from pathing import rrt, smooth_path, interpolate_path, is_reachable
from matplotlib.animation import FuncAnimation
from arm_sim import SingleJointArmSim, DoubleJointArmSim
from control import PIDController, TrajectoryFollower

def draw_cspace(obstacles):
    N = 200
    T1, T2 = np.meshgrid(np.linspace(-math.pi, math.pi, N),
                         np.linspace(-math.pi, math.pi, N))
    return is_collision_batch(T1, T2, obstacles).astype(float)

def animate_path(path, rrt_path, obstacles, title='path', use_pid=False):
    """
    path:     (t1, t2) setpoint sequence — animated directly or used as PID reference
    rrt_path: original RRT path drawn as reference in c-space
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # left plot — real world
    reach = (arm.L1 + arm.L2) * 1.15
    ax1.set_xlim(-reach, reach)
    ax1.set_ylim(-reach, reach)
    ax1.set_aspect('equal')
    ax1.set_title('Real World')
    ax1.grid(True, alpha=0.2)

    for obstacle in obstacles:
        if obstacle.getType() == ObstacleType.CIRCLE:
            center, radius = obstacle.getParams()
            ax1.add_patch(plt.Circle(center, radius, color='#D85A30', alpha=0.4))
        elif obstacle.getType() == ObstacleType.POLYGON:
            vertices = obstacle.getParams()
            if len(vertices) >= 3:
                ax1.add_patch(plt.Polygon(vertices, color='#D85A30', alpha=0.4))
            else:
                ax1.plot([vertices[0][0], vertices[1][0]], [vertices[0][1], vertices[1][1]],
                         color='#D85A30', linewidth=3, alpha=0.8, solid_capstyle='round')

    link1, = ax1.plot([], [], 'm-', linewidth=4, solid_capstyle='round')
    link2, = ax1.plot([], [], 'b-', linewidth=3, solid_capstyle='round')
    ax1.plot([0], [0], 'ko', markersize=6)
    elbow_dot, = ax1.plot([], [], 'ko', markersize=6)

    start = forward_kinematics(rrt_path[0][0], rrt_path[0][1])
    goal  = forward_kinematics(rrt_path[-1][0], rrt_path[-1][1])
    ax1.plot([0, start[0][0]], [0, start[0][1]], color='red',   linewidth=2, solid_capstyle='round')
    ax1.plot([start[0][0], start[1][0]], [start[0][1], start[1][1]], color='red',   linewidth=2, solid_capstyle='round')
    ax1.plot([0, goal[0][0]],  [0, goal[0][1]],  color='green', linewidth=2, solid_capstyle='round')
    ax1.plot([goal[0][0], goal[1][0]],  [goal[0][1], goal[1][1]],  color='green', linewidth=2, solid_capstyle='round')

    # right plot — c-space with rrt_path as reference
    ax2.set_facecolor('white')
    cmap = mcolors.ListedColormap(['white', '#D85A30'])
    ax2.imshow(grid, origin='lower',
               extent=[-math.pi, math.pi, -math.pi, math.pi],
               cmap=cmap, vmin=0, vmax=1)
    ax2.set_title('Configuration Space')
    ax2.plot([rrt_path[0][0]], [rrt_path[0][1]], 'ro', markersize=8)
    ax2.plot([rrt_path[-1][0]], [rrt_path[-1][1]], 'go', markersize=8)
    rrt_t1s = [p[0] for p in rrt_path]
    rrt_t2s = [p[1] for p in rrt_path]
    segments_t1 = [[]]
    segments_t2 = [[]]
    for i in range(len(rrt_t1s)):
        segments_t1[-1].append(rrt_t1s[i])
        segments_t2[-1].append(rrt_t2s[i])
        if i < len(rrt_t1s) - 1:
            if abs(rrt_t1s[i+1] - rrt_t1s[i]) > math.pi or abs(rrt_t2s[i+1] - rrt_t2s[i]) > math.pi:
                segments_t1.append([])
                segments_t2.append([])
    for seg_t1, seg_t2 in zip(segments_t1, segments_t2):
        ax2.plot(seg_t1, seg_t2, 'm-', linewidth=1.5, alpha=0.5)

    dot,    = ax2.plot([], [], 'bo', markersize=8, zorder=5, label='actual')
    sp_dot, = ax2.plot([], [], 'o',  markersize=6, color='orange', zorder=4, label='setpoint')

    t1_arr = np.array([p[0] for p in path])
    t2_arr = np.array([p[1] for p in path])

    # reusable 2-element buffers — mutated in-place each frame, no allocation
    _l1x = np.array([0.0, 0.0])
    _l1y = np.array([0.0, 0.0])
    _l2x = np.array([0.0, 0.0])
    _l2y = np.array([0.0, 0.0])

    if use_pid:
        upperArm = SingleJointArmSim()
        forearm  = SingleJointArmSim()
        sim = DoubleJointArmSim(upperArm, forearm)
        upperArm.setMotorPowered(True)
        forearm.setMotorPowered(True)
        pid1 = PIDController(Kp=10.0, Ki=0.0, Kd=1.0)
        pid2 = PIDController(Kp=10.0, Ki=0.0, Kd=1.0)
        traj_follower = TrajectoryFollower(sim, pid1, pid2)

        # PID comparison artists
        pid_trace,  = ax2.plot([], [], '-', color='cyan', linewidth=1.5, alpha=0.8, zorder=3, label='actual path')
        sp_link1,   = ax1.plot([], [], '--', color='purple', linewidth=2,  alpha=0.45, solid_capstyle='round')
        sp_link2,   = ax1.plot([], [], '--', color='blue', linewidth=1.5, alpha=0.45, solid_capstyle='round')
        tip_trace,  = ax1.plot([], [], '-', color='cyan', linewidth=1.2, alpha=0.7)
        ax2.legend(loc='upper right', fontsize=8)
        ax1.plot([], [], '--', color='cyan', alpha=0.6, label='setpoint')
        ax1.plot([], [], '-', color='cyan', alpha=0.7,  label='actual tip')
        ax1.legend(loc='upper right', fontsize=8)

        _pid_t1_hist = []
        _pid_t2_hist = []
        _tip_x_hist  = []
        _tip_y_hist  = []

        def _reset_sim():
            sim.upperArm.setPosition(path[0][0]);  sim.upperArm.velocity = 0.0
            sim.forearm.setPosition(path[0][1]);   sim.forearm.velocity  = 0.0
            pid1.reset();  pid2.reset()
            _pid_t1_hist.clear();  _pid_t2_hist.clear()
            _tip_x_hist.clear();   _tip_y_hist.clear()
            pid_trace.set_data([], [])
            tip_trace.set_data([], [])

        _reset_sim()
    else:
        frames = [forward_kinematics(t1, t2) for t1, t2 in path]
        elbow_x = np.array([f[0][0] for f in frames])
        elbow_y = np.array([f[0][1] for f in frames])
        tip_x   = np.array([f[1][0] for f in frames])
        tip_y   = np.array([f[1][1] for f in frames])

    _step  = [0]
    paused = [False]

    def update(_):
        idx = min(_step[0], len(path) - 1)
        _step[0] += 1

        if use_pid:
            traj_follower.follow_trajectory(path, idx)
            sim.update()
            fk = forward_kinematics(sim.upperArm.position, sim.forearm.position)
            ex, ey = fk[0];  tx, ty = fk[1]
            t1_dot, t2_dot = sim.upperArm.position, sim.forearm.position

            # setpoint ghost arm
            sp_fk = forward_kinematics(t1_arr[idx], t2_arr[idx])
            sp_ex, sp_ey = sp_fk[0];  sp_tx, sp_ty = sp_fk[1]
            sp_link1.set_data([0, sp_ex], [0, sp_ey])
            sp_link2.set_data([sp_ex, sp_tx], [sp_ey, sp_ty])

            # accumulate actual traces
            _pid_t1_hist.append(t1_dot)
            _pid_t2_hist.append(t2_dot)
            _tip_x_hist.append(tx)
            _tip_y_hist.append(ty)
            pid_trace.set_data(_pid_t1_hist, _pid_t2_hist)
            tip_trace.set_data(_tip_x_hist, _tip_y_hist)

            sp_dot.set_data([t1_arr[idx]], [t2_arr[idx]])
        else:
            ex, ey = elbow_x[idx], elbow_y[idx]
            tx, ty = tip_x[idx],   tip_y[idx]
            t1_dot, t2_dot = t1_arr[idx], t2_arr[idx]
            sp_dot.set_data([], [])

        _l1x[1] = ex;  _l1y[1] = ey
        _l2x[0] = ex;  _l2x[1] = tx
        _l2y[0] = ey;  _l2y[1] = ty
        link1.set_data(_l1x, _l1y)
        link2.set_data(_l2x, _l2y)
        elbow_dot.set_data([ex], [ey])
        dot.set_data([t1_dot], [t2_dot])

        if use_pid:
            return link1, link2, elbow_dot, dot, sp_dot, pid_trace, sp_link1, sp_link2, tip_trace
        return link1, link2, elbow_dot, dot, sp_dot

    def on_key(event):
        if event.key == ' ':
            if paused[0]:
                ani.resume()
            else:
                ani.pause()
            paused[0] = not paused[0]
        elif event.key == 'right' and paused[0]:
            update(None);  fig.canvas.draw()
        elif event.key == 'left' and paused[0]:
            _step[0] = max(_step[0] - 2, 0)
            update(None);  fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)

    ani = FuncAnimation(fig, update, frames=None,
                        interval=20, blit=True,
                        cache_frame_data=False)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    ax_btn = fig.add_axes([0.45, 0.02, 0.12, 0.05])
    btn_reset = mwidgets.Button(ax_btn, 'Reset')

    def on_reset(_):
        _step[0] = 0
        if use_pid:
            _reset_sim()
        if paused[0]:
            paused[0] = False
            ani.resume()

    btn_reset.on_clicked(on_reset)

    # prevent garbage collection — FuncAnimation and Button are local vars
    fig._anim = ani
    fig._btn  = btn_reset

# ── Obstacle Setup ────────────────────────────────────────────────────────────

OBSTACLE = []

fig_setup, ax_setup = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.24)
_init_reach = (arm.L1 + arm.L2) * 1.15
ax_setup.set_xlim(-_init_reach, _init_reach)
ax_setup.set_ylim(-_init_reach, _init_reach)
ax_setup.set_aspect('equal')
ax_setup.set_title('Place obstacles in the workspace')
ax_setup.grid(True, alpha=0.2)
ax_setup.plot(0, 0, 'ko', markersize=8)   # arm base

status_text = ax_setup.text(
    0.5, 1.1, '', transform=ax_setup.transAxes,
    ha='center', va='bottom', fontsize=11, color='steelblue'
)

ax_btn_circle  = fig_setup.add_axes([0.02, 0.04, 0.20, 0.08])
ax_btn_polygon = fig_setup.add_axes([0.26, 0.04, 0.20, 0.08])
ax_btn_undo    = fig_setup.add_axes([0.50, 0.04, 0.20, 0.08])
ax_btn_done    = fig_setup.add_axes([0.74, 0.04, 0.22, 0.08])
btn_circle  = mwidgets.Button(ax_btn_circle,  'Add Circle')
btn_polygon = mwidgets.Button(ax_btn_polygon, 'Add Polygon/Line')
btn_undo    = mwidgets.Button(ax_btn_undo,    'Undo Last')
btn_done    = mwidgets.Button(ax_btn_done,    'Done')

ax_tb_l1 = fig_setup.add_axes([0.20, 0.14, 0.18, 0.06])
ax_tb_l2 = fig_setup.add_axes([0.62, 0.14, 0.18, 0.06])
tb_l1 = mwidgets.TextBox(ax_tb_l1, 'L1:', initial=str(arm.L1))
tb_l2 = mwidgets.TextBox(ax_tb_l2, 'L2:', initial=str(arm.L2))

def _update_workspace_limits(_=None):
    try:
        l1 = float(tb_l1.text)
        l2 = float(tb_l2.text)
        if l1 > 0 and l2 > 0:
            arm.set_arm_lengths(l1, l2)
            lim = (l1 + l2) * 1.15
            ax_setup.set_xlim(-lim, lim)
            ax_setup.set_ylim(-lim, lim)
            fig_setup.canvas.draw_idle()
    except ValueError:
        pass

tb_l1.on_submit(_update_workspace_limits)
tb_l2.on_submit(_update_workspace_limits)

# mutable interaction state
_mode          = ['idle']   # 'idle' | 'circle_center' | 'circle_edge' | 'polygon'
_tmp_center    = [None]
_tmp_center_dot= [None]
_tmp_poly_verts = []
_tmp_poly_dots  = []
_tmp_poly_lines = []
_obstacle_artists = []   # tracks drawn obstacle patches/lines for clean redraw

def _set_mode(m, msg=''):
    count = len(OBSTACLE)
    count_str = f'  [{count} obstacle{"s" if count != 1 else ""}]' if count else ''
    _mode[0] = m
    status_text.set_text(msg + count_str)
    fig_setup.canvas.draw_idle()

def _redraw_obstacles():
    global _obstacle_artists
    for artist in _obstacle_artists:
        artist.remove()
    _obstacle_artists = []
    for obs in OBSTACLE:
        if obs.getType() == ObstacleType.CIRCLE:
            center, radius = obs.getParams()
            patch = plt.Circle(center, radius, color='#D85A30', alpha=0.5)
            ax_setup.add_patch(patch)
            _obstacle_artists.append(patch)
        else:
            verts = obs.getParams()
            if len(verts) >= 3:
                patch = plt.Polygon(verts, color='#D85A30', alpha=0.5)
                ax_setup.add_patch(patch)
                _obstacle_artists.append(patch)
            else:
                line, = ax_setup.plot([verts[0][0], verts[1][0]], [verts[0][1], verts[1][1]],
                                      color='#D85A30', linewidth=3, alpha=0.8, solid_capstyle='round')
                _obstacle_artists.append(line)
    fig_setup.canvas.draw_idle()

def _clear_poly_preview():
    for d in _tmp_poly_dots:  d.remove()
    for l in _tmp_poly_lines: l.remove()
    _tmp_poly_dots.clear()
    _tmp_poly_lines.clear()
    _tmp_poly_verts.clear()

def _on_circle_btn(_event):
    _clear_poly_preview()
    _set_mode('circle_center', 'Click the center of the circle')

def _on_polygon_btn(_event):
    _clear_poly_preview()
    _set_mode('polygon', 'Click 2+ vertices; Enter to finish (2 = line, 3+ = polygon)')

def _on_undo_btn(_event):
    if _mode[0] == 'polygon' and _tmp_poly_verts:
        # cancel in-progress polygon
        _clear_poly_preview()
        _set_mode('idle', '')
    elif OBSTACLE:
        OBSTACLE.pop()
        _redraw_obstacles()
        _set_mode(_mode[0], '')

def _on_done_btn(_event):
    if _mode[0] == 'polygon' and len(_tmp_poly_verts) >= 2:
        OBSTACLE.append(Obstacle(ObstacleType.POLYGON, list(_tmp_poly_verts)))
        _clear_poly_preview()
    _update_workspace_limits()
    plt.close(fig_setup)

btn_circle.on_clicked(_on_circle_btn)
btn_polygon.on_clicked(_on_polygon_btn)
btn_undo.on_clicked(_on_undo_btn)
btn_done.on_clicked(_on_done_btn)

def _on_setup_click(event):
    if event.inaxes != ax_setup:
        return
    x, y = event.xdata, event.ydata

    if _mode[0] == 'circle_center':
        _tmp_center[0] = np.array([x, y])
        dot, = ax_setup.plot(x, y, 'r+', markersize=14, markeredgewidth=2)
        _tmp_center_dot[0] = dot
        _set_mode('circle_edge', 'Click a point on the edge to set the radius')

    elif _mode[0] == 'circle_edge':
        center = _tmp_center[0]
        radius = np.linalg.norm(np.array([x, y]) - center)
        if _tmp_center_dot[0] is not None:
            _tmp_center_dot[0].remove()
            _tmp_center_dot[0] = None
        _tmp_center[0] = None
        OBSTACLE.append(Obstacle(ObstacleType.CIRCLE, (center, radius)))
        _set_mode('idle', '')
        _redraw_obstacles()

    elif _mode[0] == 'polygon':
        _tmp_poly_verts.append(np.array([x, y]))
        dot, = ax_setup.plot(x, y, 'rs', markersize=7)
        _tmp_poly_dots.append(dot)
        if len(_tmp_poly_verts) > 1:
            v1 = _tmp_poly_verts[-2]
            v2 = _tmp_poly_verts[-1]
            line, = ax_setup.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r-', linewidth=1.5)
            _tmp_poly_lines.append(line)
        fig_setup.canvas.draw_idle()

def _on_setup_key(event):
    if event.key == 'enter':
        if _mode[0] == 'polygon' and len(_tmp_poly_verts) >= 2:
            OBSTACLE.append(Obstacle(ObstacleType.POLYGON, list(_tmp_poly_verts)))
            _clear_poly_preview()
            _set_mode('idle', '')
            _redraw_obstacles()
        elif _mode[0] == 'idle':
            plt.close(fig_setup)

fig_setup.canvas.mpl_connect('button_press_event', _on_setup_click)
fig_setup.canvas.mpl_connect('key_press_event', _on_setup_key)
plt.show()

# ── C-space planning ──────────────────────────────────────────────────────────

grid = draw_cspace(OBSTACLE)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('white')
cmap = mcolors.ListedColormap(['white', '#D85A30'])
ax.imshow(grid, origin='lower',
          extent=[-math.pi, math.pi, -math.pi, math.pi],
          cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel('θ₁')
ax.set_ylabel('θ₂')
ax.set_title('click start then goal')

clicks = []
markers = []

def on_click(event):
    if event.inaxes != ax or len(clicks) >= 2:
        return
    clicks.append(np.array([event.xdata, event.ydata]))
    color = 'ro' if len(clicks) == 1 else 'go'
    marker, = ax.plot(event.xdata, event.ydata, color, markersize=12)
    markers.append(marker)
    fig.canvas.draw()

def on_key(event):
    if event.key == 'enter' and len(clicks) == 2:
        plt.close()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

if len(clicks) < 2:
    print("need to click start and goal")
else:
    start, goal = clicks[0], clicks[1]
    print(f"start: {start}, goal: {goal}")
    print("start in collision:", is_collision(start[0], start[1], OBSTACLE))
    print("goal in collision:", is_collision(goal[0], goal[1], OBSTACLE))

    if not is_reachable(grid, start, goal):
        print("no path exists — goal is not reachable from start")

    else:
        path = rrt(start, goal, OBSTACLE, max_iter=10000, step_size=0.2)

        if path is None:
            print("no path found")
        else:
            print("path length before smoothing:", len(path))
            smoothed = smooth_path(path.copy(), OBSTACLE)
            print("path length after smoothing:", len(smoothed))
            interp_path     = interpolate_path(path)
            interp_smoothed = interpolate_path(smoothed)
            # animate_path(interp_path,     interp_path,     OBSTACLE, title='raw RRT')
            # animate_path(interp_smoothed, interp_smoothed, OBSTACLE, title='smoothed')
            animate_path(interp_smoothed, interp_smoothed, OBSTACLE, title='smoothed — PID', use_pid=True)

plt.show()
