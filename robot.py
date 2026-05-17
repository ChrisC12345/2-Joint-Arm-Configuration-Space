# type: ignore
"""Entry point: wires arm simulation, PID control, path planning, and animation
into one FRC-style robot loop."""

import matplotlib.pyplot as plt
import arm
from arm import is_collision
from pathing import rrt, smooth_path, interpolate_path, is_reachable
from simulation import SingleJointArmSim, DoubleJointArmSim
from control import PIDController, TrajectoryFollower
from logger import Logger
import animation
from trajectory import *


class Robot:
    DT = 0.02  # 20 ms loop period, like FRC

    def __init__(self):
        self.sim = None
        self.pid1 = None
        self.pid2 = None
        self.follower = None
        self.trajectory = None
        self.step = 0
        self.t1 = 0.0
        self.t2 = 0.0
        self.setpoint_t1 = 0.0
        self.setpoint_t2 = 0.0

    def robotInit(self, trajectory):
        upper_arm = SingleJointArmSim(length=arm.L1 / 100)
        forearm = SingleJointArmSim(length=arm.L2 / 100)
        self.sim = DoubleJointArmSim(upper_arm, forearm)
        self.sim.upper_arm.setMotorPowered(True)
        self.sim.forearm.setMotorPowered(True)
        self.pid1 = PIDController(Kp=12, Ki=0.0, Kd=1.5)
        self.pid2 = PIDController(Kp=3, Ki=0.0, Kd=0.3)
        self.follower = TrajectoryFollower(self.sim, self.pid1, self.pid2)
        self.trajectory = trajectory
        self.step = 0
        self.sim.upper_arm.setPosition(trajectory.positions[0][0])
        self.sim.upper_arm.velocity = 0.0
        self.sim.forearm.setPosition(trajectory.positions[0][1])
        self.sim.forearm.velocity = 0.0
        self._sync_state()

    def teleopPeriodic(self):
        if self.trajectory is None:
            return
        idx = min(self.step, len(self.trajectory) - 1)
        Logger.recordData("voltage1", self.sim.upper_arm.voltage)
        Logger.recordData("voltage2", self.sim.forearm.voltage)
        Logger.update()
        self.follower.follow_trajectory(self.trajectory, idx, self.DT)
        self.sim.update()
        if self.step < len(self.trajectory) - 1:
            self.step += 1
        self._sync_state()

    def capture_state(self):
        return (
            self.sim.upper_arm.position,
            self.sim.upper_arm.velocity,
            self.sim.forearm.position,
            self.sim.forearm.velocity,
            self.pid1.integral,
            self.pid1.prev_error,
            self.pid2.integral,
            self.pid2.prev_error,
        )

    def restore_state(self, snapshot):
        ua_pos, ua_vel, fa_pos, fa_vel, p1i, p1e, p2i, p2e = snapshot
        self.sim.upper_arm.position = ua_pos
        self.sim.upper_arm.velocity = ua_vel
        self.sim.forearm.position = fa_pos
        self.sim.forearm.velocity = fa_vel
        self.pid1.integral = p1i
        self.pid1.prev_error = p1e
        self.pid2.integral = p2i
        self.pid2.prev_error = p2e
        self._sync_state()

    def reset(self):
        if self.trajectory is not None:
            self.robotInit(self.trajectory)

    def _sync_state(self):
        self.t1 = self.sim.upper_arm.position
        self.t2 = self.sim.forearm.position
        if self.trajectory is not None:
            idx = min(self.step, len(self.trajectory) - 1)
            sp = self.trajectory.positions[idx]
            self.setpoint_t1, self.setpoint_t2 = sp[0], sp[1]


if __name__ == "__main__":
    obstacles = animation.setup_obstacles()
    grid = animation.generate_cspace(obstacles)
    result = animation.pick_start_goal(grid)

    if result is None:
        print("need to click start and goal")
    else:
        start, goal = result
        print(f"start: {start}, goal: {goal}")
        print("start in collision:", is_collision(start[0], start[1], obstacles))
        print("goal in collision:", is_collision(goal[0], goal[1], obstacles))

        if not is_reachable(grid, start, goal):
            print("no path exists — goal is not reachable from start")
        else:
            path = rrt(start, goal, obstacles, max_iter=10000, step_size=0.2)
            if path is None:
                print("no path found")
            else:
                print("path length before smoothing:", len(path))
                smoothed = smooth_path(path.copy(), obstacles)
                print("path length after smoothing:", len(smoothed))
                interp_smoothed = interpolate_path(smoothed)
                trapezoid_trajectory = trapezoidal_traj(
                    smoothed, max_vel=(2.0, 2.0), max_accel=(4.0, 4.0)
                )
                print(trapezoid_trajectory)

                robot = Robot()
                robot.robotInit(trapezoid_trajectory)
                animation.animate_path(
                    trapezoid_trajectory.positions,
                    obstacles,
                    grid,
                    title="smoothed — PID",
                    robot=robot,
                )
                plt.show()
