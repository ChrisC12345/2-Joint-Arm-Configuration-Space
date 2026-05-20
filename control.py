"""PID control logic for following a trajectory with a 2-link arm simulation"""

import math

from logger import Logger
import numpy as np
from torus import torus_diff


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, position, setpoint, dt):
        error = torus_diff(setpoint, position)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.integral = 0
        self.prev_error = 0


class TrajectoryFollower:
    def __init__(
        self, arm_sim, controller1, controller2, kS=None, kV=None, kA=None, kG=None
    ):
        self.controller1 = controller1
        self.controller2 = controller2
        self.arm_sim = arm_sim
        self.kS = kS if kS is not None else (0.0, 0.0)
        self.kV = kV if kV is not None else calculate_kV(arm_sim)
        self.kA = kA if kA is not None else calculate_kA(arm_sim)
        self.kG = kG if kG is not None else calulate_kG(arm_sim)

    def follow_trajectory(self, trajectory, step, dt=0.02):
        """Output motor voltage based on trajectory
        with position, velocity, and acceleration

        Parameters
        ----
        trajectory: tuple of (positions, velocities, accelerations)
            where each is a list of (t1, t2) pairs for each time step
        step: current time step index
        dt: time step duration in seconds"""
        p_setpoint = trajectory.positions[step]
        v_setpoint = trajectory.velocities[step]
        a_setpoint = trajectory.accelerations[step]

        upper_pid = self.controller1.compute(
            self.arm_sim.upper_arm.position, p_setpoint[0], dt
        )
        forearm_pid = self.controller2.compute(
            self.arm_sim.forearm.position, p_setpoint[1], dt
        )

        upper_ff = v_setpoint[0] * self.kV[0] + a_setpoint[0] * self.kA[0]
        forearm_ff = v_setpoint[1] * self.kV[1] + a_setpoint[1] * self.kA[1]

        upper_gravity_ff = math.cos(self.arm_sim.upper_arm.position) * self.kG[0]
        forearm_gravity_ff = (
            math.cos(self.arm_sim.forearm.position + self.arm_sim.upper_arm.position)
            * self.kG[1]
        )

        upper_voltage = upper_pid + upper_ff + upper_gravity_ff
        forearm_voltage = forearm_pid + forearm_ff + forearm_gravity_ff

        Logger.recordData("upper_arm_pid", upper_pid)
        Logger.recordData("upper_arm_ff", upper_ff)
        Logger.recordData("upper_arm_gravity_ff", upper_gravity_ff)
        Logger.recordData("upper_arm_voltage", upper_voltage)

        Logger.recordData("forearm_pid", forearm_pid)
        Logger.recordData("forearm_ff", forearm_ff)
        Logger.recordData("forearm_gravity_ff", forearm_gravity_ff)
        Logger.recordData("forearm_voltage", forearm_voltage)

        Logger.recordData("traj state", trajectory.states[step])

        self.arm_sim.upper_arm.setVoltage(upper_voltage)
        self.arm_sim.forearm.setVoltage(forearm_voltage)

    def follow_position_trajectory(self, trajectory, step, dt=0.02):
        """Given a list of (t1_setpoint, t2_setpoint) pairs and a time step dt,
        compute and apply motor voltages to follow the trajectory."""
        setpoint = trajectory[step]

        upper_pid = self.controller1.compute(
            self.arm_sim.upper_arm.position, setpoint[0], dt
        )
        forearm_pid = self.controller2.compute(
            self.arm_sim.forearm.position, setpoint[1], dt
        )

        current = np.array(trajectory[step])
        next = np.array(trajectory[step + 1]) if step < len(trajectory) - 1 else current
        prev = np.array(trajectory[step - 1]) if step > 0 else current
        velocity = torus_diff(next, current) / dt
        acceleration = torus_diff(
            torus_diff(next, current), torus_diff(current, prev)
        ) / (dt**2)
        if step >= len(trajectory) - 1:
            velocity = np.array((0.0, 0.0))
            acceleration = np.array((0.0, 0.0))
        upper_ff = velocity[0] * self.kV[0] + acceleration[0] * self.kA[0]
        forearm_ff = velocity[1] * self.kV[1] + acceleration[1] * self.kA[1]

        upper_gravity_ff = math.cos(self.arm_sim.upper_arm.position) * self.kG[0]
        forearm_gravity_ff = (
            math.cos(self.arm_sim.forearm.position + self.arm_sim.upper_arm.position)
            * self.kG[1]
        )

        Logger.recordData("upper_arm_gravity_ff", upper_gravity_ff)

        upper_voltage = upper_pid + upper_ff + upper_gravity_ff
        forearm_voltage = forearm_pid + forearm_ff + forearm_gravity_ff

        Logger.recordData("upper_arm_pid", upper_pid)
        Logger.recordData("upper_arm_ff", upper_ff)
        Logger.recordData("upper_arm_voltage", upper_voltage)

        self.arm_sim.upper_arm.setVoltage(upper_voltage)
        self.arm_sim.forearm.setVoltage(forearm_voltage)


def calulate_kG(arm_sim, g=9.81):
    """calculate a theoretical value of kG"""
    forearm = arm_sim.forearm
    upper_arm = arm_sim.upper_arm
    kG1 = (
        (
            upper_arm.mass * upper_arm.dist_COM
            + forearm.mass * (upper_arm.length + forearm.dist_COM)
        )
        * g
        * upper_arm.resistance
        / (upper_arm.kE * upper_arm.gear_ratio)
    )
    kG2 = (
        forearm.mass
        * forearm.dist_COM
        * g
        * forearm.resistance
        / (forearm.kE * forearm.gear_ratio)
    )
    return kG1, kG2


def calculate_kV(arm_sim):
    """calculate a theoretical value of kV"""
    upper_kV = arm_sim.upper_arm.gear_ratio * arm_sim.upper_arm.kE
    forearm_kV = arm_sim.forearm.gear_ratio * arm_sim.forearm.kE
    return upper_kV, forearm_kV


def calculate_kA(arm_sim):
    """calculate a theoretical value of kA"""
    upper_kA = (
        (arm_sim.upper_arm.moi + arm_sim.upper_arm.length**2 * arm_sim.forearm.mass)
        * arm_sim.upper_arm.resistance
        / (arm_sim.upper_arm.kE * arm_sim.upper_arm.gear_ratio)
    )
    forearm_kA = (
        arm_sim.forearm.moi
        * arm_sim.forearm.resistance
        / (arm_sim.forearm.kE * arm_sim.forearm.gear_ratio)
    )
    return upper_kA, forearm_kA
