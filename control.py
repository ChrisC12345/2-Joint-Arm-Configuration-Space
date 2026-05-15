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

    def __init__(self, arm_sim, controller1, controller2, kV = None, kG = None):
        self.controller1 = controller1
        self.controller2 = controller2
        self.arm_sim = arm_sim
        self.kV = kV if kV is not None else calculate_kV(arm_sim)
        self.kG = kG if kG is not None else calulate_kG(arm_sim)

    def follow_trajectory(self, trajectory, step, dt = 0.02):
        """Given a list of (t1_setpoint, t2_setpoint) pairs and a time step dt,
        compute and apply motor voltages to follow the trajectory."""
        setpoint = trajectory[step]

        upper_pid = self.controller1.compute(self.arm_sim.upperArm.position, setpoint[0], dt)
        forearm_pid = self.controller2.compute(self.arm_sim.forearm.position, setpoint[1], dt)

        current = np.array(trajectory[step])
        next = np.array(trajectory[step+1]) if step < len(trajectory)-10 else current
        velocity = torus_diff(next, current) / dt
        if step == len(trajectory)-1:
            velocity = np.array((0.0, 0.0))
        upper_ff = velocity[0] * self.kV[0]
        forearm_ff = velocity[1] * self.kV[1]

        upper_gravity_ff = math.cos(self.arm_sim.upperArm.position) * self.kG[0]
        forearm_gravity_ff = math.cos(self.arm_sim.forearm.position + self.arm_sim.upperArm.position) * self.kG[1]

        Logger.recordData("forearm_g", forearm_gravity_ff)

        upper_voltage = upper_pid + upper_ff + upper_gravity_ff
        forearm_voltage = forearm_pid + forearm_ff + forearm_gravity_ff

        Logger.recordData("forearm_pid", forearm_pid)
        Logger.recordData("forearm_ff", forearm_ff)
        Logger.recordData("forearm_voltage", forearm_voltage)

        self.arm_sim.upperArm.setVoltage(upper_voltage)
        self.arm_sim.forearm.setVoltage(forearm_voltage)

def calulate_kG(arm_sim, g=9.81):
    '''calculate a theoretical value of kG'''
    forearm = arm_sim.forearm
    upperArm = arm_sim.upperArm
    kG1 = ((upperArm.mass * upperArm.dist_COM + forearm.mass * upperArm.length) 
            * g * upperArm.resistance 
            / (upperArm.kE * upperArm.gear_ratio))
    kG2 = (forearm.mass * forearm.dist_COM * g * forearm.resistance 
            / (forearm.kE * forearm.gear_ratio))
    return kG1, kG2

def calculate_kV(arm_sim):
    '''calculate a theoretical value of kV'''
    # this is a very rough estimate, assumes max velocity occurs at nominal voltage with no load
    upper_kV = arm_sim.upperArm.gear_ratio * arm_sim.upperArm.kE * 5
    forearm_kV = arm_sim.forearm.gear_ratio * arm_sim.forearm.kE * 2
    return upper_kV, forearm_kV