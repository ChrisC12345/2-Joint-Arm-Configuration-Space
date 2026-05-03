import math
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
    def __init__(self, arm_sim, controller1, controller2, kV = (0.1, 0.1), kG = (4, 2)):
        self.controller1 = controller1
        self.controller2 = controller2
        self.arm_sim = arm_sim
        self.kV = kV
        self.kG = kG

    def follow_trajectory(self, trajectory, time, dt = 0.02):
        """Given a list of (t1_setpoint, t2_setpoint) pairs and a time step dt,
        compute and apply motor voltages to follow the trajectory."""
        setpoint = trajectory[time]

        upper_pid = self.controller1.compute(self.arm_sim.position, setpoint[0], dt)
        lower_pid = self.controller2.compute(self.arm_sim.position, setpoint[1], dt)

        # need to add torus wraparound
        velocity = (trajectory[time+1] - trajectory[time])/dt if time < len(trajectory)-1 else (0,0)
        upper_ff = velocity[0] * self.kV[0]
        lower_ff = velocity[1] * self.kV[1]

        upper_gravity_ff = math.cos(self.arm_sim.position) * self.kG[0]
        lower_gravity_ff = math.cos(self.arm_sim.position) * self.kG[1]

        upper_voltage = upper_pid + upper_ff + upper_gravity_ff
        lower_voltage = lower_pid + lower_ff + lower_gravity_ff

        self.arm_sim.upperArm.setVoltage(upper_voltage)
        self.arm_sim.lowerArm.setVoltage(lower_voltage)