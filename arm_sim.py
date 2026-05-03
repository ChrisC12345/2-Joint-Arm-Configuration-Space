import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SingleJointArmSim:
    """simulates a single joint arm with a motor at the joint"""
    mass = 0.0; # mass of the arm in kg
    length = 0.0; # length of the single segmentarm in m
    distCOM = 0.0; # distance from joint to center of mass of the arm in m
    moi = 0.0; # moment of inertia of the arm about the joint in kg*m^2
    kE = 0.0; # back-emf constant of the motor in V/(rad/s)
    resistance = 0.0; # resistance of the motor in ohms
    nominalVoltage = 0.0; # maximum voltage that can be applied to the motor in volts

    position = 0.0; # angle of the arm in radians relative to motor, ccw pos
    velocity = 0.0; # angular velocity of the arm in radians/s, ccw pos
    acceleration = 0.0; # angular acceleration of the arm in radians/s^2, ccw pos
    torque = 0.0; # net torque applied to the arm in N*m, ccw pos
    endpoint = (0.0, 0.0); # (x,y) position of the end of the arm in m, relative to joint

    voltage = 0.0; # voltage applied to the motor in volts, positive voltage produces positive torque
    current = 0.0; # current through the motor in amps, positive current produces positive torque

    dt = 0.0; # time step for simulation in seconds
    motorPowered = True;

    def __init__(self, mass = 1, 
                 length = 0.25, 
                 distCOM = None,
                 moi = None, 
                 kE = 0.02, 
                 resistance = 0.03, 
                 nominalVoltage = 12.0,
                 dt = 0.02):
        self.mass = mass
        self.length = length
        self.distCOM = distCOM if distCOM is not None else length / 2
        self.moi = moi if moi is not None else mass * (length ** 2) / 3
        self.kE = kE
        self.resistance = resistance
        self.nominalVoltage = nominalVoltage
        self.dt = dt

    def setMotorPowered(self, motorPowered):
        self.motorPowered = motorPowered

    def setPosition(self, position):
        self.position = position
        self.endpoint = (self.length * math.cos(self.position), 
                         self.length * math.sin(self.position))
    
    def setVoltage(self, voltage):
        self.voltage = max(-self.nominalVoltage, min(self.nominalVoltage, voltage))

    def updateTorque(self, torque):
        self.torque = torque
        self.acceleration = self.torque / self.moi
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt
        self.endpoint = (self.length * math.cos(self.position), 
                         self.length * math.sin(self.position))

    def update(self, externalTorque):
        if self.motorPowered:
            self.current = (self.voltage - self.kE * self.velocity) / self.resistance
        else:
            self.current = 0
        self.torque = self.kE * self.current + externalTorque
        self.updateTorque(self.torque)

class DoubleJointArmSim:
    """simulates a double joint arm with motors at both joints, 
    the second joint is at the end of the first segment"""

    upperArm = SingleJointArmSim()
    forearm = SingleJointArmSim()

    g = -9.81; # gravity in m/s^2

    def __init__(self, upperArm, forearm):
        self.upperArm = upperArm
        self.forearm = forearm
    
    def calculateExternalTorques(self):
        """Returns (tau_ext1, tau_ext2): external torques on each joint from gravity
        and Coriolis/centrifugal effects, derived from the Lagrangian EOM.

        Euler-Lagrange EOM: M(q)*q_ddot + V(q, q_dot) + G(q) = tau_motor
        => external torques = -(V + G)

        Mass matrix M:
          M11 = I1 + I2 + m2*l1^2 + 2*m2*l1*r2*cos(t2)
          M12 = M21 = I2 + m2*l1*r2*cos(t2)
          M22 = I2
        Coriolis/centrifugal V (h = m2*l1*r2*sin(t2)):
          V1 = -h*(2*w1*w2 + w2^2)
          V2 = +h*w1^2
        Gravity G (PE = sum of -m*g*y since g < 0):
          G1 = -(m1*r1 + m2*l1)*g*cos(t1) - m2*r2*g*cos(t1+t2)
          G2 = -m2*r2*g*cos(t1+t2)
        """
        m1, l1, r1 = self.upperArm.mass, self.upperArm.length, self.upperArm.distCOM
        m2, r2      = self.forearm.mass, self.forearm.distCOM
        g           = self.g

        t1, t2 = self.upperArm.position, self.forearm.position
        w1, w2 = self.upperArm.velocity, self.forearm.velocity

        # Coriolis/centrifugal coupling coefficient (dM12/dt2 term)
        h = m2 * l1 * r2 * math.sin(t2)

        # raw external torques: -V - G
        tau_ext1 = (h * (2*w1*w2 + w2**2)
                    + (m1*r1 + m2*l1) * g * math.cos(t1)
                    + m2*r2 * g * math.cos(t1 + t2))
        tau_ext2 = (-h * w1**2
                    + m2*r2 * g * math.cos(t1 + t2))

        # full mass matrix (from Lagrangian)
        I1, I2 = self.upperArm.moi, self.forearm.moi
        M11 = I1 + I2 + m2*l1**2 + 2*m2*l1*r2*math.cos(t2)
        M12 = I2 + m2*l1*r2*math.cos(t2)
        M22 = I2
        det = M11*M22 - M12**2

        # solve M*alpha = tau_ext, then back out the effective per-joint torque
        # such that singleJointArmSim's  alpha = torque / moi  gives the right answer
        alpha1 = ( M22*tau_ext1 - M12*tau_ext2) / det
        alpha2 = (-M12*tau_ext1 + M11*tau_ext2) / det

        return I1*alpha1, I2*alpha2

    def update(self):
        externalTorques = self.calculateExternalTorques()
        self.upperArm.update(externalTorques[0])
        self.forearm.update(externalTorques[1])


def animateFreeFall(arm, t1_init=math.pi/2, t2_init=0.0, w1_init=0.0, w2_init=0.0):
    """Simulate and animate the double arm falling freely under gravity (no motor power).
    Runs indefinitely, computing physics on the fly each frame.

    arm:      doubleJointArmSim instance (its state will be overwritten)
    t1_init:  initial upper arm angle in radians (default: pi/2, pointing up)
    t2_init:  initial forearm angle relative to upper arm in radians
    w1_init, w2_init: initial angular velocities in rad/s
    """
    arm.upperArm.setMotorPowered(True)
    arm.forearm.setMotorPowered(True)
    arm.upperArm.setPosition(t1_init)
    arm.upperArm.velocity = w1_init
    arm.forearm.setPosition(t2_init)
    arm.forearm.velocity = w2_init

    dt = arm.upperArm.dt
    l1, l2 = arm.upperArm.length, arm.forearm.length

    reach = (l1 + l2) * 1.1
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect('equal')
    ax.set_title('Double Arm — No Motor Power')
    ax.grid(True, alpha=0.3)
    ax.plot(0, 0, 'ko', markersize=8)  # shoulder (fixed)

    link1,     = ax.plot([], [], 'm-', linewidth=4, solid_capstyle='round')
    link2,     = ax.plot([], [], 'b-', linewidth=3, solid_capstyle='round')
    elbow_dot, = ax.plot([], [], 'ko', markersize=6)
    time_text  = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

    tip_x = []
    tip_y = []
    trace, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)

    t_elapsed = [0.0]

    def draw_frame(_):
        arm.update()
        t1 = arm.upperArm.position
        t2 = arm.forearm.position
        ex = l1 * math.cos(t1)
        ey = l1 * math.sin(t1)
        tx = ex + l2 * math.cos(t1 + t2)
        ty = ey + l2 * math.sin(t1 + t2)
        link1.set_data([0, ex], [0, ey])
        link2.set_data([ex, tx], [ey, ty])
        elbow_dot.set_data([ex], [ey])
        tip_x.append(tx)
        tip_y.append(ty)
        trace.set_data(tip_x, tip_y)
        t_elapsed[0] += dt
        time_text.set_text(f't = {t_elapsed[0]:.2f} s')
        return link1, link2, elbow_dot, time_text, trace

    ani = animation.FuncAnimation(fig, draw_frame, frames=None,
                                  interval=dt * 1000, blit=True,
                                  cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return ani

if __name__ == '__main__':
    upperArm = SingleJointArmSim()
    forearm  = SingleJointArmSim()
    arm = DoubleJointArmSim(upperArm, forearm)
    ani = animateFreeFall(arm, t1_init=math.pi/2, t2_init=0.5)