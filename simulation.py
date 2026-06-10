"""calculates the physics of a single or double joint arm with motors at the joints"""

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from logger import Logger
from torus import torus_wrap


class SingleJointArmSim:
    """simulates a single joint arm with a motor at the joint"""

    def __init__(
        self,
        mass=10,
        length=0.25,
        distCOM=None,
        moi=None,
        gear_ratio=20.0,
        kE=0.02,
        resistance=0.03,
        nominalVoltage=12.0,
        friction=0.0,
        max_friction=0.0,
        dt=0.02,
    ):
        self.mass = mass
        self.length = length
        self.dist_COM = distCOM if distCOM is not None else length / 2
        self.moi = moi if moi is not None else mass * (length**2) / 3
        self.gear_ratio = gear_ratio
        self.kE = kE
        self.resistance = resistance
        self.nominal_voltage = nominalVoltage
        self.friction = friction
        self.max_static_friction = max_friction
        self.dt = dt

        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.motor_torque = 0.0
        self.torque = 0.0
        self.endpoint = (0.0, 0.0)
        self.voltage = 0.0
        self.current = 0.0
        self.motor_powered = True

    def setMotorPowered(self, motorPowered):
        self.motor_powered = motorPowered

    def setPosition(self, position):
        self.position = torus_wrap(position)
        self.endpoint = (
            self.length * math.cos(self.position),
            self.length * math.sin(self.position),
        )

    def setVoltage(self, voltage):
        self.voltage = max(-self.nominal_voltage, min(self.nominal_voltage, voltage))

    def set_motor_torque(self):
        self.current = (
            (self.voltage - self.kE * self.velocity * self.gear_ratio) / self.resistance
            if self.motor_powered
            else 0.0
        )
        self.motor_torque = self.kE * self.current * self.gear_ratio

    def update(self, torque):
        """update arm physics for one time step given total torque"""
        self.torque = (
            torque - math.copysign(self.friction, self.velocity)
            if math.fabs(torque) > self.max_static_friction
            or math.fabs(self.velocity) > 1e-3
            else 0.0
        )
        self.acceleration = self.torque / self.moi
        self.velocity += self.acceleration * self.dt
        self.position += (
            self.velocity - self.acceleration * self.dt / 2
        ) * self.dt  # assumes constant acceleration during time step
        self.position = torus_wrap(self.position)
        self.endpoint = (
            self.length * math.cos(self.position),
            self.length * math.sin(self.position),
        )


class DoubleJointArmSim:
    """simulates a double joint arm with motors at both joints,
    the second joint is at the end of the first segment"""

    def __init__(self, upper_arm, forearm, g=9.81):
        self.upper_arm = upper_arm
        self.forearm = forearm
        self.g = g

    def update(self):
        self.upper_arm.set_motor_torque()
        self.forearm.set_motor_torque()
        torques = self.calculateTorques()
        self.upper_arm.update(torques[0])
        self.forearm.update(torques[1])

    def calculateTorques(self):
        """Returns: torques on each joint from gravity, about motor
        Coriolis/centrifugal effects, and motor, derived from the Lagrangian EOM.
        """
        m1, l1, r1 = self.upper_arm.mass, self.upper_arm.length, self.upper_arm.dist_COM
        m2, r2 = self.forearm.mass, self.forearm.dist_COM
        g = -self.g

        t1, t2 = self.upper_arm.position, self.forearm.position
        w1, w2 = self.upper_arm.velocity, self.forearm.velocity

        # Coriolis/centrifugal coupling coefficient (dM12/dt2 term)
        h = m2 * l1 * r2 * math.sin(t2)

        # raw external torques: -V - G
        tau_ext1 = (
            h * (2 * w1 * w2 + w2**2)
            + (m1 * r1 + m2 * l1) * g * math.cos(t1)
            + m2 * r2 * g * math.cos(t1 + t2)
            + self.upper_arm.motor_torque
        )
        tau_ext2 = (
            -h * w1**2 + m2 * r2 * g * math.cos(t1 + t2) + self.forearm.motor_torque
        )

        # full mass matrix (from Lagrangian)
        I1, I2 = self.upper_arm.moi, self.forearm.moi
        M11 = I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * r2 * math.cos(t2)
        M12 = I2 + m2 * l1 * r2 * math.cos(t2)
        M22 = I2
        det = M11 * M22 - M12**2

        # solve M*alpha = tau_ext, then back out the effective per-joint torque
        # such that singleJointArmSim's  alpha = torque / moi  gives the right answer
        alpha1 = (M22 * tau_ext1 - M12 * tau_ext2) / det
        alpha2 = (-M12 * tau_ext1 + M11 * tau_ext2) / det

        return I1 * alpha1, I2 * alpha2


def animateFreeFall(
    arm, t1_init=math.pi / 2, t2_init=0.0, w1_init=0.0, w2_init=0.0, grid=None
):
    """Simulate and animate the double arm falling freely under gravity,
    with the configuration-space path shown alongside.

    arm:      DoubleJointArmSim instance (its state will be overwritten)
    t1_init:  initial upper arm angle in radians (default: pi/2, pointing up)
    t2_init:  initial forearm angle relative to upper arm in radians
    w1_init, w2_init: initial angular velocities in rad/s
    grid:     optional 2D collision grid from animation.generate_cspace();
              if provided the cspace panel shows the obstacle overlay
    """
    arm.upper_arm.setMotorPowered(False)
    arm.forearm.setMotorPowered(False)
    arm.upper_arm.setPosition(t1_init)
    arm.upper_arm.velocity = w1_init
    arm.forearm.setPosition(t2_init)
    arm.forearm.velocity = w2_init

    dt = arm.upper_arm.dt
    l1, l2 = arm.upper_arm.length, arm.forearm.length

    reach = (l1 + l2) * 1.1
    fig, (ax, ax_cs) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Double Arm — No Motor Power", fontsize=14)

    # --- left: real world ---
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_title("Real World")
    ax.grid(True, alpha=0.3)
    ax.plot(0, 0, "ko", markersize=8)

    (link1,) = ax.plot([], [], "m-", linewidth=4, solid_capstyle="round")
    (link2,) = ax.plot([], [], "b-", linewidth=3, solid_capstyle="round")
    (elbow_dot,) = ax.plot([], [], "ko", markersize=6)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)
    tip_x, tip_y = [], []
    (tip_trace,) = ax.plot([], [], "r-", alpha=0.5, linewidth=1)

    # --- right: configuration space ---
    ax_cs.set_facecolor("white")
    ax_cs.set_title("Configuration Space")
    ax_cs.set_xlabel("θ₁")
    ax_cs.set_ylabel("θ₂")
    ax_cs.set_xlim(-math.pi, math.pi)
    ax_cs.set_ylim(-math.pi, math.pi)

    if grid is not None:
        cmap = mcolors.ListedColormap(["white", "#D85A30"])
        ax_cs.imshow(
            grid,
            origin="lower",
            extent=[-math.pi, math.pi, -math.pi, math.pi],
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

    def _wrap(angle):
        return ((angle + math.pi) % (2 * math.pi)) - math.pi

    t1w0, t2w0 = _wrap(t1_init), _wrap(t2_init)
    ax_cs.plot([t1w0], [t2w0], "ro", markersize=8, zorder=6, label="start")

    cs_t1, cs_t2 = [], []
    (cs_trace,) = ax_cs.plot([], [], "c-", linewidth=1.5, alpha=0.8, label="path")
    (cs_dot,) = ax_cs.plot([], [], "bo", markersize=8, zorder=5, label="position")
    ax_cs.legend(loc="upper right", fontsize=8)

    t_elapsed = [0.0]

    def draw_frame(_):
        Logger.graphData("torque1", arm.upper_arm.torque)
        Logger.graphData("torque2", arm.forearm.torque)
        Logger.recordData("current1", arm.upper_arm.current)
        Logger.recordData("current2", arm.forearm.current)
        Logger.graphData("voltage1", arm.upper_arm.voltage)
        Logger.graphData("voltage2", arm.forearm.voltage)
        Logger.update()

        arm.update()
        t1 = arm.upper_arm.position
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
        tip_trace.set_data(tip_x, tip_y)
        t_elapsed[0] += dt
        time_text.set_text(f"t = {t_elapsed[0]:.2f} s")

        t1w, t2w = _wrap(t1), _wrap(t2)
        cs_t1.append(t1w)
        cs_t2.append(t2w)

        # insert NaN breaks at torus wrap-arounds so the line doesn't streak
        t1_plot, t2_plot = [], []
        for i in range(len(cs_t1)):
            t1_plot.append(cs_t1[i])
            t2_plot.append(cs_t2[i])
            if i < len(cs_t1) - 1 and (
                abs(cs_t1[i + 1] - cs_t1[i]) > math.pi
                or abs(cs_t2[i + 1] - cs_t2[i]) > math.pi
            ):
                t1_plot.append(float("nan"))
                t2_plot.append(float("nan"))

        cs_trace.set_data(t1_plot, t2_plot)
        cs_dot.set_data([t1w], [t2w])

        return link1, link2, elbow_dot, time_text, tip_trace, cs_trace, cs_dot

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=None,
        interval=dt * 1000,
        blit=True,
        cache_frame_data=False,
    )
    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    upper_arm = SingleJointArmSim(dt=0.01)
    forearm = SingleJointArmSim(dt=0.01)
    upper_arm.friction = 2
    forearm.friction = 2
    upper_arm.max_static_friction = 0
    forearm.max_static_friction = 0
    upper_arm.setVoltage(0)
    forearm.setVoltage(0)
    arm = DoubleJointArmSim(upper_arm, forearm)
    ani = animateFreeFall(arm, t1_init=0.2, t2_init=0)
