from simulation import SingleJointArmSim, DoubleJointArmSim
from control import (
    PIDController,
    TrajectoryFollower,
    calculate_kA,
    calculate_kS,
    calculate_kV,
    calculate_kG
)


class C:
    # --- Arm geometry (centimeters; simulation uses meters = L / 100) ---
    L1 = 40
    L2 = 30

    # --- Simulation ---
    DT = 0.02
    GRAVITY = 9.80665  # m/s²

    # --- Upper arm physical properties ---
    UA_MASS = 10  # kg
    UA_DIST_COM = None  # m from pivot; None = L1/200 (half-length)
    UA_MOI = None  # kg·m²; None = mass * length² / 3

    # --- Forearm physical properties ---
    FA_MASS = 10  # kg
    FA_DIST_COM = None  # m from elbow; None = L2/200
    FA_MOI = None  # kg·m²; None = mass * length² / 3

    # --- Motor model (both joints) ---
    GEAR_RATIO = 20.0
    BACK_EMF_CONSTANT = 0.02  # kE, V·s/rad
    MOTOR_RESISTANCE = 0.03  # ohms
    NOMINAL_VOLTAGE = 12.0  # V
    FRICTION = 1.0
    MAX_STATIC_FRICTION = 2.0

    # --- PID gains — joint 1 (upper arm) ---
    J1_KP = 24
    J1_KI = 48
    J1_KD = 3

    # --- PID gains — joint 2 (forearm) ---
    J2_KP = 16
    J2_KI = 32
    J2_KD = 0.3

    # --- Path planning ---
    RRT_MAX_ITER = 2000
    RRT_STEP_SIZE = 0.2
    RRT_GOAL_BIAS = 0.1
    RRT_REWIRE_RADIUS = 0.2
    CSPACE_GRID_N = 200
    SMOOTH_NODE_COST = 0.5

    # --- Trajectory ---
    TRAJ_V_MAX = (2.5, 2.5)  # rad/s per joint
    TRAJ_A_MAX = (5.0, 5.0)  # rad/s² per joint
    TRAJ_TURN_RADIUS = 0.3  # rad, c-space arc radius at waypoints
    TRAJ_V_TURN = 1.0  # rad/s, speed through waypoint arcs

    # --- Simulation objects ---
    upper_arm = SingleJointArmSim(
        mass=UA_MASS,
        length=L1 / 100,
        distCOM=UA_DIST_COM,
        moi=UA_MOI,
        gear_ratio=GEAR_RATIO,
        kE=BACK_EMF_CONSTANT,
        resistance=MOTOR_RESISTANCE,
        nominalVoltage=NOMINAL_VOLTAGE,
        friction=FRICTION,
        max_friction=MAX_STATIC_FRICTION,
        dt=DT,
    )
    forearm = SingleJointArmSim(
        mass=FA_MASS,
        length=L2 / 100,
        distCOM=FA_DIST_COM,
        moi=FA_MOI,
        gear_ratio=GEAR_RATIO,
        kE=BACK_EMF_CONSTANT,
        resistance=MOTOR_RESISTANCE,
        nominalVoltage=NOMINAL_VOLTAGE,
        friction=FRICTION,
        max_friction=MAX_STATIC_FRICTION,
        dt=DT,
    )
    sim = DoubleJointArmSim(upper_arm, forearm, DT)
    pid1 = PIDController(Kp=J1_KP, Ki=J1_KI, Kd=J1_KD)
    pid2 = PIDController(Kp=J2_KP, Ki=J2_KI, Kd=J2_KD)

    kS = calculate_kS(sim)
    kV = calculate_kV(sim)
    kA = calculate_kA(sim)
    kG = calculate_kG(sim)
    traj_follower = TrajectoryFollower(sim, pid1, pid2, kS, kV, kA, kG)


if __name__ == "__main__":
    print(f'kS: {calculate_kS(C.sim)}')
    print(f'kV: {calculate_kV(C.sim)}')
    print(f'kA: {calculate_kA(C.sim)}')
    print(f'kG: {calculate_kG(C.sim)}')
