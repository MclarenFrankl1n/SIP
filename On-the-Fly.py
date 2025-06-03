"""
Second Method: On-the-Fly
Objective: Find Ramp-up and Ramp-down time for each FOV
There are 3 components to this problem:
1. Find the constant velocity needed to match the exposure time for each projection.
2. Find ramp-up and ramp-down distance for each FOV. 
3. Find the total time taken to move from one FOV to another.
board movement timing with multi-axis synchronization

Set Bottom-Left Corner as the origin (0,0).

Assume S-curve velocity profile for the source with jerk properties.
Since velocity is constant, projection motion profile is all S-curve with cruise phase.
Though moving from one FOV to another, it might be triangular motion profile.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
ACCELERATION = 5_000_000_000  # nm/s²
VELOCITY = 1_000_000_000      # nm/s
JERK = 100_000_000_000          # nm/s³
PI = np.pi
R = 50_000_000                # nm    5cm
PROJ = 32
EXPO = 0.050  # seconds   50ms
CYCLE_TIME = 1.6  # seconds   10s

def calculate_maximum_velocity(radius, CycleTime):
    """
    Calculate the maximum velocity to be maintained to match cycle time.
    Parameters:
    Cycle time in seconds
    Returns:
    Maximum velocity in nm/s
    """
    V_max = 2 * radius * PI / CycleTime
    print(f"Maximum Velocity = {V_max:.2f} nm/s")
    # Scale jerk, accel, based on velocity
    def scale_params(jerk, accel, velocity):
        if V_max == 0:
            return jerk, accel, velocity  # no motion on this axis
        scale = V_max / velocity
        J = jerk * scale**3
        A = accel * scale**2
        V = V_max
        print(f"Jerk = {J:.2f} nm/s^3, Acceleration = {A:.2f} nm/s^2, Velocity = {V:.2f} nm/s,")
        return J, A, V

    JERK_S, ACCELERATION_S, VELOCITY_S = scale_params(JERK, ACCELERATION, VELOCITY)
    return JERK_S, ACCELERATION_S, VELOCITY_S

def calculate_ramp_up_down_time(velocity, acceleration, jerk):
    """
    Calculate the ramp-up and ramp-down time for a given velocity, acceleration, and jerk.
    Prints the time for each phase.
    Returns:
    Ramp-up time, Ramp-down time in seconds
    """
    t_j = acceleration / jerk  # time to reach max acceleration
    t_const_a = (velocity - acceleration * t_j) / acceleration  # time at constant acceleration
    if t_const_a < 0:
        t_const_a = 0  # If negative, profile is triangular
    ramp_up_time = 2 * t_j + t_const_a  # total ramp-up time
    ramp_down_time = ramp_up_time  # assuming symmetry

    print(f"Jerk-up time (t_j): {t_j:.5f} s")
    print(f"Constant acceleration time (t_const_a): {t_const_a:.5f} s")
    print(f"Jerk-down time (t_j): {t_j:.5f} s")
    print(f"Total ramp-up time: {ramp_up_time:.5f} s")
    print(f"Total ramp-down time: {ramp_down_time:.5f} s")
    return ramp_up_time, ramp_down_time

def calculate_triangular_ramp_up_down_time(v_target, j_max):
    """
    Compute ramp-up and ramp-down time assuming triangular motion profile.
    (No constant acceleration, just jerk up and down)
    
    Parameters:
    - v_target: Target cruise velocity [nm/s]
    - j_max: Maximum jerk [nm/s³]

    Returns:
    - t_ramp: Time to ramp up or ramp down [s]
    - s_ramp: Distance traveled during the ramp phase [nm]
    """
    # formula: t_j = sqrt(2*v_target/j_max)
    t_j = (2 * v_target / j_max) ** 0.5
    s_ramp = (1/3) * j_max * t_j**3   # Total distance covered during ramp phase

    print(f"Jerk-up time (t_j): {t_j:.5f} s")
    print(f"Jerk-down time (t_j): {t_j:.5f} s")
    print(f"Total ramp-up or ramp-down time: {2*t_j:.5f} s")
    print(f"Total ramp (up+down) time: {4*t_j:.5f} s")
    print(f"s_ramp (distance during ramp): {s_ramp:.1f} nm")
    return 4 * t_j, s_ramp  # Total ramp time (jerk up + down)

# Scan time per FOV (for all projections)
def calculate_scan_time(verbose=False):
    ramp_up_time, ramp_down_time = calculate_ramp_up_down_time(VELOCITY, ACCELERATION, JERK)
    scan_times = (EXPO * PROJ) + ramp_up_time + ramp_down_time
    if verbose:
        print(f"Total scan time for {PROJ} projections: {scan_times:.5f} seconds")
    return scan_times

def calculate_travel_time(distance, v_max, a_max, j_max, resolution=1000, verbose=False):
    t_j = a_max / j_max # time to reach max acceleration
    t_const_a = (v_max - a_max * t_j) / a_max #time at constant acceleration
    s_accel = 1/3 * j_max * t_j**3 + a_max * t_const_a * t_j + 0.5 * a_max * t_const_a**2
    s_total_accel_decel = 2 * s_accel
    if verbose:
        print(f"t_j = {t_j:.5f} s, t_const_a = {t_const_a:.5f} s")
        print(f"s_accel = {s_accel:.5f} nm, s_total_accel_decel = {s_total_accel_decel:.5f} nm")

    if distance < s_total_accel_decel:
        # Triangular motion profile
        t_j = (3 * distance / (2 * j_max))**(1/3)
        t_total = 4 * t_j
        t_vals = np.linspace(0, t_total, resolution)
        v_vals = np.zeros_like(t_vals)
        if verbose:
            print("Triangular Profile")

        t1 = t_j
        t2 = 2 * t_j
        t3 = 3 * t_j

        for i, t in enumerate(t_vals):
            if t < t1:
                v_vals[i] = 0.5 * j_max * t**2
            elif t < t2:
                dt = t - t_j
                v_vals[i] = (j_max * t_j**2 / 2) + j_max * t_j * dt - j_max * dt**2 / 2
            elif t < t3:
                dt = t - 2 * t_j
                v_vals[i] = (j_max * t_j**2) - j_max * dt**2 / 2
            else:
                dt = t - 3 * t_j
                v_vals[i] = j_max * t_j**2 / 2 - j_max * t_j * dt + j_max * dt**2 / 2

        return t_total, {
            'type': 'triangular',
            't': t_vals,
            'v': v_vals,
            't_total': t_total
        }

    else:
        # S-curve profile
        s_cruise = distance - s_total_accel_decel
        t_cruise = s_cruise / v_max
        print(f"t_cruise = {t_cruise:.5f} s")
        t_total = 2 * t_j + t_const_a + t_cruise + 2 * t_j + t_const_a
        t_vals = np.linspace(0, t_total, resolution)
        v_vals = np.zeros_like(t_vals)
        if verbose:
            print("S-Curve Profile")

        # Define phase boundaries
        t1 = t_j
        t2 = t1 + t_const_a
        t3 = t2 + t_j
        t4 = t3 + t_cruise
        t5 = t4 + t_j
        t6 = t5 + t_const_a
        t7 = t6 + t_j
        print(f"t1 = {t1:.5f} s, t2 = {t2:.5f} s, t3 = {t3:.5f} s, t4 = {t4:.5f} s")
        print(f"t5 = {t5:.5f} s, t6 = {t6:.5f} s, t7 = {t7:.5f} s")

        for i, t in enumerate(t_vals):
            if t < t1:
                # Phase 1: Jerk up
                v_vals[i] = 0.5 * j_max * t**2
            elif t < t2:
                # Phase 2: Constant acceleration
                dt = t - t1
                v_vals[i] = 0.5 * j_max * t_j**2 + a_max * dt
            elif t < t3:
                # Phase 3: Jerk down
                dt = t - t2
                v_vals[i] = 0.5 * j_max * t_j**2 + a_max * t_const_a + a_max * dt - 0.5 * j_max * dt**2
            elif t < t4:
                # Phase 4: Cruise
                v_vals[i] = v_max
            elif t < t5:
                # Phase 5: Jerk down (decel start)
                dt = t - t4
                v_vals[i] = v_max - 0.5 * j_max * dt**2
            elif t < t6:
                # Phase 6: Constant deceleration
                dt = t - t5
                v_vals[i] = v_max - 0.5 * j_max * t_j**2 - a_max * dt
            else:
                # Phase 7: Jerk up (decel end)
                dt = t - t6
                v_vals[i] = v_max - 0.5 * j_max * t_j**2 - a_max * t_const_a - a_max * dt + 0.5 * j_max * dt**2


        return t_total, {
            'type': 's-curve',
            't': t_vals,
            'v': v_vals,
            't_total': t_total
        }
    
    
# Multi-axis time normalization (P1, P2, Pn)
def synchronize_multi_axis_motion(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    t_x, _ = calculate_travel_time(dx, VELOCITY, ACCELERATION, JERK)
    t_y, _ = calculate_travel_time(dy, VELOCITY, ACCELERATION, JERK)

    t_sync = max(t_x, t_y)
    sync_axis = 'X' if t_x >= t_y else 'Y'
    print(f"Sync axis: {sync_axis} (t_x={t_x:.5f} s, t_y={t_y:.5f} s)")

    # Scale jerk, accel, velocity for each axis
    def scale_params(t_i):
        if t_i == 0:
            return JERK, ACCELERATION, VELOCITY  # no motion on this axis
        scale = (t_i / t_sync)
        j = JERK * scale**3
        a = ACCELERATION * scale**2
        v = VELOCITY * scale
        return j, a, v

    j_x, a_x, v_x = scale_params(t_x)
    j_y, a_y, v_y = scale_params(t_y)

    return {
        't_sync': t_sync,
        'sync_axis': sync_axis,
        'x': {'dist': dx, 'jerk': j_x, 'accel': a_x, 'vel': v_x},
        'y': {'dist': dy, 'jerk': j_y, 'accel': a_y, 'vel': v_y},
    }

# Updated board movement timing with multi-axis synchronization
def calculate_board_movement_time(rectangles):
    ptp_times = 0
    scan_times = calculate_scan_time()
    print(f"\n[Scan Time Per FOV] = {scan_times:.5f} seconds\n")

    for i in range(1, len(rectangles)):
        prev = (rectangles[i - 1]['cx'], rectangles[i - 1]['cy'])
        curr = (rectangles[i]['cx'], rectangles[i]['cy'])
        sync = synchronize_multi_axis_motion(prev, curr)
        max_dist = max(sync['x']['dist'], sync['y']['dist'])
        plot_velocity_profile(max_dist)

        segment_time = sync['t_sync'] + scan_times

        print(f"[FOV {i}] Move from {prev} to {curr}")
        print(f"         Travel   = {sync['t_sync']:.5f} s")
        print(f"         Scan     = {scan_times:.5f} s")
        print(f"         Segment  = {segment_time:.5f} s\n")

        ptp_times += segment_time

    print(f"Total Movement + Scan Time = {ptp_times:.3f} seconds")
    return ptp_times

def plot_velocity_profile(distance):
    t_total, profile = calculate_travel_time(distance, VELOCITY, ACCELERATION, JERK)
    t = profile['t']
    v = profile['v']
    plt.figure(figsize=(8, 4))
    plt.plot(t, v)
    plt.title(f"Velocity-Time Profile (distance={distance:.1e} nm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (nm/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- TESTING ---

mock_rectangles = [ 
    {'cx': 0, 'cy': 0, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 100_000_000, 'cy': 0, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 100_000_000, 'cy': 100_000_000, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 0, 'cy': 100_000_000, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 0, 'cy': 0, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000}
    ]


if __name__ == "__main__":
    print("\n--- Triangular Ramp ---")
    calculate_triangular_ramp_up_down_time(VELOCITY, JERK)

    JERK, ACCELERATION, VELOCITY = calculate_maximum_velocity(R, CYCLE_TIME)
    print(f"Using: JERK={JERK:.2e}, ACCELERATION={ACCELERATION:.2e}, VELOCITY={VELOCITY:.2e}")

    print("\n--- S-Curve Ramp ---")
    calculate_ramp_up_down_time(VELOCITY, ACCELERATION, JERK)

    print("\n--- Triangular Ramp ---")
    calculate_triangular_ramp_up_down_time(VELOCITY, JERK)

    calculate_board_movement_time(mock_rectangles)