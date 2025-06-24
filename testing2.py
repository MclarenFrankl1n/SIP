import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import pandas as pd
from scipy.interpolate import interp1d

# PARAMETERS
PI = np.pi
ACCELERATION = 5      # m/s²  
VELOCITY = 1          # m/s
JERK = 100            # m/s³
RADIUS = 0.10577      # meters
EXPO = 0.05
PROJ = 32
CycleTime = EXPO * PROJ
parabolic_ratio = 2/3
dt = 0.001
num_rev = 1.0

def calculate_maximum_velocity(radius, CycleTime, jerk, accel, velocity):
    """
    Calculate the maximum velocity, jerk, and acceleration for a circular motion profile based on the radius and cycle time.

    Args:
        radius (float): Radius of the circular path in meters.
        CycleTime (float): Total cycle time in seconds.
        jerk (float): Jerk value in m/s³.
        accel (float): Acceleration value in m/s².
        velocity (float): Desired velocity in m/s.

    Returns:
        tuple: Scaled jerk, acceleration, and maximum velocity.

    """
    V_max = 2 * radius * PI / CycleTime  # m/s
    print(f"Maximum Velocity = {V_max:.6f} m/s")
    if velocity == 0:
        return jerk, accel, velocity
    scale = V_max / velocity
    J = jerk * scale**3
    A = accel * scale**2
    V = V_max
    print(f"Jerk = {J:.6f} m/s³, Acceleration = {A:.6f} m/s², Velocity = {V:.6f} m/s")
    return J, A, V

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
    print(f"s_ramp (distance during ramp): {s_ramp/1e6:.1f} mm")
    return 4 * t_j, s_ramp  # Total ramp time (jerk up + down)

def calculate_bang_bang_time(v_target, a_max):
    """
    Bang-bang (instantaneous acceleration) profile.
    Time to reach v_target: t = v_target / a_max
    Total up+down time: 2t
    """
    t = v_target / a_max
    print(f"[Bang-bang] Time to reach v_target: {t:.5f} s")
    print(f"[Bang-bang] Total up+down time: {2*t:.5f} s")
    return 2*t

def calculate_trapezoidal_time(v_target, a_max, j_max):
    """
    Trapezoidal (acceleration-limited, optionally jerk-limited) profile.
    If jerk is not limiting, time to reach v_target: t = v_target / a_max (same as bang-bang).
    If jerk is limiting, use jerk-limited profile.
    """
    # Check if jerk is limiting
    t_j = a_max / j_max
    v_j = 0.5 * j_max * t_j**2
    if v_target <= 2 * v_j:
        # Jerk-limited triangular
        t = (2 * v_target / j_max) ** 0.5
        print(f"[Trapezoidal] Jerk-limited (triangular): t_j={t:.5f} s, total={4*t:.5f} s")
        return 4*t
    else:
        # Trapezoidal: jerk up, accel, jerk down
        t_accel = (v_target - v_j) / a_max
        total_time = 2 * (t_j + t_accel)
        print(f"[Trapezoidal] t_j={t_j:.5f} s, t_accel={t_accel:.5f} s, total={total_time:.5f} s")
        return total_time

def calculate_scan_time(radius, verbose=False, return_profile=False, plot_xy=False):
    """
    Calculate the scan time for a circular motion profile based on the radius and predefined parameters.

    This function computes the time required to complete a circular scan with a parabolic acceleration profile,
    including ramp-up, cruise, and ramp-down phases. It returns the total scan time and optionally the time and
    velocity profiles for each phase.

    Args:
        radius (float): Radius of the circular path in meters.

    Returns:
        float: Total scan time in seconds.
        (optional) tuple: Time and velocity profiles for each phase if return_profile is True.
        
    """

    peak_vel = VELOCITY_S
    peak_accel = ACCELERATION
    parabolic_ratio = 2/3

    ramp_time = peak_vel / (peak_accel * parabolic_ratio)
    half_ramp_time = ramp_time / 2
    b = peak_accel / (half_ramp_time - (half_ramp_time**2) / ramp_time)
    a = -b / ramp_time

    ramp_distance = ((peak_vel ** 2) / (2 * peak_accel * parabolic_ratio))

    print(f"Ramp time = {ramp_time:.5f} s")
    print(f"Total ramp time (up + down) = {2 * ramp_time:.5f} s")
  

    cruise_distance = 2 * np.pi * radius * num_rev
    print(f"Cruise distance = {cruise_distance:.5f} m")
    cruise_time = cruise_distance / peak_vel
    print(f"Total scan time (ramp + cruise + ramp) = {2 * ramp_time + cruise_time:.5f} s")

    t_up = np.arange(0, ramp_time, dt)
    t_cruise = np.arange(0, cruise_time, dt)
    t_down = np.arange(0, ramp_time, dt)

    # --- Ramp-up phase ---
    accel_up = a * t_up**2 + b * t_up
    vel_up = np.zeros_like(t_up)
    pos_up = np.zeros_like(t_up)
    for i in range(1, len(t_up)):
        vel_up[i] = vel_up[i-1] + accel_up[i-1] * dt
        pos_up[i] = pos_up[i-1] + vel_up[i-1] * dt
    jerk_up = np.zeros_like(t_up)
    jerk_up[1:] = (accel_up[1:] - accel_up[:-1]) / dt

    # Cruise position
    accel_cruise = np.zeros_like(t_cruise)
    vel_cruise = np.ones_like(t_cruise) * vel_up[-1]
    pos_cruise = np.zeros_like(t_cruise)
    pos_cruise[0] = pos_up[-1]
    for i in range(1, len(t_cruise)):
        pos_cruise[i] = pos_cruise[i-1] + vel_cruise[i-1] * dt
    jerk_cruise = np.zeros_like(t_cruise)

        # --- Ramp-down phase (mirror of ramp-up) ---
    accel_down = -accel_up[::-1]
    vel_down = np.zeros_like(t_down)
    pos_down = np.zeros_like(t_down)
    vel_down[0] = vel_cruise[-1]
    pos_down[0] = pos_cruise[-1]
    for i in range(1, len(t_down)):
        vel_down[i] = vel_down[i-1] + accel_down[i-1] * dt
        pos_down[i] = pos_down[i-1] + vel_down[i-1] * dt
    jerk_down = np.zeros_like(t_down)
    jerk_down[1:] = (accel_down[1:] - accel_down[:-1]) / dt

    # Concatenate
    pos = np.concatenate([pos_up[:-1], pos_cruise[:-1], pos_down])

    # Concatenate
    t = np.concatenate([
        t_up[:-1],
        t_cruise[:-1] + t_up[-1],
        t_down + t_up[-1] + t_cruise[-1]
    ])
    vel = np.concatenate([vel_up[:-1], vel_cruise[:-1], vel_down])
    total_time = t[-1]

    accel = np.concatenate([accel_up[:-1], accel_cruise[:-1], accel_down])
    vel = np.concatenate([vel_up[:-1], vel_cruise[:-1], vel_down])
    pos = np.concatenate([pos_up[:-1], pos_cruise[:-1], pos_down])
    jerk = np.concatenate([jerk_up[:-1], jerk_cruise[:-1], jerk_down])

    if plot_xy:
        degrees = np.degrees(pos / radius)
        x_pos = radius * np.cos(np.radians(degrees))
        y_pos = radius * np.sin(np.radians(degrees))
        x_vel = vel * np.sin(np.radians(degrees))
        y_vel = vel * np.cos(np.radians(degrees))
        dt_local = t[1] - t[0]
        x_accel = np.zeros_like(x_vel)
        x_accel[1:] = (x_vel[1:] - x_vel[:-1]) / dt_local
        x_accel[0] = x_accel[1]
        y_accel = np.zeros_like(y_vel)
        y_accel[1:] = (y_vel[1:] - y_vel[:-1]) / dt_local
        y_accel[0] = y_accel[1]

        t_rampup_end = t_up[-1]
        t_cruise_end = t_rampup_end + t_cruise[-1]
        t_rampdown_end = t_cruise_end + t_down[-1]

        plt.figure(figsize=(12, 8))
        plt.axvspan(0, t_rampup_end, color='#cce5ff', alpha=0.5, label='Ramp-up')
        plt.axvspan(t_rampup_end, t_cruise_end, color='#d4edda', alpha=0.5, label='Cruise')
        plt.axvspan(t_cruise_end, t_rampdown_end, color='#f8d7da', alpha=0.5, label='Ramp-down')
        plt.plot(t, x_pos, label="X Position (m)", color='blue')
        plt.plot(t, y_pos, label="Y Position (m)", color='red')
        plt.plot(t, x_vel, label="X Velocity (m/s)", color='cyan', linestyle='--')
        plt.plot(t, y_vel, label="Y Velocity (m/s)", color='magenta', linestyle='--')
        # plt.plot(t, x_accel, label="X Acceleration (m/s²)", color='green', linestyle=':')
        # plt.plot(t, y_accel, label="Y Acceleration (m/s²)", color='orange', linestyle=':')
        plt.title("X/Y Position, Velocity, and Acceleration vs Time (Scan Phase)")
        plt.xlabel("Time (s)")
        plt.ylabel("Value (SI Units)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

                # --- Circular (tangential) motion plot ---
        plt.figure(figsize=(10, 6))
        plt.plot(t, accel, label="Circular Accel", color='blue')
        plt.plot(t, vel, label="Circular Vel", color='red')
        plt.plot(t, pos, label="Circular Pos", color='orange')
        plt.plot(t, jerk, label="Circular Jerk", color='green')
        plt.title("Circular Position, Velocity, Acceleration & Jerk")
        plt.xlabel("Time (s)")
        plt.ylabel("Value (SI Units)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if verbose:
        print(f"\n[Scan Time Per FOV] = {total_time:.5f} seconds\n")



    if return_profile:
        return total_time, t, vel, accel, pos
    return total_time

def calculate_travel_time(distance, v_max, a_max, j_max, resolution=1000, verbose=False):
    """
    Calculate the travel time for a motion profile based on distance, maximum velocity, acceleration, and jerk.

    This function computes the time required to travel a specified distance either using a triangular or S-curve motion profile.
    It returns the total travel time and optionally the time and velocity profiles for each phase.

    Args:
        distance (float): Distance to travel in meters.
        v_max (float): Maximum velocity in m/s.
        a_max (float): Maximum acceleration in m/s².
        j_max (float): Maximum jerk in m/s³.
        resolution (int): Number of points in the time profile.
        verbose (bool): If True, prints detailed information about the calculation.

    Returns:
        tuple: Total travel time in seconds and a dictionary containing the motion profile type, time values, velocity values, and total time.
        
    """
    t_j = a_max / j_max  # time to reach max acceleration
    t_const_a = (v_max - a_max * t_j) / a_max  # time at constant acceleration
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
        if verbose:
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
        if verbose:
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

def synchronize_multi_axis_motion(p1, p2):
    """
    Calculate the synchronization parameters for multi-axis motion between two points.

    This function computes the time required for each axis to travel from position p1 to p2,
    and determines the synchronization axis (the axis that takes the longest time).
    It returns a dictionary with the synchronization time and parameters for each axis.

    Args:
        p1 (tuple/list): Starting positions of the axes.
        p2 (tuple/list): Ending positions of the axes.

    Returns:
        dict: A dictionary containing the synchronization time, the sync axis, and parameters for each axis.
        The parameters include distance, jerk, acceleration, and velocity for each axis.
        
    """

    n_axes = len(p1)
    distances = [abs(p2[i] - p1[i]) for i in range(n_axes)]
    times = []
    for dist in distances:
        t,_ = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK)
        times.append(t)
    t_sync = max(times)
    sync_axis_idx = times.index(t_sync)
    sync_axis = f"Axis{sync_axis_idx}"

    axis_labels = [f"Axis{i}" for i in range(n_axes)]
    print(f"Sync axis: {sync_axis} ({', '.join([f't_{axis_labels[i]}={t:.5f} s' for i, t in enumerate(times)])})")

    params = []
    for t_i in times:
        if t_i == 0:
            params.append((JERK, ACCELERATION, VELOCITY))
        else:
            scale = t_i / t_sync if t_sync > 0 else 1
            j = JERK * scale**3
            a = ACCELERATION * scale**2
            v = VELOCITY * scale
            params.append((j, a, v))
    result = {
        't_sync': t_sync,
        'sync_axis': sync_axis,
    }
    for i in range(n_axes):
        result[f"axis_{i}"] = {
            'dist': distances[i],
            'jerk': params[i][0],
            'accel': params[i][1],
            'vel': params[i][2],
        }
    return result

def calculate_board_movement_time(rectangles):
    """
    Calculate the total movement and scan time for a sequence of FOV rectangles.

    Args:
        rectangles (list[dict]): List of FOV rectangles with coordinates in meters.

    Returns:
        tuple: (total_time, segment_times) where total_time is the sum of all segments,
               and segment_times is a list of (move_time, scan_time) tuples.
    """
    ptp_times = 0
    radius = RADIUS
    segment_times = []
    scan_time, t_scan, v_scan, a_scan, pos_scan = calculate_scan_time(RADIUS, return_profile=True, plot_xy=True)

    for i in range(1, len(rectangles)):
        axis_keys = [k for k in rectangles[i] if k.startswith('c')]
        prev = tuple(rectangles[i - 1][k] for k in axis_keys)
        curr = tuple(rectangles[i][k] for k in axis_keys)
        sync = synchronize_multi_axis_motion(prev, curr)
        segment_time = sync['t_sync'] + scan_time
        print(f"[FOV {i}] Move from {prev} to {curr}")
        print(f"         Travel   = {sync['t_sync']:.5f} s")
        print(f"         Scan     = {scan_time:.5f} s")
        print(f"         Segment  = {segment_time:.5f} s\n")
        ptp_times += segment_time
        segment_times.append((sync['t_sync'], scan_time))
    print(f"Total Movement + Scan Time = {ptp_times:.3f} seconds")
    return ptp_times, segment_times

def load_FOV_from_csv(filename):
    rectangles = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert all values from nm to m
            rect = {k: float(v) / 1e9 for k, v in row.items()}
            rectangles.append(rect)
    return rectangles

def plot_motion_profile(rectangles):
    t_full = []
    pos_full = []
    v_full = []
    a_full = []
    j_full = []
    t_offset = 0
    axis_keys = [k for k in rectangles[0] if k.startswith('c')]
    pos_offset = rectangles[0][axis_keys[0]]  # Start at the first FOV center (sync axis)
    phase_spans = []

    for i in range(1, len(rectangles)):
        axis_keys = [k for k in rectangles[i] if k.startswith('c')]
        prev = np.array([rectangles[i - 1][k] for k in axis_keys])
        curr = np.array([rectangles[i][k] for k in axis_keys])
        delta = curr - prev

        sync = synchronize_multi_axis_motion(prev, curr)
        sync_axis = int(sync['sync_axis'].replace('Axis', ''))
        params = sync[f'axis_{sync_axis}']
        move_dist = params['dist']
        move_jerk = params['jerk']
        move_accel = params['accel']
        move_vel = params['vel']

        direction = np.sign(delta[sync_axis])

        # ...inside your move phase loop in plot_motion_profile...
        t_total_move, move_profile = calculate_travel_time(
            move_dist, move_vel, move_accel, move_jerk
        )
        t_move = move_profile['t'] + t_offset
        v_move = move_profile['v'] * direction

        # Integrate velocity for position (sync axis only)
        pos_move = np.zeros_like(t_move)
        pos_move[0] = pos_offset
        for j in range(1, len(t_move)):
            pos_move[j] = pos_move[j-1] + v_move[j-1] * (t_move[j] - t_move[j-1])

        pos_move = np.maximum(pos_move, 0)

        a_move = np.zeros_like(v_move)
        a_move[1:] = np.diff(v_move) / np.diff(t_move)
        j_move = np.zeros_like(a_move)
        j_move[1:] = np.diff(a_move) / np.diff(t_move)

        t_full.append(t_move)
        pos_full.append(pos_move)
        v_full.append(v_move)
        a_full.append(a_move)
        j_full.append(j_move)
        phase_spans.append((t_move[0], t_move[-1], 'move'))

        t_offset = t_move[-1]
        pos_offset = pos_move[-1]  # Reset to FOV center for next move

        # scan_time, t_scan, v_scan, a_scan, pos_scan = calculate_scan_time(RADIUS, return_profile=True)
        # t_scan = t_scan + t_offset
        # theta = pos_scan / RADIUS

        # if sync_axis == 0:
        #     pos_scan_sync = curr[0] + RADIUS * np.cos(theta)
        #     v_scan_sync   = -v_scan * np.sin(theta)
        #     a_scan_sync   = -a_scan * np.sin(theta)
        # elif sync_axis == 1:
        #     pos_scan_sync = curr[1] + RADIUS * np.sin(theta)
        #     v_scan_sync   =  v_scan * np.cos(theta)
        #     a_scan_sync   =  a_scan * np.cos(theta)
        # else:
        #     # For more axes, generalize as needed
        #     pos_scan_sync = np.ones_like(t_scan) * curr[sync_axis]
        #     v_scan_sync = np.zeros_like(t_scan)
        #     a_scan_sync = np.zeros_like(t_scan)

        # pos_scan_sync = np.maximum(pos_scan_sync, 0)

        # j_scan_sync = np.zeros_like(a_scan_sync)
        # j_scan_sync[1:] = np.diff(a_scan_sync) / np.diff(t_scan)

        # t_full.append(t_scan)
        # pos_full.append(pos_scan_sync)
        # v_full.append(v_scan_sync)
        # a_full.append(a_scan_sync)
        # j_full.append(j_scan_sync)
        # phase_spans.append((t_scan[0], t_scan[-1], 'scan'))

        # t_offset = t_scan[-1]
        # pos_offset = curr[sync_axis]  # Reset to FOV center for next move

    t_full = np.concatenate(t_full)
    pos_full = np.concatenate(pos_full)
    v_full = np.concatenate(v_full)
    a_full = np.concatenate(a_full)
    j_full = np.concatenate(j_full)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for start, end, phase in phase_spans:
        color = '#ffeeba' if phase == 'move' else '#d4edda'
        label = 'Move' if phase == 'move' else 'Scan'
        for ax in axs.flat:
            ax.axvspan(start, end, color=color, alpha=0.4, label=label if ax==axs[0,0] else "")

    axs[0, 0].plot(t_full, pos_full, color='orange', label='Sync Axis Position')
    axs[0, 0].set_title("Position (m)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(t_full, v_full, color='red')
    axs[0, 1].set_title("Velocity (m/s)")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].grid(True)

    axs[1, 0].plot(t_full, a_full, color='blue')
    axs[1, 0].set_title("Acceleration (m/s²)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Acceleration (m/s²)")
    axs[1, 0].grid(True)

    axs[1, 1].plot(t_full, j_full, color='green')
    axs[1, 1].set_title("Jerk (m/s³)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Jerk (m/s³)")
    axs[1, 1].grid(True)

    handles, labels = axs[0,0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0,0].legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_xy_motion_profile(rectangles, args):
    BUFFER_TIME = 0  # seconds

    t_full = []
    vx_full = []
    vy_full = []
    ax_full = []
    ay_full = []
    t_offset = 0
    phase_spans = []

    axis_keys = [k for k in rectangles[0] if k.startswith('c')]
    curr_pos = np.array([rectangles[0][axis_keys[0]], rectangles[0][axis_keys[1]]])
    x_full = []
    y_full = []

    for i in range(1, len(rectangles)):
        prev = np.array([rectangles[i - 1][k] for k in axis_keys])
        curr = np.array([rectangles[i][k] for k in axis_keys])
        delta = curr - prev

        vector_dist = np.linalg.norm(delta[:2])
        if vector_dist == 0:
            continue  # No move

        # # Generate velocity profile for the move
        # t_total_move, move_profile = calculate_travel_time(
        #     vector_dist, VELOCITY, ACCELERATION, JERK
        # )
        # t_move = move_profile['t'] + t_offset
        # v_profile = move_profile['v']

        # # --- SCALE velocity profile so integrated distance matches vector_dist ---
        # dt = np.diff(t_move, prepend=t_move[0])
        # dist_integrated = np.sum(v_profile * dt)
        # scale = vector_dist / dist_integrated if dist_integrated != 0 else 1.0
        # v_profile_scaled = v_profile * scale

        # # Project velocity onto axes
        # direction = delta / vector_dist
        # v_axes = np.outer(v_profile_scaled, direction)

        # # Integrate for position
        # x = np.zeros_like(t_move)
        # y = np.zeros_like(t_move)
        # x[0] = curr_pos[0]
        # y[0] = curr_pos[1]
        # for j in range(1, len(t_move)):
        #     dt_j = t_move[j] - t_move[j-1]
        #     x[j] = x[j-1] + v_axes[j-1, 0] * dt_j
        #     y[j] = y[j-1] + v_axes[j-1, 1] * dt_j


        # ax = np.zeros_like(t_move)
        # ay = np.zeros_like(t_move)
        # ax[1:] = np.diff(v_axes[:, 0]) / np.diff(t_move)
        # ay[1:] = np.diff(v_axes[:, 1]) / np.diff(t_move)

        # t_full.append(t_move)
        # x_full.append(x)
        # y_full.append(y)
        # vx_full.append(v_axes[:, 0])
        # vy_full.append(v_axes[:, 1])
        # ax_full.append(ax)
        # ay_full.append(ay)
        # phase_spans.append((t_move[0], t_move[-1], 'move'))

        # # Buffer after move
        # dt_buf = t_move[1] - t_move[0] if len(t_move) > 1 else 0.001
        # t_buf = np.arange(t_move[-1] + dt_buf, t_move[-1] + BUFFER_TIME + dt_buf, dt_buf)
        # if len(t_buf) > 0:
        #     t_full.append(t_buf)
        #     x_full.append(np.ones_like(t_buf) * x[-1])
        #     y_full.append(np.ones_like(t_buf) * y[-1])
        #     vx_full.append(np.zeros_like(t_buf))
        #     vy_full.append(np.zeros_like(t_buf))
        #     ax_full.append(np.zeros_like(t_buf))
        #     ay_full.append(np.zeros_like(t_buf))
        #     phase_spans.append((t_buf[0], t_buf[-1], 'buffer'))
        #     t_offset = t_buf[-1]
        # else:
        #     t_offset = t_move[-1]
        # # Always use the last value of the buffer for curr_pos
        # if len(t_buf) > 0:
        #     curr_pos = np.array([x_full[-1][-1], y_full[-1][-1]])
        # else:
        #     curr_pos = np.array([x[-1], y[-1]])

        # --- Scan phase after buffer ---
        scan_time, t_scan, v_scan, a_scan, pos_scan = calculate_scan_time(RADIUS, return_profile=True)
        t_scan = t_scan + t_offset

        # Circular arc: theta in radians
        theta = pos_scan / RADIUS
        x_scan = curr[0] + RADIUS * np.cos(theta)
        y_scan = curr[1] + RADIUS * np.sin(theta)
        vx_scan = -v_scan * np.sin(theta)
        vy_scan = v_scan * np.cos(theta)
        ax_scan = -a_scan * np.sin(theta)
        ay_scan = a_scan * np.cos(theta)

        t_full.append(t_scan)
        x_full.append(x_scan)
        y_full.append(y_scan)
        vx_full.append(vx_scan)
        vy_full.append(vy_scan)
        ax_full.append(ax_scan)
        ay_full.append(ay_scan)
        phase_spans.append((t_scan[0], t_scan[-1], 'scan'))

        t_offset = t_scan[-1]
        # End of scan is back at FOV center
        curr_pos = np.array([curr[0], curr[1]])

        # --- Buffer after scan phase ---
        dt_buf = t_scan[1] - t_scan[0] if len(t_scan) > 1 else 0.001
        t_buf = np.arange(t_scan[-1] + dt_buf, t_scan[-1] + BUFFER_TIME + dt_buf, dt_buf)
        if len(t_buf) > 0:
            t_full.append(t_buf)
            x_full.append(np.ones_like(t_buf) * curr_pos[0])
            y_full.append(np.ones_like(t_buf) * curr_pos[1])
            vx_full.append(np.zeros_like(t_buf))
            vy_full.append(np.zeros_like(t_buf))
            ax_full.append(np.zeros_like(t_buf))
            ay_full.append(np.zeros_like(t_buf))
            phase_spans.append((t_buf[0], t_buf[-1], 'buffer'))
            t_offset = t_buf[-1]
        else:
            t_offset = t_scan[-1]


    t_full = np.concatenate(t_full)
    x_full = np.concatenate(x_full)
    y_full = np.concatenate(y_full)
    vx_full = np.concatenate(vx_full)
    vy_full = np.concatenate(vy_full)
    ax_full = np.concatenate(ax_full)
    ay_full = np.concatenate(ay_full)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for start, end, phase in phase_spans:
        color = '#ffeeba' if phase == 'move' else ('#d4edda' if phase == 'scan' else '#eeeeee')
        label = 'Move' if phase == 'move' else ('Scan' if phase == 'scan' else 'Buffer')
        for ax in axs.flat:
            ax.axvspan(start, end, color=color, alpha=0.4, label=label if ax==axs[0,0] else "")

    # Position subplot
    axs[0, 0].plot(t_full, x_full, color='blue', label='X Position')
    axs[0, 0].plot(t_full, y_full, color='red', label='Y Position')
    axs[0, 0].set_title("X/Y Position (m)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].grid(True)

    # Velocity subplot
    axs[0, 1].plot(t_full, vx_full, color='blue', label='X Velocity')
    axs[0, 1].plot(t_full, vy_full, color='red', label='Y Velocity')
    axs[0, 1].set_title("X/Y Velocity (m/s)")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].grid(True)

    # Acceleration subplot
    axs[1, 0].plot(t_full, ax_full, color='blue', label='X Accel')
    axs[1, 0].plot(t_full, ay_full, color='red', label='Y Accel')
    axs[1, 0].set_title("X/Y Acceleration (m/s²)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Acceleration (m/s²)")
    axs[1, 0].grid(True)

    # Empty subplot (or you can add something else)
    axs[1, 1].axis('off')

    # Legends
    for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()

    # --- Verification with measured Excel data ---

    if args.measured:
        measured_df = pd.read_csv(args.measured, header=2)
        start = args.cycle_start
        end = args.cycle_end if args.cycle_end is not None else None
        measured_time = measured_df['Cycle'].values[start:end] / 1000.0  # ms to s
        measured_x = measured_df['CommandPos-0'].values[start:end] / 1e9
        measured_y = measured_df['CommandPos-1'].values[start:end] / 1e9
        measured_vx = measured_df['CommandVelocity-0'].values[start:end] / 1e9
        measured_vy = measured_df['CommandVelocity-1'].values[start:end] / 1e9

        # Align measured time to start at the same point as simulated scan
        measured_time_aligned = measured_time - measured_time[0] + t_full[0]

        plt.figure(figsize=(12, 5))

        # --- Overlay X/Y Position ---
        plt.subplot(1, 2, 1)
        plt.plot(t_full, x_full, label='Simulated X', color='blue', linestyle='--')
        plt.plot(t_full, y_full, label='Simulated Y', color='green', linestyle='--')
        plt.plot(measured_time_aligned, measured_x, label='Measured X', color='red')
        plt.plot(measured_time_aligned, measured_y, label='Measured Y', color='orange')
        plt.title('X/Y Position (Measured & Simulated)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid(True)

        # --- Overlay X/Y Velocity ---
        plt.subplot(1, 2, 2)
        plt.plot(t_full, vx_full, label='Simulated VX', color='blue', linestyle='--')
        plt.plot(t_full, vy_full, label='Simulated VY', color='green', linestyle='--')
        plt.plot(measured_time_aligned, measured_vx, label='Measured VX', color='red')
        plt.plot(measured_time_aligned, measured_vy, label='Measured VY', color='orange')
        plt.title('X/Y Velocity (Measured & Simulated)')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # --- Calculate and plot percentage differences ---

        # Interpolate simulated data onto measured time base
        interp_x = interp1d(t_full, x_full, bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d(t_full, y_full, bounds_error=False, fill_value="extrapolate")
        interp_vx = interp1d(t_full, vx_full, bounds_error=False, fill_value="extrapolate")
        interp_vy = interp1d(t_full, vy_full, bounds_error=False, fill_value="extrapolate")

        sim_x_on_meas = interp_x(measured_time_aligned)
        sim_y_on_meas = interp_y(measured_time_aligned)
        sim_vx_on_meas = interp_vx(measured_time_aligned)
        sim_vy_on_meas = interp_vy(measured_time_aligned)

        # Avoid division by zero
        sim_x_on_meas[sim_x_on_meas == 0] = np.nan
        sim_y_on_meas[sim_y_on_meas == 0] = np.nan
        sim_vx_on_meas[sim_vx_on_meas == 0] = np.nan
        sim_vy_on_meas[sim_vy_on_meas == 0] = np.nan

        # Calculate percentage difference
        percent_diff_x = 100 * (measured_x - sim_x_on_meas) / sim_x_on_meas
        percent_diff_y = 100 * (measured_y - sim_y_on_meas) / sim_y_on_meas
        percent_diff_vx = 100 * (measured_vx - sim_vx_on_meas) / sim_vx_on_meas
        percent_diff_vy = 100 * (measured_vy - sim_vy_on_meas) / sim_vy_on_meas

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(measured_time_aligned, percent_diff_x, label='X Position % Diff', color='blue')
        plt.plot(measured_time_aligned, percent_diff_y, label='Y Position % Diff', color='green')
        plt.title('Percentage Difference: Position (Measured vs Simulated)')
        plt.xlabel('Time (s)')
        plt.ylabel('Percent Difference (%)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(measured_time_aligned, percent_diff_vx, label='X Velocity % Diff', color='red')
        plt.plot(measured_time_aligned, percent_diff_vy, label='Y Velocity % Diff', color='orange')
        plt.title('Percentage Difference: Velocity (Measured vs Simulated)')
        plt.xlabel('Time (s)')
        plt.ylabel('Percent Difference (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="CycleTime and Motion Profile Calculation for Circular Motion"
    )
    parser.add_argument('--acceleration', type=float, default=5, help='Acceleration (m/s^2)')
    parser.add_argument('--velocity', type=float, default=1, help='Velocity (m/s)')
    parser.add_argument('--jerk', type=float, default=100, help='Jerk (m/s^3)')
    parser.add_argument('--radius', type=float, default=0.10577, help='Radius (m)')
    parser.add_argument('--expo', type=float, default=0.05, help='Exposure time (s)')
    parser.add_argument('--proj', type=int, default=32, help='Number of projections')
    parser.add_argument('--fov', type=str, default='ScanTests.csv', help='FOV CSV file')
    parser.add_argument('--measured', type=str, default=None, help='Measured CSV file for verification')
    parser.add_argument('--cycle_start', type=int, default=0, help='Row index to start the cycle in measured data')
    parser.add_argument('--cycle_end', type=int, default=None, help='Row index to end the cycle in measured data (exclusive)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    JERK_S, ACCELERATION_S, VELOCITY_S = calculate_maximum_velocity(
        radius=args.radius, CycleTime=CycleTime, jerk=args.jerk, accel=args.acceleration, velocity=args.velocity
    )
    ACCELERATION = args.acceleration
    JERK = args.jerk
    VELOCITY = args.velocity
    RADIUS = args.radius


    # print("\n--- Bang-bang (acceleration-limited, no jerk) ---")
    # calculate_bang_bang_time(v_target, a_max)

    # print("\n--- Trapezoidal (acceleration-limited, with jerk) ---")
    # calculate_trapezoidal_time(v_target, a_max, j_max)

    # print("\n--- Jerk-limited (triangular) ---")
    # calculate_triangular_ramp_up_down_time(v_target, j_max)


    CycleTime = args.expo * args.proj
    rectangles = load_FOV_from_csv(args.fov)
    scan_time, t_scan, v_scan, a_scan, pos_scan = calculate_scan_time(RADIUS, return_profile=True, plot_xy=True)
    # total_time, segment_times = calculate_board_movement_time(rectangles)
    # plot_motion_profile(rectangles)
    plot_xy_motion_profile(rectangles, args)

    # v_target = 0.392699  # m/s
    # t_measured = 0.118  # s (total up+down)
    # t_j = t_measured / 4
    # j_measured = 2 * v_target / (t_j ** 2)
    # print(f"Effective jerk: {j_measured:.2f} m/s³")

    # v_target = 0.392699  # m/s
    # t_measured = 0.059  # s (half ramp, since up+down = 0.118s)
    # a_measured = v_target / t_measured
    # print(f"Effective acceleration: {a_measured:.2f} m/s²")