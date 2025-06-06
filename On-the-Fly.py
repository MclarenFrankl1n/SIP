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
import csv

#parabolic motion profile
# make sure everything is variable input
# Constants
ACCELERATION = 5_000_000_000  # nm/s²   for para x2/3 for average
VELOCITY = 1_000_000_000      # nm/s
JERK = 50_000_000_000          # nm/s³
PARABOLIC_RATIO = 2/3  # ratio of parabolic to average velocity
# Defaults for Source Detector
VELOCITY_S = 1_000_000      # 1 mm/s
ACCELERATION_S = 100_000_000  # 0.1 m/s²
JERK_S = 1_000_000_000        # 1 m/s³
PI = np.pi
PROJ = 32
EXPO = 0.05  # seconds   50ms
CYCLE_TIME = PROJ*EXPO  # seconds   10s

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
    print(f"Using: JERK={JERK_S:.2e}, ACCELERATION={ACCELERATION_S:.2e}, VELOCITY={VELOCITY_S:.2e}")
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
    print("\n--- S-Curve Ramp ---")
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
    print(f"s_ramp (distance during ramp): {s_ramp/1e6:.1f} mm")
    return 4 * t_j, s_ramp  # Total ramp time (jerk up + down)

# Scan time per FOV (for all projections)
def calculate_scan_time(radius, verbose=False):
    # Use updated velocity, acceleration, and jerk that match cycle time
    JERK_S, ACCELERATION_S, VELOCITY_S = calculate_maximum_velocity(radius, CYCLE_TIME)
    t_ramp, s_ramp = calculate_triangular_ramp_up_down_time(VELOCITY_S, JERK_S)
    revolution = 1 # Number of revolutions per projection
    scan_times = CYCLE_TIME*revolution + t_ramp
    if verbose:
        print(f"Total scan time for {PROJ} projections at {revolution} revolution: {scan_times:.5f} seconds")
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
    """
    Synchronize multi-axis motion for any number of axes.
    Returns the sync axis, sync time, and scaled parameters for each axis.
    """
    n_axes = len(p1)
    distances = [abs(p2[i] - p1[i]) for i in range(n_axes)]
    times = []
    params = []

    for dist in distances:
        t, _ = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK)
        times.append(t)

    t_sync = max(times)
    sync_axis_idx = times.index(t_sync)
    sync_axis = f"Axis{sync_axis_idx}"

    axis_labels = [f"Axis{i}" for i in range(n_axes)]
    print(f"Sync axis: {sync_axis} ({', '.join([f't_{axis_labels[i]}={t:.5f} s' for i, t in enumerate(times)])})")

    def scale_params(t_i):
        if t_i == 0:
            return JERK, ACCELERATION, VELOCITY  # no motion on this axis
        scale = (t_i / t_sync)
        j = JERK * scale**3
        a = ACCELERATION * scale**2
        v = VELOCITY * scale
        return j, a, v

    for t_i in times:
        params.append(scale_params(t_i))

    # Build result dictionary
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

# Updated board movement timing with multi-axis synchronization
def calculate_board_movement_time(rectangles):
    ptp_times = 0
    radius = 105_770_000
    scan_times = calculate_scan_time(radius, verbose=True)
    print(f"\n[Scan Time Per FOV] = {scan_times:.5f} seconds\n")

    for i in range(1, len(rectangles)):
        axis_keys = [k for k in rectangles[i] if k.startswith('c')]
        prev = tuple(rectangles[i - 1][k] for k in axis_keys)
        curr = tuple(rectangles[i][k] for k in axis_keys)
        sync = synchronize_multi_axis_motion(prev, curr)

        # Plot velocity profile for the greatest distance of all axes for board movement
        max_dist = max(sync[f'axis_{j}']['dist'] for j in range(len(prev)))
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


def plot_full_motion_profile(rectangles, scan_times, time_range=None, save_prefix='full_motion'):
    """
    Plot the full motion profile (position, velocity, acceleration, jerk)
    for the entire sequence of FOV moves and scans.
    Shows 4 subplots and 1 normalized overlay plot.
    """
    t_all = []
    s_all = []
    v_all = []
    a_all = []
    j_all = []
    phase_types = []  # 'move' or 'scan'
    t_offset = 0
    s_offset = 0

    for i in range(1, len(rectangles)):
        axis_keys = [k for k in rectangles[i] if k.startswith('c')]
        prev = tuple(rectangles[i - 1][k] for k in axis_keys)
        curr = tuple(rectangles[i][k] for k in axis_keys)
        sync = synchronize_multi_axis_motion(prev, curr)
        max_dist = max(sync[f'axis_{j}']['dist'] for j in range(len(prev)))
        t_total, profile = calculate_travel_time(max_dist, VELOCITY, ACCELERATION, JERK)
        t = profile['t'] + t_offset
        v = profile['v']
        s = np.cumsum(v) * (t[1] - t[0]) + s_offset
        a = np.gradient(v, t)
        j_ = np.gradient(a, t)

        t_all.append(t)
        s_all.append(s)
        v_all.append(v)
        a_all.append(a)
        j_all.append(j_)
        phase_types.append(('move', t[0], t[-1]))

        t_offset = t[-1]
        s_offset = s[-1]

        # Use the correct scan velocity, acceleration, and jerk for scan phase
        JERK_S, ACCELERATION_S, VELOCITY_S = calculate_maximum_velocity(radius=105_770_000, CycleTime=CYCLE_TIME)
        distance_scan = VELOCITY_S * scan_times
        t_scan_rel, profile_scan = calculate_travel_time(distance_scan, VELOCITY_S, ACCELERATION_S, JERK_S)
        t_scan = profile_scan['t'] + t_offset
        v_scan = profile_scan['v']
        s_scan = np.cumsum(v_scan) * (t_scan[1] - t_scan[0]) + s_offset
        a_scan = np.gradient(v_scan, t_scan)
        j_scan = np.gradient(a_scan, t_scan)

        t_all.append(t_scan)
        s_all.append(s_scan)
        v_all.append(v_scan)
        a_all.append(a_scan)
        j_all.append(j_scan)
        phase_types.append(('scan', t_scan[0], t_scan[-1]))

        t_offset = t_scan[-1]
        s_offset = s_scan[-1]

    # Concatenate all segments
    t_full = np.concatenate(t_all)
    s_full = np.concatenate(s_all)
    v_full = np.concatenate(v_all)
    a_full = np.concatenate(a_all)
    j_full = np.concatenate(j_all)

    # 4 subplots with color grading for move/scan phases (background shading)
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    labels = ["Position (nm)", "Velocity (nm/s)", "Acceleration (nm/s²)", "Jerk (nm/s³)"]
    datas = [s_full, v_full, a_full, j_full]
    colors = {'move': 'cyan', 'scan': 'yellow'}
    linecolors = {'move': 'blue', 'scan': 'orange'}

    # Plot the full profile on each subplot
    for idx, ax in enumerate(axs):
        ax.plot(t_full, datas[idx], color=colors['move'] if idx == 0 else linecolors['move'], label=labels[idx])
        # Shade move and scan phases
        for phase, t_start, t_end in phase_types:
            if phase == 'scan':
                ax.axvspan(t_start, t_end, color='yellow', alpha=0.2, label='Scan Phase' if idx == 0 else "")
            else:
                ax.axvspan(t_start, t_end, color='cyan', alpha=0.1, label='Move Phase' if idx == 0 else "")
        ax.set_ylabel(labels[idx])
        ax.grid(True)
        if idx == 0:
            handles, phase_labels = ax.get_legend_handles_labels()
            # Only add phase legend once
            ax.legend(handles, phase_labels, loc='upper right')

    axs[3].set_xlabel("Time (s)")
    if time_range is not None:
        axs[3].set_xlim(time_range)
    plt.suptitle("Full Motion Profile (All FOVs) — Move (cyan), Scan (yellow)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{save_prefix}_subplots.png")
    plt.show()

    # Normalized overlay plot
    def normalize(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val == 0:
            return arr
        return (arr - min_val) / (max_val - min_val)

    plt.figure(figsize=(14, 4))
    plt.plot(t_full, normalize(s_full), label="Position (norm)", color='blue')
    plt.plot(t_full, normalize(v_full), label="Velocity (norm)", color='orange')
    plt.plot(t_full, normalize(a_full), label="Acceleration (norm)", color='green')
    plt.plot(t_full, normalize(j_full), label="Jerk (norm)", color='red')
    plt.xlabel("Time (s)")
    plt.title("Normalized Full Motion Profile (All FOVs)")
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    # Color the background for move and scan phases
    added_labels = set()
    for phase, t_start, t_end in phase_types:
        if phase == 'scan':
            label = 'Scan Phase' if 'Scan Phase' not in added_labels else None
            ax.axvspan(t_start, t_end, color='yellow', alpha=0.2, label=label)
            added_labels.add('Scan Phase')
        else:
            label = 'Move Phase' if 'Move Phase' not in added_labels else None
            ax.axvspan(t_start, t_end, color='cyan', alpha=0.1, label=label)
            added_labels.add('Move Phase')
    if time_range is not None:
        plt.xlim(time_range)
    else:
        plt.xlim([0, t_full[-1]])
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_normalized.png")  # Save the normalized overlay
    plt.show()

def plot_scan_phase(scan_time, velocity=VELOCITY_S, acceleration=ACCELERATION_S, jerk=JERK_S, time_range=None, save_prefix='scan_phase'):
    """
    Plot position, velocity, acceleration, and jerk for a scan phase
    including ramp-up, constant velocity, and ramp-down.
    Shows 4 subplots and 1 normalized overlay plot.
    """
    distance = velocity * scan_time
    t_total, profile = calculate_travel_time(distance, velocity, acceleration, jerk)
    t = profile['t']
    v = profile['v']
    s = np.cumsum(v) * (t[1] - t[0])
    a = np.gradient(v, t)
    j_ = np.gradient(a, t)

    # 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t, s, color='blue')
    axs[0].set_ylabel("Position (nm)")
    axs[0].grid(True)

    axs[1].plot(t, v, color='orange')
    axs[1].set_ylabel("Velocity (nm/s)")
    axs[1].grid(True)

    axs[2].plot(t, a, color='green')
    axs[2].set_ylabel("Acceleration (nm/s²)")
    axs[2].grid(True)

    axs[3].plot(t, j_, color='red')
    axs[3].set_ylabel("Jerk (nm/s³)")
    axs[3].set_xlabel("Time (s)")
    axs[3].grid(True)

    if time_range is not None:
        axs[3].set_xlim(time_range)
    plt.suptitle("Scan Phase Motion Profile (with Ramp Up/Down)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{save_prefix}_subplots.png")  # Save the 4 subplots
    plt.show()

    # Normalized overlay plot
    def normalize(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val == 0:
            return arr
        return (arr - min_val) / (max_val - min_val)

    plt.figure(figsize=(12, 4))
    plt.plot(t, normalize(s), label="Position (norm)", color='blue')
    plt.plot(t, normalize(v), label="Velocity (norm)", color='orange')
    plt.plot(t, normalize(a), label="Acceleration (norm)", color='green')
    plt.plot(t, normalize(j_), label="Jerk (norm)", color='red')
    plt.xlabel("Time (s)")
    plt.title("Normalized Scan Phase Motion Profile")
    plt.legend()
    plt.grid(True)
    if time_range is not None:
        plt.xlim(time_range)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_normalized.png")  # Save the normalized overlay
    plt.show()

# --- TESTING ---


def load_FOV_from_csv(filename):
    """
    Load rectangles (positions) from a CSV file.
    Each row should have columns like: cx, cy, cz, cw, ch, ...
    Returns a list of dictionaries.
    """
    rectangles = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert all values to int (or float if needed)
            rect = {k: int(v) for k, v in row.items()}
            rectangles.append(rect)
    return rectangles


if __name__ == "__main__":
    # print("\n--- Triangular Ramp ---")
    # calculate_triangular_ramp_up_down_time(VELOCITY, JERK)

    rectangles = load_FOV_from_csv('FOV.csv')
    JERK_S, ACCELERATION_S, VELOCITY_S = calculate_maximum_velocity(radius=105_770_000, CycleTime=CYCLE_TIME)
    calculate_board_movement_time(rectangles)
    scan_times = calculate_scan_time(radius=105_770_000, verbose=True)  # or whatever radius you use
    plot_full_motion_profile(rectangles, scan_times)  # View only the first 2 seconds
    scan_times = calculate_scan_time(radius=105_770_000, verbose=True)
    plot_scan_phase(scan_times, velocity=VELOCITY_S, acceleration=ACCELERATION_S, jerk=JERK_S)