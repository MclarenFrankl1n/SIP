import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

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
    peak_accel = ACCELERATION_S
    parabolic_ratio = 2/3

    ramp_time = peak_vel / (peak_accel * parabolic_ratio)
    half_ramp_time = ramp_time / 2
    b = peak_accel / (half_ramp_time - (half_ramp_time**2) / ramp_time)
    a = -b / ramp_time

    cruise_distance = 2 * np.pi * radius * num_rev
    cruise_time = cruise_distance / peak_vel

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
        plt.plot(t, x_accel, label="X Acceleration (m/s²)", color='green', linestyle=':')
        plt.plot(t, y_accel, label="Y Acceleration (m/s²)", color='orange', linestyle=':')
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
        return total_time, t, vel
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
    scan_time = calculate_scan_time(radius, verbose=True, plot_xy=True)

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
    pos_offset = 0
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

        # Get the sign for the sync axis
        direction = np.sign(delta[sync_axis])

        for axis_idx in range(len(axis_keys)):
            axis_params = sync[f'axis_{axis_idx}']
            axis_direction = np.sign(delta[axis_idx])
            axis_velocity = axis_params['vel'] * axis_direction
            print(f"  Axis {axis_idx}: velocity = {axis_velocity:.6f} m/s")
        print(f"Sync axis: {sync['sync_axis']} (velocity = {move_vel * direction:.6f} m/s)")

        t_total_move, move_profile = calculate_travel_time(
            move_dist, move_vel, move_accel, move_jerk
        )
        t_move = move_profile['t'] + t_offset
        v_move = move_profile['v'] * direction
        pos_move = np.cumsum(v_move) * (t_move[1] - t_move[0]) + pos_offset
        a_move = np.zeros_like(v_move)
        a_move[1:] = np.diff(v_move) / np.diff(t_move)
        j_move = np.zeros_like(a_move)
        j_move[1:] = np.diff(a_move) / np.diff(t_move)

        t_full.append(t_move)
        v_full.append(v_move)
        pos_full.append(pos_move)
        a_full.append(a_move)
        j_full.append(j_move)
        phase_spans.append((t_move[0], t_move[-1], 'move'))

        t_offset = t_move[-1]
        pos_offset = pos_move[-1]

        scan_time, t_scan, v_scan = calculate_scan_time(RADIUS, return_profile=True)
        t_scan = t_scan + t_offset
        pos_scan = np.cumsum(v_scan) * (t_scan[1] - t_scan[0]) + pos_offset
        a_scan = np.zeros_like(v_scan)
        a_scan[1:] = np.diff(v_scan) / np.diff(t_scan)
        j_scan = np.zeros_like(a_scan)
        j_scan[1:] = np.diff(a_scan) / np.diff(t_scan)

        t_full.append(t_scan)
        v_full.append(v_scan)
        pos_full.append(pos_scan)
        a_full.append(a_scan)
        j_full.append(j_scan)
        phase_spans.append((t_scan[0], t_scan[-1], 'scan'))

        t_offset = t_scan[-1]
        pos_offset = pos_scan[-1]

    t_full = np.concatenate(t_full)
    pos_full = np.concatenate(pos_full)
    v_full = np.concatenate(v_full)
    a_full = np.concatenate(a_full)
    j_full = np.concatenate(j_full)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for start, end, phase in phase_spans:
        color = '#ffeeba' if phase == 'move' else '#d4edda'
        label = 'Move' if phase == 'move' else 'Scan'
        for ax in axs.flat:
            ax.axvspan(start, end, color=color, alpha=0.4, label=label if ax==axs[0,0] else "")

    axs[0, 0].plot(t_full, pos_full, color='orange')
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
    plt.savefig(f"Full_Motion_Profile_{int(args.expo*1000)}ms{args.proj}proj.png") 
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
    parser.add_argument('--fov', type=str, default='FOV.csv', help='FOV CSV file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CycleTime = args.expo * args.proj
    rectangles = load_FOV_from_csv(args.fov)
    JERK_S, ACCELERATION_S, VELOCITY_S = calculate_maximum_velocity(
        radius=args.radius, CycleTime=CycleTime, jerk=args.jerk, accel=args.acceleration, velocity=args.velocity
    )
    total_time, segment_times = calculate_board_movement_time(rectangles)
    plot_motion_profile(rectangles)