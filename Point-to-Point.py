"""
First Method: Stop and Go
Objective: Find Point to Point Distance and Time for each FOV
There are 3 components to this problem:
1. Find the distance travelled and time taken by the source for each projection.
2. Find total time taken to capture the image for each projection. 
1&2. Scan time per FOV (for all projections)
3. Find the total time taken to move from one FOV to another.
board movement timing with multi-axis synchronization

Set Bottom-Left Corner as the origin (0,0).

Assume S-curve velocity profile for the source with jerk properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

# PARAMETERS
PI = np.pi
ACCELERATION = 5      # m/s²  
VELOCITY = 1          # m/s
JERK = 100            # m/s³
RADIUS = 0.1    # meters
EXPO = 0.05
PROJ = 32
CycleTime = EXPO * PROJ
parabolic_ratio = 2/3
dt = 0.001
num_rev = 1.0

def calculate_travel_time_trapezoidal(distance, v_max, a_max, resolution=1000):
    # Calculate ramp time and distance
    t_ramp = v_max / a_max
    s_ramp = 0.5 * a_max * t_ramp**2

    if distance < 2 * s_ramp:
        # Triangular profile (never reaches v_max)
        t_ramp = (distance / a_max)**0.5
        t_total = 2 * t_ramp
        t_vals = np.linspace(0, t_total, resolution)
        v_vals = np.zeros_like(t_vals)
        for i, t in enumerate(t_vals):
            if t < t_ramp:
                v_vals[i] = a_max * t
            else:
                v_vals[i] = a_max * (t_total - t)
    else:
        # Trapezoidal profile
        t_flat = (distance - 2 * s_ramp) / v_max
        t_total = 2 * t_ramp + t_flat
        t_vals = np.linspace(0, t_total, resolution)
        v_vals = np.zeros_like(t_vals)
        for i, t in enumerate(t_vals):
            if t < t_ramp:
                v_vals[i] = a_max * t
            elif t < t_ramp + t_flat:
                v_vals[i] = v_max
            else:
                v_vals[i] = a_max * (t_total - t)
    # Integrate velocity to get position
    p_vals = np.zeros_like(t_vals)
    for i in range(1, len(t_vals)):
        p_vals[i] = p_vals[i-1] + v_vals[i-1] * (t_vals[i] - t_vals[i-1])
    return t_total, {'type': 'trapezoidal', 't': t_vals, 'v': v_vals, 'p': p_vals, 't_total': t_total}

def calculate_travel_time_SCurve(distance, v_max, a_max, j_max, resolution=1000, verbose=True):
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
        return calculate_travel_time_trapezoidal(distance, v_max, a_max, resolution=resolution)
        # Triangular motion profile
        # t_j = (3 * distance / (2 * j_max))**(1/3)
        # t_total = 4 * t_j
        # t_vals = np.linspace(0, t_total, resolution)
        # v_vals = np.zeros_like(t_vals)
        # if verbose:
        #     print("Triangular Profile")

        # t1 = t_j
        # t2 = 2 * t_j
        # t3 = 3 * t_j

        # for i, t in enumerate(t_vals):
        #     if t < t1:
        #         v_vals[i] = 0.5 * j_max * t**2
        #     elif t < t2:
        #         dt = t - t_j
        #         v_vals[i] = (j_max * t_j**2 / 2) + j_max * t_j * dt - j_max * dt**2 / 2
        #     elif t < t3:
        #         dt = t - 2 * t_j
        #         v_vals[i] = (j_max * t_j**2) - j_max * dt**2 / 2
        #     else:
        #         dt = t - 3 * t_j
        #         v_vals[i] = j_max * t_j**2 / 2 - j_max * t_j * dt + j_max * dt**2 / 2

        # return t_total, {
        #     'type': 'triangular',
        #     't': t_vals,
        #     'v': v_vals,
        #     't_total': t_total
        # }

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

def calculate_travel_time(distance, v_max, a_max, j_max, resolution=1000, verbose=False):
    """
    Calculate the travel time for a motion profile based on distance, maximum velocity, acceleration, and jerk.
    Returns:
        tuple: Total travel time in seconds and a dictionary containing the motion profile type, time values, velocity values, and total time.
    """

    dt = 1.0 / resolution
    parabolic_ratio = 2.0/3
    ramp_time = v_max / (a_max * parabolic_ratio)
    half_ramp_time = ramp_time / 2
    b = a_max / (half_ramp_time - (half_ramp_time**2) / ramp_time)
    a = -b / ramp_time

    # Distance covered during ramp-up (and ramp-down)
    ramp_distance = (v_max ** 2) / (2 * a_max * parabolic_ratio)
    cruise_distance = distance - 2 * ramp_distance
    cruise_time = cruise_distance / v_max

    total_ramp_distance = 2 * ramp_distance
    print(f"total_ramp_distance = {total_ramp_distance:.5f} m  ")
    print(f"distance = {distance:.5f} m  ")

    # If distance is too short for full S-curve, use triangular (jerk-limited) profile
    if distance <= total_ramp_distance + 1e-10:
        # --- Parabolic (jerk-limited) triangular profile using scan logic ---
        # Find ramp_time such that total area under velocity curve = distance
        # For parabolic profile: ramp_distance = (v_peak ** 2) / (2 * a_max * parabolic_ratio)
        # Total move distance = 2 * ramp_distance
        v_peak = np.sqrt(distance * a_max * parabolic_ratio)
        ramp_time = v_peak / (a_max * parabolic_ratio)
        half_ramp_time = ramp_time / 2
        b = a_max / (half_ramp_time - (half_ramp_time**2) / ramp_time)
        a = -b / ramp_time

        t_up = np.linspace(0, ramp_time, int(ramp_time/dt)+1)
        accel_up = a * t_up**2 + b * t_up
        vel_up = np.zeros_like(t_up)
        pos_up = np.zeros_like(t_up)
        for i in range(1, len(t_up)):
            vel_up[i] = vel_up[i-1] + accel_up[i-1] * dt
            pos_up[i] = pos_up[i-1] + vel_up[i-1] * dt

        # Ramp-down (mirror of ramp-up)
        t_down = np.linspace(0, ramp_time, int(ramp_time/dt)+1)
        accel_down = -accel_up[::-1]
        vel_down = np.zeros_like(t_down)
        pos_down = np.zeros_like(t_down)
        vel_down[0] = vel_up[-1]
        pos_down[0] = pos_up[-1]
        for i in range(1, len(t_down)):
            vel_down[i] = vel_down[i-1] + accel_down[i-1] * dt
            pos_down[i] = pos_down[i-1] + vel_down[i-1] * dt

        # Concatenate
        t = np.concatenate([t_up[:-1], t_down + t_up[-1]])
        v = np.concatenate([vel_up[:-1], vel_down])
        a_prof = np.concatenate([accel_up[:-1], accel_down])
        p = np.concatenate([pos_up[:-1], pos_down])
        t_total = t[-1]

        if verbose:
            print("Triangular (parabolic/jerk-limited) profile using scan logic")
            print(f"ramp_time = {ramp_time:.5f} s, v_peak = {v_peak:.5f} m/s, t_total = {t_total:.5f} s")

        return t_total, {
            'type': 'triangular',
            't': t,
            'v': v,
            'a': a_prof,
            'p': p,
            't_total': t_total
        }

    else:
        # S-curve profile (parabolic ramps, cruise, parabolic ramps)

        # Distance covered during ramp-up (and ramp-down)
        cruise_time = cruise_distance / v_max if cruise_distance > 0 else 0

        total_time = ramp_time * 2 + cruise_time
        print(f"Total time = {total_time:.5f} s")

        t_up = np.linspace(0, ramp_time, int(ramp_time/dt)+1)
        accel_up = a * t_up**2 + b * t_up
        vel_up = np.zeros_like(t_up)
        pos_up = np.zeros_like(t_up)
        for i in range(1, len(t_up)):
            vel_up[i] = vel_up[i-1] + accel_up[i-1] * dt
            pos_up[i] = pos_up[i-1] + vel_up[i-1] * dt

        t_cruise = np.linspace(0, cruise_time, int(cruise_time/dt)+1)
        vel_cruise = np.ones_like(t_cruise) * vel_up[-1]
        pos_cruise = np.zeros_like(t_cruise)
        pos_cruise[0] = pos_up[-1]
        for i in range(1, len(t_cruise)):
            pos_cruise[i] = pos_cruise[i-1] + vel_cruise[i-1] * dt

        # Ramp-down (mirror of ramp-up)
        t_down = np.linspace(0, ramp_time, int(ramp_time/dt)+1)
        accel_down = -accel_up[::-1]
        vel_down = np.zeros_like(t_down)
        pos_down = np.zeros_like(t_down)
        vel_down[0] = vel_cruise[-1] if len(vel_cruise) > 0 else vel_up[-1]
        pos_down[0] = pos_cruise[-1] if len(pos_cruise) > 0 else pos_up[-1]
        for i in range(1, len(t_down)):
            vel_down[i] = vel_down[i-1] + accel_down[i-1] * dt
            pos_down[i] = pos_down[i-1] + vel_down[i-1] * dt

        # Concatenate
        t = np.concatenate([t_up[:-1], t_cruise[:-1] + t_up[-1], t_down + t_up[-1] + t_cruise[-1]])
        v = np.concatenate([vel_up[:-1], vel_cruise[:-1], vel_down])
        a_prof = np.concatenate([accel_up[:-1], np.zeros_like(t_cruise[:-1]), accel_down])
        p = np.concatenate([pos_up[:-1], pos_cruise[:-1], pos_down])
        t_total = t[-1]

        return t_total, {
            'type': 's-curve',
            't': t,
            'v': v,
            'a': a_prof,
            'p': p,
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
        t,_ = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK, verbose=True)
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

def synchronize_multi_axis_motion_SCurve(p1, p2):
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
        t,_ = calculate_travel_time_SCurve(dist, VELOCITY, ACCELERATION, JERK, verbose=True)
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

def load_FOV_from_csv(filename):
    rectangles = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert all values from nm to m
            rect = {k: float(v) / 1e9 for k, v in row.items()}
            rectangles.append(rect)
    return rectangles

def plot_xy_motion_profile(rectangles, args):

    t_full = []
    vx_full = []
    vy_full = []
    ax_full = []
    ay_full = []
    phase_spans = []

    axis_keys = [k for k in rectangles[0] if k.startswith('c')]
    curr_pos = np.array([rectangles[0][axis_keys[0]], rectangles[0][axis_keys[1]]])
    x_full = []
    y_full = []

    for i in range(1, len(rectangles)):
        prev = np.array([rectangles[i - 1][k] for k in axis_keys])
        curr = np.array([rectangles[i][k] for k in axis_keys])
        delta = curr - prev

        # Synchronize axes
        sync = synchronize_multi_axis_motion(prev, curr)
        sync_axis = int(sync['sync_axis'].replace('Axis', ''))
        t_sync = sync['t_sync']

        # Get parameters for both axes
        params0 = sync['axis_0']
        params1 = sync['axis_1']

        # Generate sync axis profile
        sync_params = sync[f'axis_{sync_axis}']
        t_total_move, move_profile = calculate_travel_time(
            sync_params['dist'], sync_params['vel'], sync_params['accel'], sync_params['jerk']
        )
        t_move = move_profile['t']

        # Interpolate non-sync axis profile to sync axis time base
        vx = np.zeros_like(t_move)
        vy = np.zeros_like(t_move)
        for axis, params in enumerate([params0, params1]):
            t_axis, prof_axis = calculate_travel_time(
                params['dist'], params['vel'], params['accel'], params['jerk']
            )
            v_interp = np.interp(t_move , prof_axis['t'], prof_axis['v'])
            if axis == 0:
                vx = v_interp * np.sign(delta[0])
            else:
                vy = v_interp * np.sign(delta[1])

        # Integrate for position
        x = np.zeros_like(t_move)
        y = np.zeros_like(t_move)
        x[0] = curr_pos[0]
        y[0] = curr_pos[1]
        for j in range(1, len(t_move)):
            dt_j = t_move[j] - t_move[j-1]
            x[j] = x[j-1] + vx[j-1] * dt_j
            y[j] = y[j-1] + vy[j-1] * dt_j

        # --- Scale position so it ends at the correct target ---
        if abs(x[-1] - curr[0]) > 1e-9:
            x = x + (curr[0] - x[-1]) * (t_move / t_move[-1])
        if abs(y[-1] - curr[1]) > 1e-9:
            y = y + (curr[1] - y[-1]) * (t_move / t_move[-1])

        # Acceleration
        ax = np.zeros_like(t_move)
        ay = np.zeros_like(t_move)
        ax[1:] = np.diff(vx) / np.diff(t_move)
        ay[1:] = np.diff(vy) / np.diff(t_move)

        t_full.append(t_move)
        x_full.append(x)
        y_full.append(y)
        vx_full.append(vx)
        vy_full.append(vy)
        ax_full.append(ax)
        ay_full.append(ay)
        phase_spans.append((t_move[0], t_move[-1], 'move'))

        curr_pos = curr.copy()


    t_full = np.concatenate(t_full)
    x_full = np.concatenate(x_full)
    y_full = np.concatenate(y_full)
    vx_full = np.concatenate(vx_full)
    vy_full = np.concatenate(vy_full)
    ax_full = np.concatenate(ax_full)
    ay_full = np.concatenate(ay_full)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

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

    # Empty subplot
    axs[1, 1].axis('off')

    # Legends
    for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()   

def plot_xy_motion_profile_scurve(rectangles, args):

    axis_keys = [k for k in rectangles[0] if k.startswith('c')]
    curr_pos = np.array([rectangles[0][axis_keys[0]], rectangles[0][axis_keys[1]]])

    t_full, x_full, y_full, vx_full, vy_full, ax_full, ay_full = [], [], [], [], [], [], []

    for i in range(1, len(rectangles)):
        prev = np.array([rectangles[i - 1][k] for k in axis_keys])
        curr = np.array([rectangles[i][k] for k in axis_keys])
        delta = curr - prev

                # --- Use multi-axis sync to get sync time and axis ---
        sync = synchronize_multi_axis_motion(prev, curr)
        sync_axis = int(sync['sync_axis'].replace('Axis', ''))
        t_sync = sync['t_sync']

        # Generate S-curve profile for sync axis (with its own limits)
        sync_params = sync[f'axis_{sync_axis}']
        _, sync_prof = calculate_travel_time_SCurve(
            sync_params['dist'], sync_params['vel'], sync_params['accel'], sync_params['jerk']
        )
        t_sync_base = sync_prof['t']
        v_sync = sync_prof['v'] * np.sign(delta[sync_axis])

        # Generate S-curve profile for non-sync axis (with its own limits)
        non_sync_axis = 1 - sync_axis
        non_sync_params = sync[f'axis_{non_sync_axis}']
        _, non_sync_prof = calculate_travel_time_SCurve(
            non_sync_params['dist'], VELOCITY, ACCELERATION, JERK
        )
        v_non_sync = np.interp(t_sync_base, non_sync_prof['t'], non_sync_prof['v']) * np.sign(delta[non_sync_axis])

        # Assign vx, vy according to which is sync axis
        if sync_axis == 0:
            vx = v_sync
            vy = v_non_sync
        else:
            vx = v_non_sync
            vy = v_sync

        # Integrate for position
        x = np.zeros_like(t_sync_base)
        y = np.zeros_like(t_sync_base)
        x[0] = curr_pos[0]
        y[0] = curr_pos[1]
        for j in range(1, len(t_sync_base)):
            dt_j = t_sync_base[j] - t_sync_base[j-1]
            x[j] = x[j-1] + vx[j-1] * dt_j
            y[j] = y[j-1] + vy[j-1] * dt_j

        # Acceleration
        ax = np.zeros_like(t_sync_base)
        ay = np.zeros_like(t_sync_base)
        ax[1:] = np.diff(vx) / np.diff(t_sync_base)
        ay[1:] = np.diff(vy) / np.diff(t_sync_base)

        t_full.append(t_sync_base)
        x_full.append(x)
        y_full.append(y)
        vx_full.append(vx)
        vy_full.append(vy)
        ax_full.append(ax)
        ay_full.append(ay)

        curr_pos = curr.copy()

    t_full = np.concatenate(t_full)
    x_full = np.concatenate(x_full)
    y_full = np.concatenate(y_full)
    vx_full = np.concatenate(vx_full)
    vy_full = np.concatenate(vy_full)
    ax_full = np.concatenate(ax_full)
    ay_full = np.concatenate(ay_full)

    # --- Plotting (same as before) ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(t_full, x_full, color='blue', label='X Position')
    axs[0, 0].plot(t_full, y_full, color='red', label='Y Position')
    axs[0, 0].set_title("X/Y Position (m)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].grid(True)
    axs[0, 1].plot(t_full, vx_full, color='blue', label='X Velocity')
    axs[0, 1].plot(t_full, vy_full, color='red', label='Y Velocity')
    axs[0, 1].set_title("X/Y Velocity (m/s)")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].grid(True)
    axs[1, 0].plot(t_full, ax_full, color='blue', label='X Accel')
    axs[1, 0].plot(t_full, ay_full, color='red', label='Y Accel')
    axs[1, 0].set_title("X/Y Acceleration (m/s²)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Acceleration (m/s²)")
    axs[1, 0].grid(True)
    axs[1, 1].axis('off')
    for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_individual_axis_scurve(rectangles, args):
    axis_keys = [k for k in rectangles[0] if k.startswith('c')]
    for i in range(1, len(rectangles)):
        prev = np.array([rectangles[i - 1][k] for k in axis_keys])
        curr = np.array([rectangles[i][k] for k in axis_keys])
        delta = curr - prev

        t0, prof0 = calculate_travel_time_SCurve(abs(delta[0]), VELOCITY, ACCELERATION, JERK, verbose=True)
        t1, prof1 = calculate_travel_time_SCurve(abs(delta[1]), VELOCITY, ACCELERATION, JERK, verbose=True)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(prof0['t'], prof0['v'] * np.sign(delta[0]), label='X Velocity', color='blue')
        plt.plot(prof1['t'], prof1['v'] * np.sign(delta[1]), label='Y Velocity', color='red')
        plt.title('Individual Axis S-curve Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        ax = np.zeros_like(prof0['t'])
        ay = np.zeros_like(prof1['t'])
        ax[1:] = np.diff(prof0['v']) / np.diff(prof0['t'])
        ay[1:] = np.diff(prof1['v']) / np.diff(prof1['t'])
        plt.plot(prof0['t'], ax, label='X Accel', color='blue')
        plt.plot(prof1['t'], ay, label='Y Accel', color='red')
        plt.title('Individual Axis S-curve Acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="CycleTime and Motion Profile Calculation for Point-to-Point"
    )
    parser.add_argument('--acceleration', type=float, default=5, help='Acceleration (m/s^2)')
    parser.add_argument('--velocity', type=float, default=1, help='Velocity (m/s)')
    parser.add_argument('--jerk', type=float, default=100, help='Jerk (m/s^3)')
    parser.add_argument('--radius', type=float, default=0.1, help='Radius (m)')
    parser.add_argument('--expo', type=float, default=0.05, help='Exposure time (s)')
    parser.add_argument('--proj', type=int, default=32, help='Number of projections')
    parser.add_argument('--fov', type=str, default='ScanTests.csv', help='FOV CSV file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ACCELERATION = args.acceleration
    JERK = args.jerk
    VELOCITY = args.velocity
    RADIUS = args.radius

    CycleTime = args.expo * args.proj
    rectangles = load_FOV_from_csv(args.fov)
    # plot_xy_motion_profile(rectangles, args)
    plot_xy_motion_profile_scurve(rectangles, args)
    plot_individual_axis_scurve(rectangles, args)