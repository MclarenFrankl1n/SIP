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
RADIUS = 0.1    # meters
EXPO = 0.05
PROJ = 32
CycleTime = EXPO * PROJ
parabolic_ratio = 2/3
DT = 0.001
num_rev = 1.0

def calculate_travel_time(distance, v_max, a_max, j_max, verbose=False):
    """
    Calculate the travel time for a motion profile based on distance, maximum velocity, acceleration, and jerk.
    Returns:
        tuple: Total travel time in seconds and a dictionary containing the motion profile type, time values, velocity values, and total time.
    """

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

        t_up = np.linspace(0, ramp_time, int(ramp_time/DT)+1)
        accel_up = a * t_up**2 + b * t_up
        vel_up = np.zeros_like(t_up)
        pos_up = np.zeros_like(t_up)
        for i in range(1, len(t_up)):
            vel_up[i] = vel_up[i-1] + accel_up[i-1] * DT
            pos_up[i] = pos_up[i-1] + vel_up[i-1] * DT

        # Ramp-down (mirror of ramp-up)
        t_down = np.linspace(0, ramp_time, int(ramp_time/DT)+1)
        accel_down = -accel_up[::-1]
        vel_down = np.zeros_like(t_down)
        pos_down = np.zeros_like(t_down)
        vel_down[0] = vel_up[-1]
        pos_down[0] = pos_up[-1]
        for i in range(1, len(t_down)):
            vel_down[i] = vel_down[i-1] + accel_down[i-1] * DT
            pos_down[i] = pos_down[i-1] + vel_down[i-1] * DT

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

        t_up = np.linspace(0, ramp_time, int(ramp_time/DT)+1)
        accel_up = a * t_up**2 + b * t_up
        vel_up = np.zeros_like(t_up)
        pos_up = np.zeros_like(t_up)
        for i in range(1, len(t_up)):
            vel_up[i] = vel_up[i-1] + accel_up[i-1] * DT
            pos_up[i] = pos_up[i-1] + vel_up[i-1] * DT

        t_cruise = np.linspace(0, cruise_time, int(cruise_time/DT)+1)
        vel_cruise = np.ones_like(t_cruise) * vel_up[-1]
        pos_cruise = np.zeros_like(t_cruise)
        pos_cruise[0] = pos_up[-1]
        for i in range(1, len(t_cruise)):
            pos_cruise[i] = pos_cruise[i-1] + vel_cruise[i-1] * DT

        # Ramp-down (mirror of ramp-up)
        t_down = np.linspace(0, ramp_time, int(ramp_time/DT)+1)
        accel_down = -accel_up[::-1]
        vel_down = np.zeros_like(t_down)
        pos_down = np.zeros_like(t_down)
        vel_down[0] = vel_cruise[-1] if len(vel_cruise) > 0 else vel_up[-1]
        pos_down[0] = pos_cruise[-1] if len(pos_cruise) > 0 else pos_up[-1]
        for i in range(1, len(t_down)):
            vel_down[i] = vel_down[i-1] + accel_down[i-1] * DT
            pos_down[i] = pos_down[i-1] + vel_down[i-1] * DT

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

    t_up = np.linspace(0, ramp_time, int(ramp_time/DT)+1)
    t_cruise = np.linspace(0, cruise_time, int(cruise_time/DT)+1)
    t_down = np.linspace(0, ramp_time, int(ramp_time/DT)+1)

    # --- Ramp-up phase ---
    accel_up = a * t_up**2 + b * t_up
    vel_up = np.zeros_like(t_up)
    pos_up = np.zeros_like(t_up)
    for i in range(1, len(t_up)):
        vel_up[i] = vel_up[i-1] + accel_up[i-1] * DT
        pos_up[i] = pos_up[i-1] + vel_up[i-1] * DT
    jerk_up = np.zeros_like(t_up)
    jerk_up[1:] = (accel_up[1:] - accel_up[:-1]) / DT

    # Cruise position
    accel_cruise = np.zeros_like(t_cruise)
    vel_cruise = np.ones_like(t_cruise) * vel_up[-1]
    pos_cruise = np.zeros_like(t_cruise)
    pos_cruise[0] = pos_up[-1]
    for i in range(1, len(t_cruise)):
        pos_cruise[i] = pos_cruise[i-1] + vel_cruise[i-1] * DT
    jerk_cruise = np.zeros_like(t_cruise)

        # --- Ramp-down phase (mirror of ramp-up) ---
    accel_down = -accel_up[::-1]
    vel_down = np.zeros_like(t_down)
    pos_down = np.zeros_like(t_down)
    vel_down[0] = vel_cruise[-1]
    pos_down[0] = pos_cruise[-1]
    for i in range(1, len(t_down)):
        vel_down[i] = vel_down[i-1] + accel_down[i-1] * DT
        pos_down[i] = pos_down[i-1] + vel_down[i-1] * DT
    jerk_down = np.zeros_like(t_down)
    jerk_down[1:] = (accel_down[1:] - accel_down[:-1]) / DT

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

    print(pos[0], pos[-1])

    if verbose:
        print(f"\n[Scan Time Per FOV] = {total_time:.5f} seconds\n")



    if return_profile:
        return total_time, t, vel, accel, pos, ramp_distance
    return total_time

def get_ptp_positions(rectangles):
    """
    Returns X, Y, Z positions at every ms for all point-to-point (P2P) moves,
    """
    axis_keys = [k for k in rectangles[0] if k.startswith('c')]
    t_full = []
    x_full = []
    y_full = []
    z_full = []
    t_offset = 0
    curr_pos = np.array([rectangles[0][axis_keys[0]], rectangles[0][axis_keys[1]], rectangles[0].get('cz', 0.0)])

    for i in range(1, len(rectangles)):
        prev = np.array([rectangles[i - 1][k] for k in axis_keys] + [rectangles[i - 1].get('cz', 0.0)])
        curr = np.array([rectangles[i][k] for k in axis_keys] + [rectangles[i].get('cz', 0.0)])
        delta = curr - prev

        sync = synchronize_multi_axis_motion(prev, curr)
        sync_axis = int(sync['sync_axis'].replace('Axis', ''))
        params0 = sync['axis_0']
        params1 = sync['axis_1']
        params2 = sync.get('axis_2', {'dist':0,'vel':0,'accel':0,'jerk':0})

        sync_params = sync[f'axis_{sync_axis}']
        t_total_move, move_profile = calculate_travel_time(
            sync_params['dist'], sync_params['vel'], sync_params['accel'], sync_params['jerk']
        )
        t_move = move_profile['t'] + t_offset

        vx = np.zeros_like(t_move)
        vy = np.zeros_like(t_move)
        vz = np.zeros_like(t_move)
        for axis, params in enumerate([params0, params1, params2]):
            t_axis, prof_axis = calculate_travel_time(
                params['dist'], params['vel'], params['accel'], params['jerk']
            )
            v_interp = np.interp(t_move - t_offset, prof_axis['t'], prof_axis['v'])
            if axis == 0:
                vx = v_interp * np.sign(delta[0])
            elif axis == 1:
                vy = v_interp * np.sign(delta[1])
            else:
                vz = v_interp * np.sign(delta[2])

        x = np.zeros_like(t_move)
        y = np.zeros_like(t_move)
        z = np.zeros_like(t_move)
        x[0] = curr_pos[0]
        y[0] = curr_pos[1]
        z[0] = curr_pos[2]
        for j in range(1, len(t_move)):
            dt_j = t_move[j] - t_move[j-1]
            x[j] = x[j-1] + vx[j-1] * dt_j
            y[j] = y[j-1] + vy[j-1] * dt_j
            z[j] = z[j-1] + vz[j-1] * dt_j

        t_full.append(t_move)
        x_full.append(x)
        y_full.append(y)
        z_full.append(z)
        t_offset = t_move[-1]
        curr_pos = curr.copy()

    t_full = np.concatenate(t_full)
    x_full = np.concatenate(x_full)
    y_full = np.concatenate(y_full)
    z_full = np.concatenate(z_full)

    # Interpolate to every ms
    t_ms = np.arange(t_full[0], t_full[-1], 0.001)
    x_ms = np.interp(t_ms, t_full, x_full)
    y_ms = np.interp(t_ms, t_full, y_full)
    z_ms = np.interp(t_ms, t_full, z_full)

    df = pd.DataFrame({'time_s': t_ms, 'x_m': x_ms, 'y_m': y_ms, 'z_m': z_ms})
    return df

def get_circular_positions(center, radius, t_offset=0.0, z=0.0):
    """
    Returns X, Y, Z positions at every ms for circular motion at a given FOV,
    """
    scan_time, t_scan, v_scan, a_scan, pos_scan, ramp_distance = calculate_scan_time(radius, return_profile=True)
    t_scan = t_scan + t_offset
    theta = (pos_scan - ramp_distance) / radius  # Start at negative offset

    x_scan = center[0] + radius * np.cos(theta)
    y_scan = center[1] + radius * np.sin(theta)
    z_scan = np.ones_like(x_scan) * z

    # Interpolate to exact ms steps
    t_ms = np.arange(t_scan[0], t_scan[-1], 0.001)
    x_ms = np.interp(t_ms, t_scan, x_scan)
    y_ms = np.interp(t_ms, t_scan, y_scan)
    z_ms = np.interp(t_ms, t_scan, z_scan)

    df = pd.DataFrame({'time_s': t_ms, 'x_m': x_ms, 'y_m': y_ms, 'z_m': z_ms})
    return df

def load_FOV_from_csv(filename):
    rectangles = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rect = {}
            for k, v in row.items():
                try:
                    rect[k] = float(v) / 1e9
                except Exception as e:
                    print(f"Error converting {k}={v}: {e}")
                    rect[k] = 0.0
            rectangles.append(rect)
    return rectangles

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

    CycleTime = args.expo * args.proj
    try:
        rectangles = load_FOV_from_csv(args.fov)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    print(f"Loaded {len(rectangles)} FOVs from {args.fov}")
    print(rectangles)

    # Only run P2P if there are at least two FOVs
    if len(rectangles) >= 2:
        df_ptp = get_ptp_positions(rectangles)
        df_ptp.to_csv("ptp_positions_every_ms.csv", index=False)
        print("Saved P2P positions at every ms to ptp_positions_every_ms.csv")
    else:
        print("Not enough FOVs for P2P motion (need at least 2 rows in CSV).")

    # Always run circular motion for each FOV
for i, rect in enumerate(rectangles):
    center = [rect.get('cx', 0.0), rect.get('cy', 0.0)]
    z = rect.get('cz', 0.0)
    df_circ = get_circular_positions(center, RADIUS, t_offset=0.0, z=z)
    df_circ.to_csv(f"circular_positions_fov{i}_every_ms.csv", index=False)
    print(f"Saved circular positions for FOV {i} at every ms to circular_positions_fov{i}_every_ms.csv")