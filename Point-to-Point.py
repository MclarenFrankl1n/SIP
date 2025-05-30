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

# Constants
ACCELERATION = 5_000_000_000  # nm/s²
VELOCITY = 1_000_000_000      # nm/s
JERK = 100_000_000_000          # nm/s³
PI = np.pi
R = 50_000_000                # nm    5cm
PROJ = 32
EXPO = 0.050  # seconds   50ms

def calculate_travel_time(distance, v_max, a_max, j_max, resolution=1000):
    t_j = a_max / j_max # time to reach max acceleration
    t_const_a = (v_max - a_max * t_j) / a_max #time at constant acceleration
    s_accel = 1/3 * j_max * t_j**3 + a_max * t_const_a * t_j + 0.5 * a_max * t_const_a**2
    s_total_accel_decel = 2 * s_accel
    print(f"t_j = {t_j:.5f} s, t_const_a = {t_const_a:.5f} s")
    print(f"s_accel = {s_accel:.5f} nm, s_total_accel_decel = {s_total_accel_decel:.5f} nm")

    if distance < s_total_accel_decel:
        # Triangular motion profile
        t_j = (3 * distance / (2 * j_max))**(1/3)
        t_total = 4 * t_j
        t_vals = np.linspace(0, t_total, resolution)
        v_vals = np.zeros_like(t_vals)
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


# Scan time per FOV (for all projections)
def calculate_scan_time(verbose=False):
    scan_times = 0
    dist = 2 * PI * R / PROJ
    for i in range(PROJ):
        time,_ = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK)
        scan_times += EXPO + time
        if verbose:
            print(f"  [Scan {i:02d}] Arc Dist = {dist:.1f} nm, Travel = {time:.5f}s, Total = {EXPO + time:.5f}s")
    return scan_times

# Multi-axis time normalization (X and Y)
def synchronize_multi_axis_motion(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    t_x, _ = calculate_travel_time(dx, VELOCITY, ACCELERATION, JERK)
    t_y, _ = calculate_travel_time(dy, VELOCITY, ACCELERATION, JERK)

    t_sync = max(t_x, t_y)

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
        'x': {'dist': dx, 'jerk': j_x, 'accel': a_x, 'vel': v_x},
        'y': {'dist': dy, 'jerk': j_y, 'accel': a_y, 'vel': v_y},
    }

# Updated board movement timing with multi-axis synchronization
def calculate_board_movement_time(rectangles):
    ptp_times = 0
    scan_times = calculate_scan_time(verbose=True)
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
    {'cx': 0, 'cy': 0},
    {'cx': 100_000_000, 'cy': 0},
    {'cx': 100_000_000, 'cy': 100_000_000},
    {'cx': 200_000_000, 'cy': 100_000_000}
    ]


if __name__ == "__main__":
    calculate_board_movement_time(mock_rectangles)

    arc_distance = 1000_000_000
    plot_velocity_profile(arc_distance)