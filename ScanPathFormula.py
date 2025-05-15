import numpy as np
import matplotlib.pyplot as plt

# Constants
ACCELERATION = 5_000_000_000  # nm/s²
VELOCITY = 1_000_000_000      # nm/s
JERK = 100_000_000_000          # nm/s³
PI = 3.141592653589793
R = 50_000_000                # nm    5cm
PROJ = 32
EXPO = 0.050  # seconds   50ms

def calculate_travel_time(distance, v_max, a_max, j_max, resolution=1000):
    t_j = a_max / j_max
    t_const_a = (v_max - a_max * t_j) / a_max
    s_accel = j_max * t_j**3 + a_max * t_const_a * t_j + 0.5 * a_max * t_const_a**2
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
    for i in range(PROJ):
        dist = 2 * PI * R / PROJ
        time,_ = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK)
        scan_times += EXPO + time
        if verbose:
            print(f"  [Scan {i:02d}] Arc Dist = {dist:.1f} nm, Travel = {time:.5f}s, Total = {EXPO + time:.5f}s")
    return scan_times

# Euclidean distance
def calculate_euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Detailed board movement timing
def calculate_board_movement_time(rectangles):
    ptp_times = 0
    scan_times = calculate_scan_time(verbose=True)
    print(f"\n[Scan Time Per FOV] = {scan_times:.5f} seconds\n")

    for i in range(1, len(rectangles)):
        prev = (rectangles[i - 1]['cx'], rectangles[i - 1]['cy'])
        curr = (rectangles[i]['cx'], rectangles[i]['cy'])
        dist = calculate_euclidean_distance(prev, curr)
        plot_velocity_profile(dist)
        travel_time,_ = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK)
        segment_time = travel_time + scan_times

        print(f"[FOV {i}] Move from {prev} to {curr}")
        print(f"         Distance = {dist:.1f} nm")
        print(f"         Travel   = {travel_time:.5f} s")
        print(f"         Scan     = {scan_times:.5f} s")
        print(f"         Segment  = {segment_time:.5f} s\n")

        ptp_times += segment_time

    print(f"Total Movement + Scan Time = {ptp_times:.3f} seconds")
    return ptp_times

def plot_fov_path(rectangles):
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, rect in enumerate(rectangles):
        x, y = rect['cx'], rect['cy']
        ax.plot(x, y, 'bo')  # Point for FOV center
        ax.text(x, y + 5_000_000, f"FOV {i}", fontsize=9, ha='center', color='blue')

        if i > 0:
            prev_x, prev_y = rectangles[i - 1]['cx'], rectangles[i - 1]['cy']
            ax.annotate("",
                        xy=(x, y),
                        xytext=(prev_x, prev_y),
                        arrowprops=dict(arrowstyle="->", color='red'))

    ax.set_title("FOV Movement Path")
    ax.set_xlabel("X Position (nm)")
    ax.set_ylabel("Y Position (nm)")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

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
    # calculate_board_movement_time(mock_rectangles)
    plot_fov_path(mock_rectangles)

    arc_distance = 1000_000_000
    plot_velocity_profile(arc_distance)


