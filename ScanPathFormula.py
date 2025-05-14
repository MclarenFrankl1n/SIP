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

# Travel time using S-curve motion profile
def calculate_travel_time(distance, v_max, a_max, j_max):
    t_j = a_max / j_max  # Time to reach max acceleration/deceleration
    t_const_a = (v_max - a_max * t_j) / a_max  # Constant accel/decel duration to hit v_max
    s_accel = j_max * t_j**3 + a_max * t_const_a * t_j + 0.5 * a_max * t_const_a**2  # Distance during full accel
    s_total_accel_decel = 2 * s_accel  # Total distance for full accel + decel
    print(f"t_j = {t_j:.5f} s, t_const_a = {t_const_a:.5f} s")
    print(f"s_accel = {s_accel:.5f} nm, s_total_accel_decel = {s_total_accel_decel:.5f} nm")

    if distance < s_total_accel_decel:
        t_j = (3*distance / (2 * j_max))**(1/3)
        print("Triangular profile")
        return 4 * t_j
    else:
        s_cruise = distance - s_total_accel_decel
        t_cruise = s_cruise / v_max
        t_accel = 2 * t_j + t_const_a
        t_decel = 2 * t_j + t_const_a
        print("S-curve profile")
        return t_accel + t_cruise + t_decel

# Scan time per FOV (for all projections)
def calculate_scan_time(verbose=False):
    scan_times = 0
    for i in range(PROJ):
        dist = 2 * PI * R / PROJ
        time = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK)
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
        travel_time = calculate_travel_time(dist, VELOCITY, ACCELERATION, JERK)
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


# --- TESTING ---

mock_rectangles = [ 
    {'cx': 0, 'cy': 0},
    {'cx': 100_000_000, 'cy': 0},
    {'cx': 100_000_000, 'cy': 100_000_000},
    {'cx': 200_000_000, 'cy': 100_000_000}
    ]
    

if __name__ == "__main__":
    calculate_board_movement_time(mock_rectangles)
    plot_fov_path(mock_rectangles)
