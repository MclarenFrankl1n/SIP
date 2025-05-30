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
    Returns:
    Ramp-up time, Ramp-down time in seconds
    """
    t_j = acceleration / jerk  # time to reach max acceleration
    t_const_a = (velocity - acceleration * t_j) / acceleration  # time at constant acceleration
    ramp_up_time = 2 * t_j + t_const_a  # total ramp-up time
    ramp_down_time = ramp_up_time  # assuming symmetry
    print(f"Ramp-up Time = {ramp_up_time:.5f} s, Ramp-down Time = {ramp_down_time:.5f} s")
    return ramp_up_time, ramp_down_time



# --- TESTING ---

mock_rectangles = [ 
    {'cx': 0, 'cy': 0, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 100_000_000, 'cy': 0, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 100_000_000, 'cy': 100_000_000, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 0, 'cy': 100_000_000, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000},
    {'cx': 0, 'cy': 0, 'cz': 0, 'cw': 100_000_000, 'ch': 100_000_000}
    ]


if __name__ == "__main__":
    JERK, ACCELERATION, VELOCITY = calculate_maximum_velocity(R, CYCLE_TIME)
    print(f"Using: JERK={JERK:.2e}, ACCELERATION={ACCELERATION:.2e}, VELOCITY={VELOCITY:.2e}")
    calculate_ramp_up_down_time(VELOCITY, ACCELERATION, JERK)