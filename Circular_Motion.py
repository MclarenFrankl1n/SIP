import numpy as np
import matplotlib.pyplot as plt

PI = np.pi

def calculate_maximum_velocity(radius, CycleTime, jerk, accel, velocity):
    """
    All parameters and returns are in SI units (meters, seconds).
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

# Reference values (all SI units)
ACCELERATION = 5      # m/s²  
VELOCITY = 1          # m/s
JERK = 100            # m/s³
RADIUS = 0.10577      # meters (was 105.77 mm)
EXPO = 0.05
PROJ = 32
CycleTime = EXPO * PROJ

# Calculate scaled parameters for cruise phase
JERK_S, ACCELERATION_S, VELOCITY_S = calculate_maximum_velocity(
    RADIUS, CycleTime, JERK, ACCELERATION, VELOCITY
)

peak_vel = VELOCITY_S
peak_accel = ACCELERATION_S
peak_jerk = JERK_S
radius = RADIUS

# --- Parameters (edit as needed) ---
parabolic_ratio = 2/3
dt = 0.001
num_rev = 1.0

# --- Ramping time and distance ---
ramp_time = peak_vel / (peak_accel * parabolic_ratio)
half_ramp_time = ramp_time / 2
ramp_distance = (peak_vel**2 / peak_accel)  # meters

# --- Quadratic acceleration coefficients ---
b = peak_accel / (half_ramp_time - (half_ramp_time**2) / ramp_time)
a = -b / ramp_time

# --- Time arrays ---
cruise_distance = 2 * np.pi * radius  # meters (cruise phase only)
cruise_time = cruise_distance / peak_vel  # seconds

t_up = np.arange(0, ramp_time, dt)
t_cruise = np.arange(ramp_time, ramp_time + cruise_time, dt)
t_down = np.arange(ramp_time + cruise_time, ramp_time + cruise_time + ramp_time, dt)

# Then concatenate as:
t = np.concatenate([t_up, t_cruise, t_down])

# --- Ramp-up phase ---
accel_up = a * t_up**2 + b * t_up
vel_up = np.zeros_like(t_up)
pos_up = np.zeros_like(t_up)
for i in range(1, len(t_up)):
    vel_up[i] = vel_up[i-1] + accel_up[i-1] * dt
    pos_up[i] = pos_up[i-1] + vel_up[i-1] * dt
jerk_up = np.zeros_like(t_up)
jerk_up[1:] = (accel_up[1:] - accel_up[:-1]) / dt

# --- Cruise phase ---
accel_cruise = np.zeros_like(t_cruise)
vel_cruise = np.ones_like(t_cruise) * peak_vel
pos_cruise = pos_up[-1] + np.cumsum(vel_cruise) * dt
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


# Add margin (e.g., 0.1s) after the last point
margin = 0.1  # seconds
t_margin = np.arange(t[-1] + dt, t[-1] + margin + dt, dt)
t = np.concatenate([t, t_margin])

# Pad other arrays with their final value for the margin
pos = np.concatenate([pos_up, pos_cruise, pos_down])
vel = np.concatenate([vel_up, vel_cruise, vel_down])
accel = np.concatenate([accel_up, accel_cruise, accel_down])
jerk = np.concatenate([jerk_up, jerk_cruise, jerk_down])

accel = np.concatenate([accel, np.full_like(t_margin, accel[-1])])
vel = np.concatenate([vel, np.full_like(t_margin, vel[-1])])
pos = np.concatenate([pos, np.full_like(t_margin, pos[-1])])
jerk = np.concatenate([jerk, np.zeros_like(t_margin)])

# --- Circular projection ---
degrees = np.degrees(pos / radius)
x_pos = radius * np.cos(np.radians(degrees))  # meters
y_pos = radius * np.sin(np.radians(degrees))  # meters
x_vel = vel * np.sin(np.radians(degrees))
y_vel = vel * np.cos(np.radians(degrees))
x_accel = np.gradient(x_vel, dt)
y_accel = np.gradient(y_vel, dt)

print(f"peak_accel = {peak_accel} m/s^2")
print(f"peak_vel = {peak_vel} m/s")
print(f"parabolic_ratio = {parabolic_ratio}")
print(f"dt = {dt} s")
print(f"radius = {radius} m")
print(f"num_rev = {num_rev}")
print(f"ramp_time = {ramp_time:.6f} s")
print(f"half_ramp_time = {half_ramp_time:.6f} s")
print(f"ramp_distance = {ramp_distance:.6f} m")
print(f"a (quadratic coefficient) = {a:.6f}")
print(f"b (linear coefficient) = {b:.6f}")

print("\nFirst 10 values:")
print("Time\tCirc Accel\tCirc Vel\tCirc Pos\tCirc Jerk\tDegrees\t\tX Pos (m)\tY Pos (m)\tX Vel (m/s)\tY Vel (m/s)\tX Accel\t\tY Accel")
for i in range(10):
    deg = np.degrees(degrees[i])
    print(f"{t[i]:.3f}\t{accel[i]:.7f}\t{vel[i]:.7f}\t{pos[i]:.7f}\t{jerk[i]:.7f}\t"
      f"{deg:.7f}\t{x_pos[i]:.7f}\t{y_pos[i]:.7f}\t{x_vel[i]:.7f}\t{y_vel[i]:.7f}\t"
      f"{x_accel[i]:.7f}\t{y_accel[i]:.7f}")

# Define phase boundaries
t_rampup_end = ramp_time
t_cruise_end = ramp_time + cruise_time
t_rampdown_end = ramp_time + cruise_time + ramp_time

# --- Plotting motion profile: position, velocity, acceleration, jerk ---
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

for ax in axs2.flat:
    # Ramp-up: light blue
    ax.axvspan(0, t_rampup_end, color='#cce5ff', alpha=0.5, label='Ramp-up' if ax==axs2[0,0] else "")
    # Cruise: light green
    ax.axvspan(t_rampup_end, t_cruise_end, color='#d4edda', alpha=0.5, label='Cruise' if ax==axs2[0,0] else "")
    # Ramp-down: light red
    ax.axvspan(t_cruise_end, t_rampdown_end, color='#f8d7da', alpha=0.5, label='Ramp-down' if ax==axs2[0,0] else "")

axs2[0, 0].plot(t, pos, color='orange')
axs2[0, 0].set_title("Position (m)")
axs2[0, 0].set_xlabel("Time (s)")
axs2[0, 0].set_ylabel("Position (m)")
axs2[0, 0].grid(True)

axs2[0, 1].plot(t, vel, color='red')
axs2[0, 1].set_title("Velocity (m/s)")
axs2[0, 1].set_xlabel("Time (s)")
axs2[0, 1].set_ylabel("Velocity (m/s)")
axs2[0, 1].grid(True)

axs2[1, 0].plot(t, accel, color='blue')
axs2[1, 0].set_title("Acceleration (m/s²)")
axs2[1, 0].set_xlabel("Time (s)")
axs2[1, 0].set_ylabel("Acceleration (m/s²)")
axs2[1, 0].grid(True)

axs2[1, 1].plot(t, jerk, color='green')
axs2[1, 1].set_title("Jerk (m/s³)")
axs2[1, 1].set_xlabel("Time (s)")
axs2[1, 1].set_ylabel("Jerk (m/s³)")
axs2[1, 1].grid(True)

# Add legend for phase backgrounds
handles, labels = axs2[0,0].get_legend_handles_labels()
if handles:
    axs2[0,0].legend(loc="upper right")

plt.tight_layout()
plt.show()

# --- Plotting all X/Y position, velocity, acceleration on the same plot ---
plt.figure(figsize=(12, 8))

# Backgrounds for phases
plt.axvspan(0, t_rampup_end, color='#cce5ff', alpha=0.5, label='Ramp-up')
plt.axvspan(t_rampup_end, t_cruise_end, color='#d4edda', alpha=0.5, label='Cruise')
plt.axvspan(t_cruise_end, t_rampdown_end, color='#f8d7da', alpha=0.5, label='Ramp-down')

plt.plot(t, x_pos, label="X Position (m)", color='blue')
plt.plot(t, y_pos, label="Y Position (m)", color='red')
plt.plot(t, x_vel, label="X Velocity (m/s)", color='cyan', linestyle='--')
plt.plot(t, y_vel, label="Y Velocity (m/s)", color='magenta', linestyle='--')
plt.plot(t, x_accel, label="X Acceleration (m/s²)", color='green', linestyle=':')
plt.plot(t, y_accel, label="Y Acceleration (m/s²)", color='orange', linestyle=':')

plt.title("X/Y Position, Velocity, and Acceleration vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Value (SI Units)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()