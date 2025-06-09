import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

# Raw data string (your example)
data = """
23 7 7 7 7
50 20.5 22.8 18.0 18.5
71 22.3 28.0 32.0 27.3
86 23.5 28.0 34.5 35.0
101 23.0 28.3 34.0 35.0
113 23.0 28.3 34.0 35.0
"""

# Parse data string to numpy array (float64)
data_list = [list(map(float, line.split())) for line in data.strip().split('\n')]
data_array = np.array(data_list, dtype=np.float64)

days = data_array[:, 0]
data_array = data_array[:, 1:]  # Remove days column

# Filter zeros per column
filtered_columns = [
    data_array[:, i][data_array[:, i] != 0] for i in range(data_array.shape[1])
]

# Temperature treatments corresponding to each column
temperature_treatments = np.array([26, 30, 34, 38], dtype=np.float64)

# Build global dataset arrays: coordinates (r, T) and values L
r_all = []
T_all = []
L_all = []

for i, T in enumerate(temperature_treatments):
    for r, L in zip(days, filtered_columns[i]):
        r_all.append(r)
        T_all.append(T)
        L_all.append(L)

r_all = np.array(r_all, dtype=np.float64)
T_all = np.array(T_all, dtype=np.float64)
L_all = np.array(L_all, dtype=np.float64)

# Build RBF interpolator (float64)
rbf = RBFInterpolator(
    np.column_stack((r_all, T_all)),
    L_all,
    kernel='multiquadric',
    smoothing=0.1,
    epsilon=0.1
)

# Plot the RBF interpolation surface to check fit

# Create grid for visualization
r_vals = np.linspace(days.min(), days.max(), 100)
T_vals = np.linspace(temperature_treatments.min(), temperature_treatments.max(), 100)
R, T_grid = np.meshgrid(r_vals, T_vals)
points_grid = np.column_stack((R.flatten(), T_grid.flatten()))

# Evaluate interpolator on the grid
L_fit = rbf(points_grid).reshape(R.shape)

# Plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(R, T_grid, L_fit, cmap='viridis', alpha=0.8)

# Scatter original data points
ax.scatter(r_all, T_all, L_all, color='red', s=50, label='Original data')

ax.set_xlabel('Days')
ax.set_ylabel('Temperature')
ax.set_zlabel('Number of Internodes')
ax.set_title('RBF Interpolation Fit')
ax.legend()

plt.show()

# Dataset parameters
T = 100  # Number of time steps
num_examples = 100  # Number of temperature samples
temp_min = 20.0
temp_max = 40.0

# Generate time grid (float64)
time_grid = np.arange(T, dtype=np.float64)

# Randomly sample temperatures in [temp_min, temp_max]
temperatures = temp_min + (temp_max - temp_min) * np.random.rand(num_examples)

# Preallocate array for curves: shape (num_examples, T)
curves = np.empty((num_examples, T), dtype=np.float64)

print("Building dataset...")

# For each temperature, evaluate the interpolator along the time grid
for i, temp in enumerate(temperatures):
    # Create input points (r, T) for all times with current temperature
    input_points = np.column_stack((time_grid, np.full_like(time_grid, temp)))

    # Evaluate RBF interpolator (float64)
    curves[i] = rbf(input_points)

print("Dataset built!")

# Save dataset to disk
np.savez(
    'dataset_offline.npz',
    times=time_grid,
    temperatures=temperatures,
    curves=curves
)

print("Dataset saved to 'dataset_offline.npz'")
