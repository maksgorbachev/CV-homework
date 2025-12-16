import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

np.random.seed(42)

points = np.random.uniform(0, 100, size=(10000, 3))
xyz = points[:, :3]

np.savetxt("cloud.txt", xyz)

scalar_field_initial = xyz[:, 2] + np.random.normal(0, 5, size=xyz.shape[0])

scalar_1 = np.full(xyz.shape[0], 10.0)
np.savetxt("task1_constant.txt", np.column_stack([xyz, scalar_1]))

scalar_2 = scalar_1.copy()
scalar_2 *= 2.5
np.savetxt("task2_multiplied.txt", np.column_stack([xyz, scalar_2]))

scalar_3 = scalar_2.copy()
scalar_3 += 15
np.savetxt("task3_added.txt", np.column_stack([xyz, scalar_3]))

scalar_4 = scalar_field_initial.copy()
indices = np.argsort(xyz[:, 2])
scalar_sorted = scalar_4[indices]
smoothed_sorted = gaussian_filter1d(scalar_sorted, sigma=2)
scalar_4_smoothed = np.empty_like(scalar_4)
scalar_4_smoothed[indices] = smoothed_sorted
np.savetxt("task4_gaussian.txt", np.column_stack([xyz, scalar_4_smoothed]))

scalar_5 = scalar_field_initial.copy()
indices = np.argsort(xyz[:, 2])
scalar_sorted = scalar_5[indices]
gradient = np.gradient(scalar_sorted)
scalar_5_gradient = np.empty_like(scalar_5)
scalar_5_gradient[indices] = gradient
np.savetxt("task5_gradient.txt", np.column_stack([xyz, scalar_5_gradient]))

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

scalar_6 = scalar_field_initial.copy()
indices = np.argsort(xyz[:, 2])
scalar_sorted = scalar_6[indices]
smoothed_ma = moving_average(scalar_sorted, window_size=10)
scalar_6_ma = np.empty_like(scalar_6)
scalar_6_ma[indices] = smoothed_ma
np.savetxt("task6_moving_average.txt", np.column_stack([xyz, scalar_6_ma]))

scalar_7 = scalar_field_initial.copy()
normed = (scalar_7 - scalar_7.min()) / (scalar_7.max() - scalar_7.min())
colors = plt.cm.viridis(normed)[:, :3]
np.savetxt("task7_colors.txt", np.column_stack([xyz, colors]))

scalar_8 = scalar_field_initial.copy()
mean_val = scalar_8.mean()
std_val = scalar_8.std()
min_val = scalar_8.min()
max_val = scalar_8.max()
median_val = np.median(scalar_8)

stats = {
    "Mean": mean_val,
    "Std": std_val,
    "Min": min_val,
    "Max": max_val,
    "Median": median_val
}

with open("task8_statistics.txt", "w") as f:
    for key, value in stats.items():
        f.write(f"{key}: {value:.4f}\n")

scalar_9 = scalar_field_initial.copy()
normed_9 = (scalar_9 - min_val) / (max_val - min_val)
np.savetxt("task9_normalized.txt", np.column_stack([xyz, normed_9]))

scalar_10 = scalar_field_initial.copy()
nan_indices = np.random.choice(len(scalar_10), size=500, replace=False)
scalar_10_with_nan = scalar_10.copy()
scalar_10_with_nan[nan_indices] = np.nan

def interpolate_nan(data):
    nans = np.isnan(data)
    if not nans.any():
        return data
    x = np.arange(len(data))
    interp_func = interp1d(x[~nans], data[~nans], bounds_error=False, fill_value="extrapolate")
    return interp_func(x)

scalar_10_filled = interpolate_nan(scalar_10_with_nan)
np.savetxt("task10_interpolated.txt", np.column_stack([xyz, scalar_10_filled]))

scalar_11 = scalar_field_initial.copy()
threshold_min = np.percentile(scalar_11, 25)
threshold_max = np.percentile(scalar_11, 75)
mask = (scalar_11 >= threshold_min) & (scalar_11 <= threshold_max)
filtered_points_11 = xyz[mask]
filtered_scalar_11 = scalar_11[mask]
np.savetxt("task11_filtered.txt", np.column_stack([filtered_points_11, filtered_scalar_11]))

scalar_12 = normed_9.copy()
xyz_12 = xyz.copy()
xyz_12[:, 2] = scalar_12 * 100
np.savetxt("task12_scalar_as_z.txt", xyz_12)

scalar_13 = None

def visualize_scalar_field(xyz, scalar, title, filename, cmap='viridis'):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    normed = (scalar - scalar.min()) / (scalar.max() - scalar.min())
    colors = plt.cm.get_cmap(cmap)(normed)

    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        c=scalar, cmap=cmap, s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Scalar Value')

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

visualize_scalar_field(xyz, scalar_1, "Task 1: Constant Scalar Field", "task1_constant.png")
visualize_scalar_field(xyz, scalar_2, "Task 2: Multiplied Scalar Field", "task2_multiplied.png")
visualize_scalar_field(xyz, scalar_3, "Task 3: Added Scalar Field", "task3_added.png")
visualize_scalar_field(xyz, scalar_4_smoothed, "Task 4: Gaussian Filtered", "task4_gaussian.png")
visualize_scalar_field(xyz, scalar_5_gradient, "Task 5: Gradient", "task5_gradient.png", cmap='coolwarm')
visualize_scalar_field(xyz, scalar_6_ma, "Task 6: Moving Average", "task6_moving_average.png")
visualize_scalar_field(xyz, normed_9, "Task 9: Normalized [0,1]", "task9_normalized.png")
visualize_scalar_field(filtered_points_11, filtered_scalar_11, "Task 11: Filtered Points", "task11_filtered.png")
visualize_scalar_field(xyz_12, xyz_12[:, 2], "Task 12: Scalar as Z-coordinate", "task12_scalar_as_z.png", cmap='plasma')

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=1, alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Task 7: RGB Colors from Scalar Field')
plt.savefig("task7_colors.png", dpi=150, bbox_inches='tight')
plt.close()

print("Task 1: Constant scalar field created")
print(f"  Value: 10.0, Points: {len(scalar_1)}")
print()

print("Task 2: Multiplied scalar field")
print(f"  Multiplier: 2.5, Result range: [{scalar_2.min():.2f}, {scalar_2.max():.2f}]")
print()

print("Task 3: Added to scalar field")
print(f"  Added value: 15, Result range: [{scalar_3.min():.2f}, {scalar_3.max():.2f}]")
print()

print("Task 4: Gaussian filter applied")
print(f"  Sigma: 2, Original range: [{scalar_4.min():.2f}, {scalar_4.max():.2f}]")
print(f"  Smoothed range: [{scalar_4_smoothed.min():.2f}, {scalar_4_smoothed.max():.2f}]")
print()

print("Task 5: Gradient computed")
print(f"  Gradient range: [{scalar_5_gradient.min():.2f}, {scalar_5_gradient.max():.2f}]")
print(f"  Mean gradient: {scalar_5_gradient.mean():.4f}")
print()

print("Task 6: Moving average applied")
print(f"  Window size: 10, Result range: [{scalar_6_ma.min():.2f}, {scalar_6_ma.max():.2f}]")
print()

print("Task 7: RGB colors generated")
print(f"  Color map: viridis, Colors shape: {colors.shape}")
print()

print("Task 8: Statistics computed")
for key, value in stats.items():
    print(f"  {key}: {value:.4f}")
print()

print("Task 9: Normalized to [0, 1]")
print(f"  Range: [{normed_9.min():.4f}, {normed_9.max():.4f}]")
print()

print("Task 10: Interpolated missing values")
print(f"  NaN count before: {np.isnan(scalar_10_with_nan).sum()}")
print(f"  NaN count after: {np.isnan(scalar_10_filled).sum()}")
print()

print("Task 11: Filtered by scalar value")
print(f"  Threshold: [{threshold_min:.2f}, {threshold_max:.2f}]")
print(f"  Points before: {len(xyz)}, Points after: {len(filtered_points_11)}")
print(f"  Retention rate: {len(filtered_points_11)/len(xyz)*100:.2f}%")
print()

print("Task 12: Scalar field used as Z-coordinate")
print(f"  New Z range: [{xyz_12[:, 2].min():.2f}, {xyz_12[:, 2].max():.2f}]")
print()

print("Task 13: Scalar field deleted (set to None)")
print()

print("All tasks completed successfully!")
