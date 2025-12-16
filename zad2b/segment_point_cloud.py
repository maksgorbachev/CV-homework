import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

points = np.random.uniform(0, 100, size=(100000, 3))
np.savetxt("synthetic_cloud.xyz", points)

def filter_by_bbox(points, xmin, xmax, ymin, ymax, zmin, zmax):
    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )
    return points[mask]

def filter_by_distance(points, center, radius):
    distances = np.linalg.norm(points - center, axis=1)
    return points[distances <= radius]

def filter_by_circle_xy(points, center_xy, radius):
    distances_xy = np.sqrt((points[:, 0] - center_xy[0])**2 + (points[:, 1] - center_xy[1])**2)
    return points[distances_xy <= radius]

def show_cloud(points, title="Point Cloud", filename=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

filtered_bbox = filter_by_bbox(points, 20, 50, 30, 70, 10, 40)
high_points = points[points[:, 2] > 80]
near_center = filter_by_distance(points, center=np.array([50, 50, 50]), radius=25)
circle_xy = filter_by_circle_xy(points, center_xy=np.array([50, 50]), radius=30)

np.savetxt("bbox_filtered.xyz", filtered_bbox)
np.savetxt("high_points.xyz", high_points)
np.savetxt("near_center.xyz", near_center)
np.savetxt("circle_xy_filtered.xyz", circle_xy)

show_cloud(points, "Original Cloud", "original_cloud.png")
show_cloud(filtered_bbox, "BBox Filtered [20,50]x[30,70]x[10,40]", "bbox_filtered.png")
show_cloud(high_points, "High Points (Z > 80)", "high_points.png")
show_cloud(near_center, "Near Center (R=25 from [50,50,50])", "near_center.png")
show_cloud(circle_xy, "Circle XY Filter (R=30 from [50,50])", "circle_xy_filtered.png")

print(f"Original points: {len(points)}")
print(f"BBox filtered: {len(filtered_bbox)}")
print(f"High points: {len(high_points)}")
print(f"Near center: {len(near_center)}")
print(f"Circle XY filtered: {len(circle_xy)}")
