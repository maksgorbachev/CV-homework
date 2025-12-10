import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_ply(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[2])
        if line.strip() == 'end_header':
            header_end = i + 1
            break

    points = []
    for i in range(header_end, header_end + num_vertices):
        coords = lines[i].strip().split()
        points.append([float(coords[0]), float(coords[1]), float(coords[2])])

    return np.array(points)


def random_subsampling(points, n_samples):
    indices = np.random.choice(points.shape[0], n_samples, replace=False)
    return points[indices]


def voxel_grid_subsampling(points, voxel_size):
    coords = (points / voxel_size).astype(int)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]


def farthest_point_sampling(points, n_samples):
    n_points = points.shape[0]
    selected_indices = np.zeros(n_samples, dtype=int)
    distances = np.full(n_points, np.inf)

    selected_indices[0] = np.random.randint(n_points)

    for i in range(1, n_samples):
        last_selected = selected_indices[i-1]
        dist_to_last = np.sum((points - points[last_selected])**2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected_indices[i] = np.argmax(distances)

    return points[selected_indices]


def visualize_point_clouds(original, random_sub, voxel_sub, fps_sub):
    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=0.1)
    ax1.set_title(f'Original ({len(original)} points)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(random_sub[:, 0], random_sub[:, 1], random_sub[:, 2], c='red', s=1)
    ax2.set_title(f'Random ({len(random_sub)} points)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(voxel_sub[:, 0], voxel_sub[:, 1], voxel_sub[:, 2], c='green', s=1)
    ax3.set_title(f'Voxel Grid ({len(voxel_sub)} points)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(fps_sub[:, 0], fps_sub[:, 1], fps_sub[:, 2], c='purple', s=1)
    ax4.set_title(f'FPS ({len(fps_sub)} points)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("Loading point cloud...")
    points = load_ply('../zad1/bun315.ply')
    print(f"Loaded {len(points)} points")
    print(f"Point cloud bounds: X[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}], "
          f"Y[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}], "
          f"Z[{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")

    n_samples_random = 5000
    voxel_size = 0.002
    n_samples_fps = 1000

    print("\n" + "="*60)
    print("Random Subsampling")
    print("="*60)
    start_time = time.time()
    subsampled_random = random_subsampling(points, n_samples_random)
    random_time = time.time() - start_time
    print(f"Time: {random_time:.4f} seconds")
    print(f"Output points: {len(subsampled_random)}")
    np.savetxt('subsampled_random.xyz', subsampled_random)
    print("Saved to: subsampled_random.xyz")

    print("\n" + "="*60)
    print("Voxel Grid Subsampling")
    print("="*60)
    start_time = time.time()
    subsampled_voxel = voxel_grid_subsampling(points, voxel_size)
    voxel_time = time.time() - start_time
    print(f"Voxel size: {voxel_size}")
    print(f"Time: {voxel_time:.4f} seconds")
    print(f"Output points: {len(subsampled_voxel)}")
    np.savetxt('subsampled_voxel.xyz', subsampled_voxel)
    print("Saved to: subsampled_voxel.xyz")

    print("\n" + "="*60)
    print("Farthest Point Sampling")
    print("="*60)
    start_time = time.time()
    subsampled_fps = farthest_point_sampling(points, n_samples_fps)
    fps_time = time.time() - start_time
    print(f"Time: {fps_time:.4f} seconds")
    print(f"Output points: {len(subsampled_fps)}")
    np.savetxt('subsampled_fps.xyz', subsampled_fps)
    print("Saved to: subsampled_fps.xyz")

    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    print(f"Random Subsampling:      {random_time:.4f}s ({len(subsampled_random)} points)")
    print(f"Voxel Grid Subsampling:  {voxel_time:.4f}s ({len(subsampled_voxel)} points)")
    print(f"Farthest Point Sampling: {fps_time:.4f}s ({len(subsampled_fps)} points)")

    fastest = min(random_time, voxel_time, fps_time)
    if fastest == random_time:
        print("\nFastest method: Random Subsampling")
    elif fastest == voxel_time:
        print("\nFastest method: Voxel Grid Subsampling")
    else:
        print("\nFastest method: Farthest Point Sampling")

    print("\n" + "="*60)
    print("Visualizing results...")
    print("="*60)
    visualize_point_clouds(points, subsampled_random, subsampled_voxel, subsampled_fps)
    print("Visualization saved to: comparison.png")


if __name__ == "__main__":
    main()
