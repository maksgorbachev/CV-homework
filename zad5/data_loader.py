"""
Модуль для загрузки и предобработки данных ModelNet10
"""
import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torch


def read_off_file(file_path):
    """
    Читает .off файл и возвращает вершины и грани
    """
    with open(file_path, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')

        n_verts, n_faces, _ = tuple([int(s) for s in f.readline().strip().split(' ')])

        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in f.readline().strip().split(' ')])

        faces = []
        for i_face in range(n_faces):
            face = [int(s) for s in f.readline().strip().split(' ')][1:]
            faces.append(face)

    return np.array(verts), np.array(faces)


def sample_points_from_mesh(vertices, faces, num_points=1024):
    """
    Семплирует точки с поверхности меша
    """
    # Создаем mesh с помощью Open3D
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Семплируем точки
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)

    return points


def normalize_point_cloud(points):
    """
    Нормализует облако точек:
    1. Центрирует по среднему
    2. Масштабирует в единичную сферу [-1, 1]
    """
    # Центрируем
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # Масштабируем
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / furthest_distance

    return points


class ModelNet10Dataset(Dataset):
    """
    Dataset для ModelNet10
    """
    def __init__(self, root_dir, split='train', num_points=1024, augment=False):
        """
        Args:
            root_dir: путь к директории ModelNet10
            split: 'train' или 'test'
            num_points: количество точек для сэмплирования
            augment: применять ли аугментацию
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment

        # Получаем список категорий
        self.categories = sorted([d for d in os.listdir(root_dir)
                                if os.path.isdir(os.path.join(root_dir, d))
                                and d != '__MACOSX'])

        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # Собираем все файлы
        self.files = []
        self.labels = []

        for cat in self.categories:
            cat_dir = os.path.join(root_dir, cat, split)
            if not os.path.exists(cat_dir):
                continue

            files_in_cat = [f for f in os.listdir(cat_dir) if f.endswith('.off')]

            for file in files_in_cat:
                self.files.append(os.path.join(cat_dir, file))
                self.labels.append(self.category_to_idx[cat])

        print(f'Loaded {len(self.files)} files from {split} split')
        print(f'Categories: {self.categories}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Читаем файл
        file_path = self.files[idx]
        label = self.labels[idx]

        # Загружаем меш и семплируем точки
        vertices, faces = read_off_file(file_path)
        points = sample_points_from_mesh(vertices, faces, self.num_points)

        # Нормализуем
        points = normalize_point_cloud(points)

        # Аугментация (если нужна)
        if self.augment and self.split == 'train':
            # Случайное вращение вокруг оси Y (вертикальной)
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            points = points @ rotation_matrix.T

            # Добавление гауссовского шума
            noise = np.random.normal(0, 0.02, points.shape)
            points = points + noise

        # Конвертируем в тензоры
        points = torch.from_numpy(points).float()
        label = torch.tensor(label).long()

        return points, label


def get_data_loaders(root_dir, batch_size=32, num_points=1024, num_workers=4):
    """
    Создает data loaders для train и test
    """
    train_dataset = ModelNet10Dataset(
        root_dir=root_dir,
        split='train',
        num_points=num_points,
        augment=True
    )

    test_dataset = ModelNet10Dataset(
        root_dir=root_dir,
        split='test',
        num_points=num_points,
        augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.categories


if __name__ == '__main__':
    # Тестирование
    train_loader, test_loader, categories = get_data_loaders(
        root_dir='ModelNet10',
        batch_size=8,
        num_points=1024
    )

    print(f'\nКатегории: {categories}')
    print(f'Количество категорий: {len(categories)}')
    print(f'Train batches: {len(train_loader)}')
    print(f'Test batches: {len(test_loader)}')

    # Проверяем первый batch
    for points, labels in train_loader:
        print(f'\nФорма points: {points.shape}')
        print(f'Форма labels: {labels.shape}')
        print(f'Диапазон координат: [{points.min():.3f}, {points.max():.3f}]')
        break
