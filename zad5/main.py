"""
Главный скрипт для выполнения практического задания 5
Классификация 3D-объектов с использованием PointNet на датасете ModelNet10
"""
import os
import sys
import argparse
import torch

from pointnet import PointNetClassification, PointNetLoss
from data_loader import get_data_loaders
from train import main as train_main
from visualize import main as visualize_main


def check_dependencies():
    """
    Проверяет наличие необходимых библиотек
    """
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'open3d': 'Open3D',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)

    if missing:
        print(f'Ошибка: Не установлены следующие пакеты: {", ".join(missing)}')
        print('Установите их с помощью: pip install -r requirements.txt')
        return False

    return True


def test_data_loader():
    """
    Тестирование загрузчика данных
    """
    print('\n' + '='*80)
    print('ТЕСТИРОВАНИЕ ЗАГРУЗЧИКА ДАННЫХ')
    print('='*80 + '\n')

    train_loader, test_loader, categories = get_data_loaders(
        root_dir='ModelNet10',
        batch_size=8,
        num_points=1024,
        num_workers=0
    )

    print(f'✓ Загрузчики созданы успешно')
    print(f'  Количество классов: {len(categories)}')
    print(f'  Классы: {", ".join(categories)}')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Test batches: {len(test_loader)}')

    # Тестовый batch
    points, labels = next(iter(train_loader))
    print(f'\n✓ Тестовый batch загружен')
    print(f'  Points shape: {points.shape}')
    print(f'  Labels shape: {labels.shape}')


def test_model():
    """
    Тестирование модели PointNet
    """
    print('\n' + '='*80)
    print('ТЕСТИРОВАНИЕ МОДЕЛИ POINTNET')
    print('='*80 + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')

    # Создание модели
    model = PointNetClassification(
        num_classes=10,
        use_input_transform=True,
        use_feature_transform=True,
        dropout=0.3
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'\n✓ Модель создана')
    print(f'  Количество параметров: {num_params:,}')

    # Тестовый forward pass
    batch_size = 4
    num_points = 1024
    x = torch.randn(batch_size, num_points, 3).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)

    outputs, feature_transform = model(x)

    print(f'\n✓ Forward pass успешен')
    print(f'  Input: {x.shape}')
    print(f'  Output: {outputs.shape}')
    print(f'  Feature transform: {feature_transform.shape}')

    # Тестовый backward pass
    criterion = PointNetLoss()
    loss, cls_loss, reg_loss = criterion(outputs, labels, feature_transform)
    loss.backward()

    print(f'\n✓ Backward pass успешен')
    print(f'  Total loss: {loss.item():.4f}')
    print(f'  Classification loss: {cls_loss.item():.4f}')
    print(f'  Regularization loss: {reg_loss.item():.4f}')


def run_training():
    """
    Запуск обучения модели
    """
    print('\n' + '='*80)
    print('ЗАПУСК ОБУЧЕНИЯ')
    print('='*80)
    train_main()


def run_visualization():
    """
    Запуск визуализации результатов
    """
    print('\n' + '='*80)
    print('ЗАПУСК ВИЗУАЛИЗАЦИИ')
    print('='*80)
    visualize_main()


def main():
    """
    Главная функция
    """
    parser = argparse.ArgumentParser(description='PointNet для классификации 3D объектов')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'train', 'visualize', 'test'],
                       help='Режим работы: all (обучение + визуализация), train (только обучение), '
                            'visualize (только визуализация), test (тестирование)')

    args = parser.parse_args()

    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)

    # Проверка наличия датасета
    if not os.path.exists('ModelNet10'):
        print('Ошибка: Датасет ModelNet10 не найден!')
        print('Распакуйте ModelNet10.zip в текущую директорию.')
        sys.exit(1)

    print('\n' + '='*80)
    print('ПРАКТИЧЕСКОЕ ЗАДАНИЕ 5')
    print('Классификация 3D-объектов на основе облака точек')
    print('с использованием архитектуры PointNet и датасета ModelNet10')
    print('='*80)

    try:
        if args.mode == 'test':
            # Тестирование компонентов
            test_data_loader()
            test_model()

        elif args.mode == 'train':
            # Только обучение
            run_training()

        elif args.mode == 'visualize':
            # Только визуализация
            run_visualization()

        else:  # all
            # Полный пайплайн
            print('\nЗапуск полного пайплайна...')

            # 1. Обучение
            run_training()

            # 2. Визуализация
            run_visualization()

        print('\n' + '='*80)
        print('ВЫПОЛНЕНИЕ ЗАВЕРШЕНО!')
        print('='*80 + '\n')

    except KeyboardInterrupt:
        print('\n\nПрервано пользователем.')
        sys.exit(0)

    except Exception as e:
        print(f'\n\nОшибка: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
