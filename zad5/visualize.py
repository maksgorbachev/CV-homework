"""
Модуль для визуализации результатов обучения и предсказаний PointNet
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import open3d as o3d
from tqdm import tqdm

from pointnet import PointNetClassification
from data_loader import ModelNet10Dataset


def plot_training_history(history_path, save_path=None):
    """
    Строит графики изменения loss и accuracy во время обучения
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 0].plot(history['learning_rate'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Сводка
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1

    summary_text = f"""
    Итоговые результаты:

    • Лучшая Val Accuracy: {best_val_acc:.2f}%
      (эпоха {best_epoch})

    • Финальная Train Accuracy: {final_train_acc:.2f}%
    • Финальная Val Accuracy: {final_val_acc:.2f}%

    • Всего эпох: {len(history['train_loss'])}
    """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'График сохранён: {save_path}')

    plt.show()


def evaluate_model(model, test_loader, categories, device):
    """
    Оценка модели на тестовом датасете
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print('Оценка модели на тестовом датасете...')

    with torch.no_grad():
        for points, labels in tqdm(test_loader):
            points = points.to(device)
            labels = labels.to(device)

            outputs, _ = model(points)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuracy
    accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f'\nОбщая точность: {accuracy:.2f}%')

    # Classification Report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=categories))

    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, categories, save_path=None):
    """
    Строит confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    # Нормализованная confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories,
                yticklabels=categories, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories,
                yticklabels=categories, ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix сохранена: {save_path}')

    plt.show()

    return cm


def plot_per_class_accuracy(y_true, y_pred, categories, save_path=None):
    """
    Строит график точности по классам
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Сортируем по точности
    sorted_indices = np.argsort(per_class_acc)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn(per_class_acc[sorted_indices])
    bars = plt.barh(range(len(categories)), per_class_acc[sorted_indices] * 100, color=colors)

    plt.yticks(range(len(categories)), [categories[i] for i in sorted_indices])
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # Добавляем значения на барах
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', ha='left', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'График точности по классам сохранён: {save_path}')

    plt.show()


def visualize_point_clouds_with_predictions(model, dataset, categories, device, num_samples=10, save_dir=None):
    """
    Визуализирует облака точек с предсказаниями и истинными метками
    """
    model.eval()

    # Выбираем случайные примеры
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 4 * ((num_samples + 4) // 5)))

    for i, idx in enumerate(indices):
        points, label = dataset[idx]
        points_batch = points.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs, _ = model(points_batch)
            probs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(1).item()
            confidence = probs[0, pred].item()

        points_np = points.cpu().numpy()

        # 3D scatter plot
        ax = fig.add_subplot((num_samples + 4) // 5, 5, i + 1, projection='3d')

        # Цвет по высоте (z-координата)
        colors = points_np[:, 2]

        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                  c=colors, cmap='viridis', s=1, alpha=0.6)

        true_label = categories[label]
        pred_label = categories[pred]

        title_color = 'green' if pred == label else 'red'
        title = f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})'

        ax.set_title(title, fontsize=10, color=title_color, fontweight='bold')
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, 'point_cloud_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Визуализация облаков точек сохранена: {save_path}')

    plt.show()


def visualize_3d_point_cloud(points, title="Point Cloud", window_name="Point Cloud"):
    """
    Визуализация облака точек в 3D с помощью Open3D
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Раскрашиваем по высоте
    colors = np.zeros_like(points)
    z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
    colors[:, 0] = z_normalized  # R
    colors[:, 1] = 1 - z_normalized  # G
    colors[:, 2] = 0.5  # B
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def load_model(checkpoint_path, device):
    """
    Загружает обученную модель из чекпоинта
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = PointNetClassification(
        num_classes=checkpoint['config']['num_classes'],
        use_input_transform=checkpoint['config']['use_input_transform'],
        use_feature_transform=checkpoint['config']['use_feature_transform'],
        dropout=checkpoint['config']['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    categories = checkpoint['categories']

    print(f'Модель загружена из {checkpoint_path}')
    print(f'Эпоха: {checkpoint["epoch"] + 1}')
    print(f'Val Accuracy: {checkpoint["val_acc"]:.2f}%')

    return model, categories


def main():
    """
    Основная функция для визуализации результатов
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Пути
    checkpoint_path = 'checkpoints/best_model.pth'
    history_path = 'checkpoints/training_history.json'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 1. Визуализация истории обучения
    print('1. Построение графиков обучения...')
    plot_training_history(
        history_path,
        save_path=os.path.join(results_dir, 'training_history.png')
    )

    # 2. Загрузка модели
    print('\n2. Загрузка обученной модели...')
    model, categories = load_model(checkpoint_path, device)

    # 3. Загрузка тестовых данных
    print('\n3. Загрузка тестовых данных...')
    test_dataset = ModelNet10Dataset(
        root_dir='ModelNet10',
        split='test',
        num_points=1024,
        augment=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # 4. Оценка модели
    print('\n4. Оценка модели...')
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, categories, device)

    # 5. Confusion Matrix
    print('\n5. Построение Confusion Matrix...')
    plot_confusion_matrix(
        y_true, y_pred, categories,
        save_path=os.path.join(results_dir, 'confusion_matrix.png')
    )

    # 6. Per-Class Accuracy
    print('\n6. Построение графика точности по классам...')
    plot_per_class_accuracy(
        y_true, y_pred, categories,
        save_path=os.path.join(results_dir, 'per_class_accuracy.png')
    )

    # 7. Визуализация предсказаний
    print('\n7. Визуализация облаков точек с предсказаниями...')
    visualize_point_clouds_with_predictions(
        model, test_dataset, categories, device,
        num_samples=15,
        save_dir=results_dir
    )

    print(f'\n{"="*60}')
    print('Визуализация завершена!')
    print(f'Результаты сохранены в: {results_dir}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
