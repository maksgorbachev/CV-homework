"""
Скрипт для обучения PointNet на датасете ModelNet10
"""
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from pointnet import PointNetClassification, PointNetLoss
from data_loader import get_data_loaders


class Trainer:
    """
    Класс для обучения PointNet
    """
    def __init__(self, model, train_loader, test_loader, categories, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.categories = categories
        self.config = config

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f'Используется устройство: {self.device}')

        # Loss and optimizer
        self.criterion = PointNetLoss(feature_transform_reg_weight=config['reg_weight'])
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999)
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config['scheduler_step'],
            gamma=config['scheduler_gamma']
        )

        # Директория для сохранения результатов
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        # История обучения
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """
        Обучение одной эпохи
        """
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Train]')

        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs, feature_transform = self.model(points)

            # Compute loss
            loss, cls_loss, reg_loss = self.criterion(outputs, labels, feature_transform)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Статистика
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Обновление progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # Средние значения за эпоху
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_reg_loss = total_reg_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc, avg_cls_loss, avg_reg_loss

    def validate(self, epoch):
        """
        Валидация модели
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Для confusion matrix
        all_preds = []
        all_labels = []

        pbar = tqdm(self.test_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Val]  ')

        with torch.no_grad():
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs, feature_transform = self.model(points)

                # Compute loss
                loss, _, _ = self.criterion(outputs, labels, feature_transform)

                # Статистика
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Обновление progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        # Средние значения
        avg_loss = total_loss / len(self.test_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)

    def train(self):
        """
        Полный цикл обучения
        """
        print(f'\n{"="*60}')
        print(f'Начало обучения')
        print(f'{"="*60}')
        print(f'Количество эпох: {self.config["num_epochs"]}')
        print(f'Batch size: {self.config["batch_size"]}')
        print(f'Learning rate: {self.config["learning_rate"]}')
        print(f'Количество точек: {self.config["num_points"]}')
        print(f'{"="*60}\n')

        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc, cls_loss, reg_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, all_preds, all_labels = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # Вывод результатов эпохи
            print(f'\nEpoch {epoch+1}/{self.config["num_epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f} (cls: {cls_loss:.4f}, reg: {reg_loss:.4f}) | Acc: {train_acc:.2f}%')
            print(f'  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%')
            print(f'  LR: {current_lr:.6f}')

            # Сохранение лучшей модели
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f'  ✓ Новая лучшая модель! Accuracy: {val_acc:.2f}%')

            # Периодическое сохранение
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)

        print(f'\n{"="*60}')
        print(f'Обучение завершено!')
        print(f'Лучшая Val Accuracy: {self.best_val_acc:.2f}%')
        print(f'{"="*60}\n')

        # Сохранение истории
        self.save_history()

        return self.history

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """
        Сохранение чекпоинта модели
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'categories': self.categories
        }

        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, path)
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, path)

    def save_history(self):
        """
        Сохранение истории обучения
        """
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f'История обучения сохранена: {history_path}')


def main():
    """
    Основная функция для запуска обучения
    """
    # Конфигурация
    config = {
        # Данные
        'data_root': 'ModelNet10',
        'num_points': 1024,
        'batch_size': 32,
        'num_workers': 4,

        # Модель
        'num_classes': 10,
        'use_input_transform': True,
        'use_feature_transform': True,
        'dropout': 0.3,

        # Обучение
        'num_epochs': 50,
        'learning_rate': 0.001,
        'reg_weight': 0.001,

        # Scheduler
        'scheduler_step': 20,
        'scheduler_gamma': 0.5,

        # Сохранение
        'save_dir': 'checkpoints',
        'save_interval': 10,
    }

    print('Конфигурация:')
    for key, value in config.items():
        print(f'  {key}: {value}')

    # Загрузка данных
    print('\nЗагрузка данных...')
    train_loader, test_loader, categories = get_data_loaders(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_points=config['num_points'],
        num_workers=config['num_workers']
    )

    # Создание модели
    print('\nСоздание модели...')
    model = PointNetClassification(
        num_classes=config['num_classes'],
        use_input_transform=config['use_input_transform'],
        use_feature_transform=config['use_feature_transform'],
        dropout=config['dropout']
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Количество параметров: {num_params:,}')

    # Создание тренера
    trainer = Trainer(model, train_loader, test_loader, categories, config)

    # Обучение
    history = trainer.train()

    return history


if __name__ == '__main__':
    main()
