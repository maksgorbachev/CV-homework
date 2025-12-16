"""
Реализация архитектуры PointNet для классификации 3D объектов
Основано на статье: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Transformation Network (T-Net) для выравнивания входных данных
    Предсказывает матрицу трансформации
    """
    def __init__(self, k=3):
        """
        Args:
            k: размерность входа (3 для координат, 64 для признаков)
        """
        super(TNet, self).__init__()
        self.k = k

        # Shared MLPs
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Args:
            x: [B, k, N] - batch точек
        Returns:
            transform: [B, k, k] - матрица трансформации
        """
        batch_size = x.size(0)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Инициализация как единичная матрица
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x


class PointNetBackbone(nn.Module):
    """
    Backbone PointNet для извлечения признаков
    """
    def __init__(self, use_input_transform=True, use_feature_transform=True):
        super(PointNetBackbone, self).__init__()
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

        # Input transformation
        if use_input_transform:
            self.input_transform = TNet(k=3)

        # MLP (64, 64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transformation
        if use_feature_transform:
            self.feature_transform = TNet(k=64)

        # MLP (64, 128, 1024)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        """
        Args:
            x: [B, N, 3] - batch точек
        Returns:
            global_feat: [B, 1024] - глобальный признак
            feature_transform: матрица трансформации признаков (для регуляризации)
        """
        batch_size = x.size(0)
        num_points = x.size(1)

        # Транспонируем для conv1d: [B, N, 3] -> [B, 3, N]
        x = x.transpose(2, 1)

        # Input transformation
        if self.use_input_transform:
            transform = self.input_transform(x)
            x = x.transpose(2, 1)  # [B, 3, N] -> [B, N, 3]
            x = torch.bmm(x, transform)  # [B, N, 3] x [B, 3, 3] = [B, N, 3]
            x = x.transpose(2, 1)  # [B, N, 3] -> [B, 3, N]

        # MLP (64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        feature_transform = None
        if self.use_feature_transform:
            feature_transform = self.feature_transform(x)
            x = x.transpose(2, 1)  # [B, 64, N] -> [B, N, 64]
            x = torch.bmm(x, feature_transform)  # [B, N, 64] x [B, 64, 64] = [B, N, 64]
            x = x.transpose(2, 1)  # [B, N, 64] -> [B, 64, N]

        # MLP (64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling для получения глобального признака
        global_feat = torch.max(x, 2, keepdim=False)[0]  # [B, 1024]

        return global_feat, feature_transform


class PointNetClassification(nn.Module):
    """
    PointNet для классификации
    """
    def __init__(self, num_classes=10, use_input_transform=True, use_feature_transform=True, dropout=0.3):
        super(PointNetClassification, self).__init__()

        self.backbone = PointNetBackbone(
            use_input_transform=use_input_transform,
            use_feature_transform=use_feature_transform
        )

        # Классификатор
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, 3] - batch точек
        Returns:
            scores: [B, num_classes] - логиты классов
            feature_transform: матрица трансформации признаков
        """
        global_feat, feature_transform = self.backbone(x)

        # Классификатор
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x, feature_transform


def feature_transform_regularizer(trans):
    """
    Regularization loss для feature transformation matrix
    Ограничение: матрица должна быть близка к ортогональной
    L = ||I - AA^T||^2
    """
    d = trans.size(1)
    I = torch.eye(d, device=trans.device).unsqueeze(0).repeat(trans.size(0), 1, 1)
    loss = torch.mean(torch.norm(I - torch.bmm(trans, trans.transpose(2, 1)), dim=(1, 2)))
    return loss


class PointNetLoss(nn.Module):
    """
    Loss function для PointNet
    """
    def __init__(self, feature_transform_reg_weight=0.001):
        super(PointNetLoss, self).__init__()
        self.feature_transform_reg_weight = feature_transform_reg_weight

    def forward(self, outputs, labels, feature_transform=None):
        """
        Args:
            outputs: [B, num_classes] - предсказания модели
            labels: [B] - истинные метки
            feature_transform: [B, 64, 64] - матрица трансформации признаков
        """
        # Classification loss
        classification_loss = F.cross_entropy(outputs, labels)

        # Regularization loss
        if feature_transform is not None:
            reg_loss = feature_transform_regularizer(feature_transform)
            total_loss = classification_loss + self.feature_transform_reg_weight * reg_loss
        else:
            reg_loss = torch.tensor(0.0)
            total_loss = classification_loss

        return total_loss, classification_loss, reg_loss


if __name__ == '__main__':
    # Тестирование архитектуры
    print("Тестирование PointNet архитектуры...")

    # Создаем случайный batch
    batch_size = 8
    num_points = 1024
    num_classes = 10

    x = torch.randn(batch_size, num_points, 3)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Создаем модель
    model = PointNetClassification(num_classes=num_classes)
    print(f"\nМодель создана")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    outputs, feature_transform = model(x)
    print(f"\nВход: {x.shape}")
    print(f"Выход: {outputs.shape}")
    print(f"Feature transform: {feature_transform.shape if feature_transform is not None else None}")

    # Loss
    criterion = PointNetLoss()
    loss, cls_loss, reg_loss = criterion(outputs, labels, feature_transform)
    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Classification loss: {cls_loss.item():.4f}")
    print(f"Regularization loss: {reg_loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("\nBackward pass успешен!")

    print("\n✓ Архитектура работает корректно")
