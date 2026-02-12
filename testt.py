#!/usr/bin/env python3
# testt.py (train and evaluate CIFAR-10)

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def main():
    # 1. 配置
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size  = 1024  # 可根据显存调整
    num_epochs  = 10
    lr          = 1e-3
    num_workers = 4

    # 2. 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # 3. 加载数据集
    train_set = datasets.CIFAR10(
        root='data', train=True, download=False, transform=transform
    )
    test_set = datasets.CIFAR10(
        root='data', train=False, download=False, transform=transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 4. 模型、优化器
    model     = models.resnet18(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. 训练与评估循环
    for epoch in range(1, num_epochs + 1):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)                 # [batch_size, 10]
            loss = F.cross_entropy(logits, labels) # 逐样本交叉熵
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- 测试阶段 ---
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, labels, reduction='sum')
                test_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_test_loss = test_loss / total
        test_acc = correct / total

        print(f"Epoch {epoch}/{num_epochs}  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Test Loss: {avg_test_loss:.4f}  "
              f"Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main()
