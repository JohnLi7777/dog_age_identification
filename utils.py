import numpy as np
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2



def get_transforms_crop():
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 缩放短边到256
        transforms.RandomCrop(224),  # 随机裁剪
        transforms.RandomApply([  # 随机选择增强组合
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.2),
            # transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=5)
        ], p=0.7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# 计算训练集的均值和标准差
def age_mean_std(txt_file):
    ages = []
    with open(txt_file, 'r') as f:
        for line in f:
            _, age = line.strip().split('\t')
            ages.append(int(age))
    ages = np.array(ages)
    return np.mean(ages), np.std(ages)


def plot_curves(train_losses, val_losses, val_maes):
    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig('loss_curves_se.png')
    plt.show()

    # 绘制 MAE 曲线
    plt.figure()
    plt.plot(val_maes, label='Validation MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (months)')
    plt.title('Validation MAE Curve')
    plt.legend()
    plt.savefig('mae_curve_se.png')
    plt.show()


# 混合损失函数
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


