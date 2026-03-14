import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
class DynamicPad:
    def __init__(self, max_dim, fill=0):
        self.max_dim = max_dim
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        padded_img = Image.new("RGB", (self.max_dim, self.max_dim), self.fill)
        padded_img.paste(img.resize((new_w, new_h)),
                         ((self.max_dim - new_w) // 2, (self.max_dim - new_h) // 2))
        return padded_img


def get_transforms(max_dim):
    train_transform = transforms.Compose([
        DynamicPad(max_dim=max_dim),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        DynamicPad(max_dim=max_dim),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def get_transforms_crop():
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 缩放短边到256
        transforms.RandomCrop(224),  # 随机裁剪
        transforms.RandomApply([  # 随机选择增强组合
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=5)
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

# 自定义数据集类
class DogAgeDataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None, mean=0.0, std=1.0):
        self.data = []
        self.img_dir = img_dir
        self.transform = transform
        self.mean = mean
        self.std = std

        with open(txt_file, 'r') as f:
            for line in f:
                img_name, age = line.strip().split('\t')
                img_name = img_name.replace('*', '_')
                age = int(age)
                self.data.append((img_name, age))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, age = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 年龄归一化
        age_norm = (age - self.mean) / self.std
        return image, torch.tensor(age_norm, dtype=torch.float32)


# 计算训练集的均值和标准差
def age_mean_std(txt_file):
    ages = []
    with open(txt_file, 'r') as f:
        for line in f:
            _, age = line.strip().split('\t')
            ages.append(int(age))
    ages = np.array(ages)
    return np.mean(ages), np.std(ages)


# 初始化模型
def initialize_model(config):
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model.to(device)

# def plot_curves(train_losses, val_losses, val_maes):
#     # 绘制损失曲线
#     plt.figure()
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss Curves')
#     plt.legend()
#     plt.savefig('loss_curves.png')
#     plt.show()
#
#     # 绘制 MAE 曲线
#     plt.figure()
#     plt.plot(val_maes, label='Validation MAE', color='orange')
#     plt.xlabel('Epoch')
#     plt.ylabel('MAE (months)')
#     plt.title('Validation MAE Curve')
#     plt.legend()
#     plt.savefig('mae_curve.png')
#     plt.show()


# 训练函数
def train_model(model, criterion, optimizer, train_loader, val_loader, train_mean, train_std,
                num_epochs, model_save_path, patience, writer):
    train_losses = []
    val_losses = []
    val_maes = []
    best_val_mae = float('inf')
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], 学习率: {current_lr:.6f}')
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader,
                                 desc=f"Epoch {epoch + 1}/{num_epochs}",
                                 leave=True)
        # 训练阶段
        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度剪枝
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # 更新进度条的损失
            train_loader_tqdm.set_postfix(loss=running_loss / ((train_loader_tqdm.n + 1) * inputs.size(0)))

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        # 验证阶段
        val_mae, val_loss = validate_model(model, criterion, val_loader, train_mean, train_std)
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # 分段MAE评估
        age_groups = validate_age_groups(model, val_loader, train_mean, train_std)
        for k, v in age_groups.items():
            writer.add_scalar(f'MAE/Group_{k}', v, epoch)

        # 在每个epoch结束后记录指标
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} months')
        print('-' * 60)
        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model, model_save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    # 绘制损失曲线
    # plot_curves(train_losses, val_losses, val_maes)


# 验证函数
def validate_model(model, criterion, val_loader, train_mean, train_std):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反归一化
            preds = outputs * train_std + train_mean
            trues = labels * train_std + train_mean

            running_loss += loss.item() * inputs.size(0)
            running_mae += torch.abs(preds - trues).sum().item()

    total_loss = running_loss / len(val_loader.dataset)
    total_mae = running_mae / len(val_loader.dataset)
    return total_mae, total_loss

def validate_age_groups(model, val_loader, mean, std, bins=[24, 48, 72, 96, 120, 144, 168, 192]):
    model.eval()
    # 初始化 mae_dict 以适应新的年龄分组
    mae_dict = {f'<{bins[0]}': 0}
    for i in range(len(bins) - 1):
        mae_dict[f'{bins[i]}-{bins[i + 1]}'] = 0
    mae_dict[f'>{bins[-1]}'] = 0

    # 初始化 count_dict 以适应新的年龄分组
    count_dict = {k: 1e-6 for k in mae_dict.keys()}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            predis = outputs * std + mean
            trues = labels * std + mean

            # 分年龄段统计
            for p, t in zip(predis, trues):
                age = t.item()
                if age < bins[0]:
                    key = f'<{bins[0]}'
                elif age < bins[-1]:
                    for i in range(len(bins) - 1):
                        if bins[i] <= age < bins[i + 1]:
                            key = f'{bins[i]}-{bins[i + 1]}'
                            break
                else:
                    key = f'>{bins[-1]}'

                mae_dict[key] += abs(p.item() - age)
                count_dict[key] += 1

    return {k: v / count_dict[k] for k, v in mae_dict.items()}

if __name__ == '__main__':
    same_seeds(522)
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 替换代码中的硬编码参数为配置项
    BATCH_SIZE = config['hyperparameters']['batch_size']
    NUM_EPOCHS = config['hyperparameters']['num_epochs']
    LEARNING_RATE = config['hyperparameters']['learning_rate']
    train_txt = config['paths']['train_txt']
    train_img_dir = config['paths']['train_img_dir']
    val_txt = config['paths']['val_txt']
    val_img_dir = config['paths']['val_img_dir']
    model_save_path = config['paths']['model_save_path']
    patience = config['training']['patience']
    # 计算训练集的均值和标准差
    train_mean, train_std = age_mean_std(train_txt)
    train_mean = float(train_mean.astype(np.float32))  # 转换为Python float类型
    train_std = float(train_std.astype(np.float32))  # 转换为Python float类型

    # 获取数据转换
    train_transform, val_transform = get_transforms_crop()

    # 创建数据集和数据加载器
    train_dataset = DogAgeDataset(train_txt, train_img_dir,
                                  transform=train_transform,
                                  mean=train_mean, std=train_std)
    val_dataset = DogAgeDataset(val_txt, val_img_dir,
                                transform=val_transform,
                                mean=train_mean, std=train_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=config['data_loader']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=config['data_loader']['num_workers'])

    # 日志
    log_dir = 'logs'
    time_now = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    writer = SummaryWriter(log_dir=os.path.join(
        log_dir, "resnet34_head", time_now))


    # 初始化模型
    model = initialize_model(config)
    # # 添加模型结构可视化
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 根据实际输入尺寸调整
    # writer.add_graph(model, dummy_input)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=config['optimizer']['weight_decay'])
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['optimizer']['scheduler']['mode'],
        factor=config['optimizer']['scheduler']['factor'],
        patience=config['optimizer']['scheduler']['patience'])
    # 训练模型
    train_model(model, criterion, optimizer, train_loader, val_loader,
                train_mean, train_std, NUM_EPOCHS, model_save_path, patience, writer)

    writer.close()