import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import *
from model_lsx import *

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
# print(device)
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

    def get_age_groups(self, window=12, step=6):
        groups = []
        max_age = 192
        for _, age in self.data:
            # 计算每个年龄可能属于的多个组
            group_indices = []
            for start in range(0, max_age, step):
                end = start + window
                if start <= age < end:
                    group_indices.append(start // step)
            groups.append(np.random.choice(group_indices) if group_indices else 15)
        return np.array(groups)


# SE模块定义
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 方便地将不同尺寸的输入特征图统一到指定的大小
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# SEBottleneck定义
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=dilation, dilation=dilation,
                               groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes * self.expansion, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 完整模型定义
class ResNetSEReg(nn.Module):
    def __init__(self, block, layers, num_classes=1, pretrained=False):
        super(ResNetSEReg, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        norm_layer = nn.BatchNorm2d

        # 输入层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(0.2)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # 修改输出层
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.3),  # 添加Dropout
        #     nn.Linear(512 * block.expansion, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)
        # )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights('model/resnet50-19c8e357.pth')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _load_pretrained_weights(self, weight_path):
        """加载本地预训练权重"""
        pretrain_dict = torch.load(weight_path, weights_only=False)

        # 过滤不匹配的键
        model_dict = self.state_dict()
        # 特殊处理conv1的权重（适应灰度图输入时可能需要删除此判断）
        if model_dict['conv1.weight'].shape[1] != pretrain_dict['conv1.weight'].shape[1]:
            pretrain_dict.pop('conv1.weight')

        # 通用过滤规则
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if k in model_dict and model_dict[k].shape == v.shape}

        # 加载可匹配的参数
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        print(f"成功加载 {len(pretrain_dict)}/{len(model_dict)} 个参数")

        # 初始化新添加的SE模块参数
        for m in self.modules():
            if isinstance(m, SELayer):
                nn.init.kaiming_normal_(m.fc[0].weight)
                nn.init.kaiming_normal_(m.fc[2].weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


# 模型构建函数
def resnet50_se_reg(pretrained=False):
    model = ResNetSEReg(SEBottleneck, [3, 4, 6, 3], pretrained=pretrained)
    return model


# 训练函数
def train_model(model, criterion, optimizer, train_loader, val_loader, train_mean, train_std,
                num_epochs, pati, writer):
    train_losses = []
    val_losses = []
    val_maes = []
    best_val_mae = float('inf')
    no_improve = 0
    # 标志变量，用于记录 val_loss 是否首次低于 20
    first_below_20 = False

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
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # 更新进度条的损失
            train_loader_tqdm.set_postfix(loss=running_loss / ((train_loader_tqdm.n + 1) * inputs.size(0)))

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        # 验证阶段
        val_mae, val_loss = validate_model(model, criterion, val_loader, train_mean, train_std)
        # scheduler_red.step(val_loss)
        if val_mae < 20 and not first_below_20:
            # 首次 val_mae 低于 20，将学习率减半
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.01
            first_below_20 = True
            scheduler_red.step(val_loss)
        elif first_below_20:
            # 后续 val_loss 低于 20 时，使用 scheduler_red 调整学习率
            scheduler_red.step(val_loss)
        else:
            # val_loss 未低于 20 时，使用 scheduler_cos 调整学习率
            scheduler_cos.step()
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
        # for name, param in model.named_parameters():
        #     layer, attr = os.path.splitext(name)
        #     attr = attr[1:]
        #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} months')
        print('-' * 60)
        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model, 'best_model_resnet50se_head.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= pati:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    # # 绘制损失曲线
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
    # 替换代码中的硬编码参数为配置项
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    train_txt = "./annotations/train_correct.txt"
    train_img_dir = "./dog_face_train"
    val_txt = "./annotations/val.txt"
    val_img_dir = "./valset"
    pati = 10
    same_seeds(522)
    # 计算训练集的均值和标准差
    train_mean, train_std = age_mean_std(train_txt)

    # 获取数据转换
    # train_transform, val_transform = get_transforms(MAX_DIM)
    train_transform, val_transform = get_transforms_crop()
    # 创建数据集和数据加载器
    train_dataset = DogAgeDataset(train_txt, train_img_dir,
                                  transform=train_transform,
                                  mean=train_mean, std=train_std)
    val_dataset = DogAgeDataset(val_txt, val_img_dir,
                                transform=val_transform,
                                mean=train_mean, std=train_std)



    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


    # 日志
    log_dir = 'logs'
    time_now = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    writer = SummaryWriter(log_dir=os.path.join(
        log_dir, 'Resnet50_SE_head', time_now))

    # 初始化模型
    model = resnet50_se_reg(pretrained=True).to(device)

    # model = ResNetCBAMReg(pretrained=True).to(device)
    criterion = nn.L1Loss()
    # criterion = HybridLoss(alpha=0.8)

    # 修改优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # optimizer = optim.AdamW([
    #     {'params': model.conv1.parameters(), 'lr': 1e-6},  # 底层小学习率
    #     {'params': model.layer3.parameters()},
    #     {'params': model.layer4.parameters()},
    #     {'params': model.fc.parameters(), 'lr': 1e-3}  # 顶层大学习率
    # ], lr=1e-4, weight_decay=1e-4)
    # 学习率调度器
    scheduler_red = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2)

    scheduler_cos = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 周期长度
        T_mult=2,  # 周期倍增系数
        eta_min=1e-6  # 最小学习率
    )

    # 训练模型
    train_model(model, criterion, optimizer, train_loader, val_loader,
                train_mean, train_std, NUM_EPOCHS, pati, writer)
    writer.close()
