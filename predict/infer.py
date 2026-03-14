import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

FPN_CHANNELS = 256
# 新增注意力模块（SE Block）
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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

# 修改后的Bottleneck with SE
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes)  # 添加 SE 模块
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # 应用 SE 模块

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 修改后的ResNet18-SE模型
class ResNet18SE(nn.Module):
    def __init__(self, block=SEBasicBlock, layers=[2, 2, 2, 2]):
        super(ResNet18SE, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)

        # 初始化权重
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)
        return self.fc(x)


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

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights('model/resnet50-0676ba61.pth')

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
        pretrain_dict = torch.load(weight_path)

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
        x = self.fc(x)

        return x




def get_transforms_crop():
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return None, val_transform

def age_mean_std(txt_file):
    ages = []
    with open(txt_file, 'r') as f:
        for line in f:
            _, age = line.strip().split('\t')
            ages.append(int(age))
    return np.mean(ages), np.std(ages)

if __name__ == "__main__":

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = torch.load('../model/best_model_resnet18.pth', map_location=device, weights_only=False)

    model.eval()

    # 获取数据预处理和统计量
    train_mean, train_std = age_mean_std('../annotations/train_correct.txt')
    _, val_transform = get_transforms_crop()

    # 处理验证集并生成预测结果
    pred_lines = []
    a=0
    with open('../annotations/val.txt', 'r') as f:
        for line in tqdm(f, desc='Processing images'):
            img_name_orig, _ = line.strip().split('\t')
            img_name = img_name_orig.replace('A*', 'A_')

            # 加载和预处理图像
            img_path = os.path.join('../valset', img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                tensor_img = val_transform(img).unsqueeze(0).to(device)

                # 执行预测
                with torch.no_grad():
                    pred = model(tensor_img).item()
                    pred_age = pred * train_std + train_mean  # 反归一化
                    pred_age = int(round(pred_age))  # 四舍五入取整

                # 构建新行
                new_line = f"{img_name_orig}\t{pred_age}\n"
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                a= a+1
                new_line = line  # 保留原始行作为容错

            pred_lines.append(new_line)

    # 保存预测结果
    with open('pred_result.txt', 'w') as f:
        f.writelines(pred_lines)

    print("预测完成！结果已保存到 pred_result.txt")
    print(a)