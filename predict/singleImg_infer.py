import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

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

# SE模块定义
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
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation,
                               groups, bias=False)
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

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # face2
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # face3
        # self.fc1 = nn.Linear(512 * block.expansion, 512)
        # self.fc2 = nn.Linear(512, num_classes)

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
        # x = self.fc2(x)

        return x


# 模型构建函数
def resnet50_se_reg(pretrained=False):
    model = ResNetSEReg(SEBottleneck, [3, 4, 6, 3], pretrained=pretrained)
    return model


if __name__ == "__main__":

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = torch.load('../model/best_model_se2_20.5505.pth', map_location=device)

    model.eval()

    # 获取数据预处理和统计量
    train_mean, train_std = age_mean_std('../annotations/train_correct.txt')
    _, val_transform = get_transforms_crop()
    img_path = '../redbook_img/2.jpg'
    img = Image.open(img_path).convert('RGB')
    tensor_img = val_transform(img).unsqueeze(0).to(device)

    # 执行预测
    with torch.no_grad():
        pred = model(tensor_img).item()
        pred_age = pred * train_std + train_mean  # 反归一化
        pred_age = round(pred_age,1)

    print(f"预测年龄：{pred_age} 个月")