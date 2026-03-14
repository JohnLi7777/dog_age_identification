import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
# 164 196 174要改
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 必须与训练代码完全相同的模型定义
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_shortcut else self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=192, alpha=1.0, round_nearest=8):
        super().__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLU(3, input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MobileNetV2Reg(nn.Module):
    def __init__(self, alpha=1.0, round_nearest=8):
        super().__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLU(3, input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze()

# 数据预处理（必须与训练时一致）
class DynamicPad:
    def __init__(self, max_dim=128, fill=0):
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


def get_transforms():
    val_transform = transforms.Compose([
        DynamicPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return None, val_transform

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
            # ages.append(int(age))
            ages.append(float(age)) # 线性回归
    return np.mean(ages), np.std(ages)


if __name__ == '__main__':
    # 加载整个模型
    model = torch.load('../best_model_cls.pth', map_location=device)
    model.eval()

    # 获取数据预处理和统计量
    train_mean, train_std = age_mean_std('../annotations/newtrain2.txt')
    # 初始化预处理
    _, val_transform = get_transforms()

    # 处理验证集并生成预测结果
    pred_lines = []
    input_file = '../annotations/val.txt'
    img_dir = '../valset'
    a = 0
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc='Processing images'):
            img_name_orig, _ = line.strip().split('\t')
            img_name = img_name_orig.replace('A*', 'A_')

            try:
                # 加载和预处理图像
                img_path = os.path.join(img_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                tensor_img = val_transform(img).unsqueeze(0).to(device)

                # 执行预测
                with torch.no_grad():
                    ############################ 分类
                    # output = model(tensor_img)
                    # pred_age = torch.argmax(output, dim=1).item()
                    ############################
                    pred = model(tensor_img)
                    pred_age = pred.item() * train_std + train_mean  # 反归一化
                    pred_age = int(round(pred_age))  # 四舍五入取整

                # 构建结果行
                new_line = f"{img_name_orig}\t{pred_age}\n"
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                a = a + 1
                new_line = line  # 保留原始行作为容错

            pred_lines.append(new_line)

    # 保存预测结果
    with open('pred_result.txt', 'w') as f:
        f.writelines(pred_lines)

    print("预测完成！结果已保存到 pred_result.txt")
    print(a)
#
# def predict_single_image(img_path, model, transform):
#     try:
#         img = Image.open(img_path).convert('RGB')
#         tensor_img = transform(img).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             output = model(tensor_img)
#             pred_age = torch.argmax(output).item()
#
#         return pred_age
#     except Exception as e:
#         print(f"预测失败: {str(e)}")
#         return -1
#
#
# # 使用示例
# age = predict_single_image("test.jpg", model, transform)
# print(f"预测年龄：{age} 个月")