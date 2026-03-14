import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir  # 图片文件夹的路径
        self.transform = transform  # 数据预处理
        self.img_files = []  # 图片文件列表

        # 使用 os.walk() 遍历主文件夹及子文件夹
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):  # 检查文件扩展名
                    self.img_files.append(os.path.join(root, file))

    def __len__(self):  # 获取数据集大小
        return len(self.img_files)

    def __getitem__(self, idx):  # 获取图片数据
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建自定义数据集实例
custom_dataset = CustomDataset(img_dir='../valset', transform=transform)

# 创建数据加载器
custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

# 初始化均值和标准差
mean = torch.zeros(3)
std = torch.zeros(3)

# 计算均值和标准差
for images in custom_loader:
    for i in range(3):  # 遍历RGB三个通道
        mean[i] += images[:, i, :, :].mean()  # 计算每个通道的均值
        std[i] += images[:, i, :, :].std()  # 计算每个填充的标准差

# 计算平均值
mean /= len(custom_loader)
std /= len(custom_loader)

print('一共有{}张图片'.format(custom_dataset.__len__()))
print(f'均值: {mean}')
print(f'标准差: {std}')