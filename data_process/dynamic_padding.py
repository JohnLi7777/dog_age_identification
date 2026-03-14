from PIL import Image
import matplotlib.pyplot as plt

# 假设这是你定义的 DynamicPad 类
class DynamicPad:
    def __init__(self, max_dim=512, fill=0):
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

# 读取原图
original_image_path = '../dog_face_train/A_wkgxQJqBTGoAAAAAAAAAAAAAAQAAAQ.jpg'  # 替换为图片的路径
original_img = Image.open(original_image_path)
print(original_img.size)
# 创建 DynamicPad 实例
max_dim = 256  # 最大维度
dynamic_pad = DynamicPad(max_dim=max_dim)

# 使用 DynamicPad 对原图进行处理
padded_img = dynamic_pad(original_img)

# 显示原图和处理后的图像
plt.figure(figsize=(10, 5))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title('Original Image')
plt.axis('off')  # 不显示坐标轴

# 显示调整后的图像
plt.subplot(1, 2, 2)
plt.imshow(padded_img)
plt.title('Padded Image')
plt.axis('off')  # 不显示坐标轴

# 展示图像
plt.show()