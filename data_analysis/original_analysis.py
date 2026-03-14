'''
    数据样本分析
    画出图像分辨率散点图
'''
import matplotlib
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')


def plot_resolution(dataset_root_path):
    img_size_list = []  # 承接所有图片的长宽数据
    small_images = []  # 用于记录长或宽小于200的图片文件名

    for root, dirs, files in os.walk(dataset_root_path):
        for file_i in files:
            file_i_full_path = os.path.join(root, file_i)
            try:
                img_i = Image.open(file_i_full_path)
                img_i_size = img_i.size  # 获取单张图像的长宽
                img_size_list.append((file_i, img_i_size))  # 记录图片名称和尺寸

                # 检查长或宽是否小于
                if img_i.size[0] < 200 and img_i.size[1] < 200:
                    small_images.append(file_i)  # 记录文件名
            except Exception as e:
                print(f"无法打开文件 {file_i_full_path}，可能不是图像文件: {e}")

    # print("所有图片的尺寸:")
    # for filename, size in img_size_list:
    #     print(f"{filename}: {size}")

    if small_images:
        print("长或宽小于200的图片数量:")
        print(len(small_images))
        print("长或宽小于200的图片文件名:")
        for small_image in small_images:
            print(small_image)
    else:
        print("没有找到长或宽小于300的图片。")

    if not img_size_list:
        print("没有找到有效的图片文件。")
        return

    width_list = [img_size_list[i][1][0] for i in range(len(img_size_list))]
    height_list = [img_size_list[i][1][1] for i in range(len(img_size_list))]

    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

    plt.scatter(width_list, height_list, s=1)
    plt.xlabel("宽")
    plt.ylabel("高")
    plt.title("图像宽高分布")

    # 添加红色虚线 y = x
    max_dimension = max(max(width_list), max(height_list))
    plt.plot([0, max_dimension], [0, max_dimension], 'r--', label='宽 = 高')  # 画出y=x的红色虚线

    # plt.legend()  # 添加图例
    plt.show()

# 定义一个函数来提取年龄
def extract_ages(file_path):
    ages = []  # 用于存储年龄的列表
    age_list = []
    outlier = 0
    with open(file_path, 'r') as file:  # 打开文件进行读取
        for line in file:
            parts = line.strip().split()  # 按空格分隔每一行
            if len(parts) > 1:  # 确保行中有两个部分
                try:
                    age = int(parts[1])  # 尝试将年龄部分转换为整数
                    ages.append(age)  # 将年龄添加到列表中
                except ValueError:
                    print(f"无法转换 {parts[1]} 为整数")  # 如果转换失败，输出警告

    # 定义年龄段的范围，并输出年龄段的计数
    bins = range(0, 600, 10)  # 年龄段，0到192，每10岁一个区间
    counts, _ = np.histogram(ages, bins=bins)
    print(counts.sum())
    # 输出每个年龄段的图片数量
    for i in range(len(counts)):
        age_range = f"{bins[i]}-{bins[i + 1]}"  # 计算年龄段
        age_list.append(counts[i])
        if counts[i] > 192:
            outlier += 1
        print(f"年龄段 {age_range} 的图片数量: {counts[i]}")
    # print(age_list)
    print(f"异常年龄占比：{(outlier / len(ages)) * 100:.3f}%")
    # 绘制柱状图
    plt.figure(figsize=(12, 8))  # 设置图形大小
    # bins：指定将数据分区的方式；edgecolor：指定了柱子的边框颜色；alpha：控制透明度
    counts, _, patches = plt.hist(ages, bins=range(0, 200, 12), edgecolor='black', alpha=0.7)
    plt.title('年龄分布柱状图')  # 设置标题
    plt.xlabel('年龄 (月)')  # 设置 x 轴标签
    plt.ylabel('图片数量')  # 设置 y 轴标签
    plt.xticks(range(0, 200, 12))  # 设置 x 轴刻度
    # plt.xticks([0, 6, 12, 72, 132, 192])  # 设置 x 轴刻度
    plt.grid(axis='y', alpha=0.75)  # 添加 y 轴网格
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
    # 在每个柱子上方显示具体数值
    for count, patch in zip(counts, patches):
        height = patch.get_height()  # 获取柱子的高度
        plt.text(patch.get_x() + patch.get_width() / 2, height, int(height),
                 ha='center', va='bottom', fontsize=10)  # 在柱子上方添加文本

    # 显示图形
    plt.show()


if __name__ == '__main__':
    dataset_root_path = "../trainset"
    plot_resolution(dataset_root_path)
    # extract_ages("../annotations/train.txt")