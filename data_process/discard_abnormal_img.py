import os

# 读取年龄映射
age_mapping = {}
with open('../annotations/train.txt', 'r') as f:
    for line in f:
        line = line.replace('A*', 'A_')
        parts = line.strip().split('\t')
        if len(parts) == 2:
            age_mapping[parts[0]] = int(parts[1])

# 图片路径
image_dir = '../enhance_train_dataset'

# 遍历年龄映射，删除年龄大于192的狗的图片
for image_name, age in age_mapping.items():
    if age > 192:
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):  # 检查文件是否存在
            os.remove(image_path)  # 删除文件
            print(f"Deleted: {image_path}")
        else:
            print(f"File not found: {image_path}")

print("Deletion process completed.")