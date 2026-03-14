import cv2
import os
import random
import numpy as np
import uuid
from collections import defaultdict


def process_images():
    # 读取排除列表
    excluded_files = set()
    with open('extract_dog_face_yolov11/NoDetectionDogFace_train.txt', 'r', encoding='ANSI') as f:
        for line in f:
            filename = line.split(',')[0].strip()
            excluded_files.add(filename)

    # 读取年龄映射
    age_mapping = {}
    with open('../annotations/train.txt', 'r') as f:
        for line in f:
            line = line.replace('A*', 'A_')
            parts = line.strip().split('\t')
            if len(parts) == 2:
                age_mapping[parts[0]] = int(parts[1])

    # 统计十年区间的分布
    decade_counts = defaultdict(int)
    for filename, age in age_mapping.items():
        if age > 189:
            continue
        decade_start = (age // 10) * 10
        decade_key = (decade_start, decade_start + 10)
        decade_counts[decade_key] += 1

    # 计算最大数量和需要增强的信息
    max_count = max(decade_counts.values(), default=0)
    decade_info = defaultdict(dict)
    for decade_key, count in decade_counts.items():
        required_rot = max_count - count
        if required_rot <= 0:
            continue
        base_rot = required_rot // count
        remainder = required_rot % count
        decade_info[decade_key] = {
            'base_rot': base_rot,
            'remainder': remainder,
            'processed_count': 0
        }

    # 创建输出目录
    os.makedirs('../dog_face_train_rotation2', exist_ok=True)

    processed_count = 0
    with open('../annotations/newtrain2.txt', 'w', encoding='utf-8') as out_file:
        # 遍历原始图片
        for filename in os.listdir('../dog_face_train'):
            if filename in excluded_files:
                print(f"跳过排除文件: {filename}")
                continue

            if filename not in age_mapping:
                print(f"警告: {filename} 无年龄数据，已跳过")
                continue

            age = age_mapping[filename]
            if age > 189:
                continue

            img_path = os.path.join('../dog_face_train', filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {filename}")
                continue

            # 保存原图
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join('../dog_face_train_rotation2', f"{base_name}.jpg")
            cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            out_file.write(f"{base_name}.jpg\t{age}\n")

            # 计算十年区间
            decade_start = (age // 10) * 10
            decade_key = (decade_start, decade_start + 10)

            # 检查是否需要增强
            if decade_key in decade_info:
                info = decade_info[decade_key]
                current_processed = info['processed_count']

                # 计算需要生成的旋转次数
                if current_processed < info['remainder']:
                    rotations_needed = info['base_rot'] + 1
                else:
                    rotations_needed = info['base_rot']

                # 生成旋转增强
                for i in range(rotations_needed):
                    angle = random.randint(30, 330)  # 排除接近0度和360度的旋转

                    # 计算旋转矩阵
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)

                    # 计算新边界尺寸
                    cos = np.abs(M[0, 0])
                    sin = np.abs(M[0, 1])
                    nW = int((h * sin) + (w * cos))
                    nH = int((h * cos) + (w * sin))

                    # 调整旋转矩阵
                    M[0, 2] += (nW / 2) - center[0]
                    M[1, 2] += (nH / 2) - center[1]

                    # 执行旋转
                    rotated = cv2.warpAffine(
                        img, M, (nW, nH),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                        flags=cv2.INTER_LINEAR
                    )

                    # 生成唯一文件名
                    suffix = uuid.uuid4().hex[:6]
                    new_filename = f"{base_name}_rot_{suffix}.jpg"
                    output_path = os.path.join('../dog_face_train_rotation2', new_filename)

                    cv2.imwrite(output_path, rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    out_file.write(f"{new_filename}\t{age}\n")

                # 更新处理计数器
                decade_info[decade_key]['processed_count'] += 1

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 张图片")

    print(f"处理完成！共处理 {processed_count} 张图片")


if __name__ == "__main__":
    process_images()