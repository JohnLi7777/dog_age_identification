import dlib
import cv2
import os

# 初始化检测器
detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')

# 定义路径
trainset_dir = '../../trainset'
output_dir = '../../dog_face_train'
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

# 支持的图片扩展名
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# 遍历所有图片文件
for filename in os.listdir(trainset_dir):
    if filename.lower().endswith(valid_exts):
        img_path = os.path.join(trainset_dir, filename)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            with open("../extract_dog_face_yolov11/NoDetectionDogFace_train.txt", 'a') as log_file:
                log_file.write(f"{filename},无法读取图片\n")
            print(f"无法读取图片: {img_path}")
            continue

        # 转换颜色空间
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 检测狗脸
        dets = detector(img_rgb, upsample_num_times=0)

        if not dets:
            with open("../extract_dog_face_yolov11/NoDetectionDogFace_train.txt", 'a') as log_file:
                log_file.write(f"{filename},未检测到狗脸\n")  # 记录文件名
            print(f"{filename} 未检测到狗脸")
            continue

        # 处理每个检测结果
        base_name, ext = os.path.splitext(filename)
        for i, detection in enumerate(dets):
            # 获取坐标
            x1 = max(0, detection.rect.left())
            y1 = max(0, detection.rect.top())
            x2 = min(img.shape[1], detection.rect.right())
            y2 = min(img.shape[0], detection.rect.bottom())

            # 检查坐标有效性
            if x1 >= x2 or y1 >= y2:
                with open("../extract_dog_face_yolov11/NoDetectionDogFace_train.txt", 'a') as log_file:
                    log_file.write(f"{filename},无效区域\n")
                print(f"无效区域: {filename} 检测结果 {i}")
                continue

            # 裁剪狗脸区域（使用原始BGR图像）
            face_crop = img[y1:y2, x1:x2]

            # 生成保存文件名（多个检测结果添加序号）
            save_name = f"{base_name}_{i}{ext}" if len(dets) > 1 else filename
            save_path = os.path.join(output_dir, save_name)

            # 保存图片
            cv2.imwrite(save_path, face_crop)
            print(f"已保存: {save_path}")

print("处理完成！")