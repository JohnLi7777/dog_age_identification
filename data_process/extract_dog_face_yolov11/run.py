import os
import cv2
from ultralytics import YOLO
import numpy as np

# 加载训练好的 YOLO 模型
model_path = '../../model/dogFaceDetect.pt'
model = YOLO(model_path)

# 输入图像和输出图像路径
input_folder = '../trainset/'
output_folder = '../dog_face_train/'

# 创建输出文件夹如果不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历 input_folder 中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 支持的图像格式
        img_path = os.path.join(input_folder, filename)

        # 读取图像
        image = cv2.imread(img_path)

        # 使用模型进行推理，返回检测结果
        results = model.predict(img_path)

        # 遍历检测结果
        for result in results:  # results 是一个列表，每个元素是一个检测结果
            boxes = result.boxes  # 获取 boxes 对象

            # 确保有检测结果
            if boxes is not None and len(boxes.xyxy) > 0:
                confidences = boxes.conf.cpu().numpy()  # 获取置信度数组
                max_conf_index = np.argmax(confidences)  # 找到最大置信度的索引

                # 获取最大置信度框的信息
                best_box = boxes.xyxy[max_conf_index]
                x1, y1, x2, y2 = map(int, best_box[:4])  # 获取坐标


                # 裁剪图像
                dog_face = image[y1:y2, x1:x2]

                # 输出保存路径
                output_path = os.path.join(output_folder, filename)

                # 保存裁剪出的狗脸图像
                cv2.imwrite(output_path, dog_face)

            else:
                cv2.imwrite(os.path.join(output_folder, filename), image)
                with open("NoDetectionDogFace_train.txt", 'a') as log_file:
                    log_file.write(f"{filename},未检测到狗脸\n")

print("所有狗脸图像已保存至:", output_folder)
