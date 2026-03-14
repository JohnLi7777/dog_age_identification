from ultralytics import YOLO
import numpy as np
import cv2
import os

# 加载sam模型，如果没有这个框架也会自动的为你下载
model = YOLO("../../model/yolov8x-seg.pt")

output_dir = '../../enhance_val_dataset'
os.makedirs(output_dir, exist_ok=True)


# 遍历 trainset 文件夹中的所有图像文件
input_dir = '../../valset'
for filename in os.listdir(input_dir):
    # 检查文件扩展名，确保是图片格式
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 可以根据需要添加其他格式
        img_path = os.path.join(input_dir, filename)

        # 读取图像
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 记录原始尺寸
        original_shape = orig_img.shape[:2]  # 记录 (height, width)

        # 这里特别注意，因为使用yolov8训练的时候默认会把图片resize成448*640的尺寸，所以这里也得改成你训练的尺寸
        orig_img = cv2.resize(orig_img, (448, 640))  # 注意OpenCV中尺寸是先宽度后高度

        # 使用模型进行推理， 后面save=True的参数可以输出测试分割的图片
        results = model(orig_img, save=False)

        # 初始化合并的mask
        combined_mask = None
        detected_objects = []

        # 检查是否有检测到的实例
        if results[0].masks is not None:
            for i in range(len(results[0].masks)):
                # 获取类别ID
                class_id = int(results[0].boxes.cls[i].item())
                detected_objects.append(model.names[class_id])  # 添加检测到的对象
                # 检查是否为狗（COCO数据集中狗的ID为16）
                if model.names[class_id] in ['dog', 'teddy bear', 'cow', 'bear', 'cat', 'sheep']:
                    # 获取当前mask并转换为numpy数组
                    mask = results[0].masks.data[i].cpu().numpy().astype(np.bool_)
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = np.logical_or(combined_mask, mask)

        if combined_mask is None:
            detected_objects_str = ', '.join(detected_objects) if detected_objects else 'none'
            # 如果没有检测到狗或毛绒玩具熊，保存原图并记录文件名
            orig_img = cv2.resize(orig_img, (original_shape[1], original_shape[0]))
            cv2.imwrite(os.path.join(output_dir, filename), orig_img)  # 保存原图
            with open("no_detection_val_images.txt", 'a') as log_file:
                log_file.write(f"{filename},{detected_objects_str}\n")  # 记录文件名
            print(f"图像 {filename} 中未检测到狗或毛绒玩具熊，已保存原图并记录在日志中。")
            continue

        # 应用合并后的mask
        masked_image = np.zeros_like(orig_img)
        masked_image[combined_mask] = orig_img[combined_mask]

        # 如果你想要背景透明（假设原始图像是RGB格式）
        # 创建一个RGBA图像，其中背景是透明的
        alpha_channel = np.ones(orig_img.shape[:2], dtype=orig_img.dtype) * 255
        masked_image_rgba = cv2.merge((masked_image[..., 0], masked_image[..., 1], masked_image[..., 2], alpha_channel))
        masked_image_rgba[~combined_mask] = (0, 0, 0, 0)
        masked_image_rgba = cv2.resize(masked_image_rgba, (original_shape[1], original_shape[0]))
        # masked_image_rgba = cv2.resize(masked_image_rgba, (224, 224))

        # 保存处理后的图像
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, masked_image_rgba)
        print(f"处理完: {filename}，已保存至: {output_path}")


