import dlib, cv2, os
import matplotlib.pyplot as plt

detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')

img_path = '../../trainset/A_3IcdQqJS5koAAAAAAAAAAAAAAQAAAQ.jpg'
filename, ext = os.path.splitext(os.path.basename(img_path))
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 16))
plt.imshow(img)
plt.axis('off')  # 不显示坐标轴
plt.title('Original Image')
plt.show()  # 显示图像

dets = detector(img, upsample_num_times=2)

print(dets)

img_result = img.copy()

for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(),
                                                                                      d.rect.right(), d.rect.bottom(),
                                                                                      d.confidence))

    x1, y1 = d.rect.left(), d.rect.top()
    x2, y2 = d.rect.right(), d.rect.bottom()

    cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255, 0, 0), lineType=cv2.LINE_AA)

plt.figure(figsize=(16, 16))
plt.imshow(img_result)
plt.axis('off')  # 不显示坐标轴
plt.title('Original Image')
plt.show()  # 显示图像