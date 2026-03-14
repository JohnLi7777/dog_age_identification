import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import  QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QTextEdit
from ollama import chat
from ultralytics import YOLO
import sys
import nav_ui,infer_ui


# 自定义信号类
class StreamResponseThread(QThread):
    update_signal = pyqtSignal(str)  # 文字更新信号
    complete_signal = pyqtSignal()  # 完成信号
    error_signal = pyqtSignal(str)  # 错误信号

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            stream = chat(
                model='deepseek-r1:7b',
                messages=[{'role': 'user', 'content': self.prompt}],
                stream=True,
            )

            full_response = ""
            thinking = False
            for chunk in stream:
                content = chunk['message']['content']
                if "<think>" in content:
                    thinking = True
                if "</think>" in content:
                    thinking = False
                    content = content.split("</think>")[-1]
                if not thinking:
                    full_response += content
                    self.update_signal.emit(content)  # 发送每个片段

            self.complete_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))


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
            ages.append(int(age))
    return np.mean(ages), np.std(ages)

# SE模块定义
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# SEBottleneck定义
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation,
                               groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes * self.expansion, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 完整模型定义
class ResNetSEReg(nn.Module):
    def __init__(self, block, layers, num_classes=1, pretrained=False):
        super(ResNetSEReg, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        norm_layer = nn.BatchNorm2d

        # 输入层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights('model/resnet50-0676ba61.pth')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _load_pretrained_weights(self, weight_path):
        """加载本地预训练权重"""
        pretrain_dict = torch.load(weight_path)

        # 过滤不匹配的键
        model_dict = self.state_dict()
        # 特殊处理conv1的权重（适应灰度图输入时可能需要删除此判断）
        if model_dict['conv1.weight'].shape[1] != pretrain_dict['conv1.weight'].shape[1]:
            pretrain_dict.pop('conv1.weight')

        # 通用过滤规则
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if k in model_dict and model_dict[k].shape == v.shape}

        # 加载可匹配的参数
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        print(f"成功加载 {len(pretrain_dict)}/{len(model_dict)} 个参数")

        # 初始化新添加的SE模块参数
        for m in self.modules():
            if isinstance(m, SELayer):
                nn.init.kaiming_normal_(m.fc[0].weight)
                nn.init.kaiming_normal_(m.fc[2].weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# 模型构建函数
def resnet50_se_reg(pretrained=False):
    model = ResNetSEReg(SEBottleneck, [3, 4, 6, 3], pretrained=pretrained)
    return model


# 主界面1
class UI_main_1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "首页"
        self.ui = nav_ui.Ui_NavWindow()
        self.ui.setupUi(self)
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # 获取显示器分辨率
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.6)
        self.width = int(self.screenwidth * 0.6)
        print("Screen height {}".format(self.screenheight))
        print("Screen width {}".format(self.screenwidth))
        self.resize(self.width, self.height)
        self.setWindowTitle(self.title)

        self.ui.pushButton.clicked.connect(lambda: self.go_to_main_2())
        self.ui.pushButton_2.clicked.connect(lambda: self.exit())

        self.show()

    # 跳转到主界面2
    def go_to_main_2(self):
        self.win=UI_main_2()
        self.close()

    # 关闭程序
    def exit(self):
        self.close()

    # 拖动
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

# 主界面2
class UI_main_2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "推理"
        self.ui = infer_ui.Ui_InferWindow()
        self.ui.setupUi(self)
        self.img_path = None
        self.videoName = ''
        self.cap = None
        # 初始化 QTimer
        # self.timer = QTimer()
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # 获取显示器分辨率
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.6)
        self.width = int(self.screenwidth * 0.6)
        self.resize(self.width, self.height)
        self.setWindowTitle(self.title)

        self.ui.pushButton.clicked.connect(lambda: self.open_image())
        self.ui.pushButton_2.clicked.connect(lambda: self.detect())
        self.ui.pushButton_3.clicked.connect(lambda: self.interpret())
        self.ui.pushButton_4.clicked.connect(lambda: self.go_to_main_1())
        self.show()

        # 新增初始化部分
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_mean = 0
        self.train_std = 1
        self.response_thread = None
        self.current_response = ""  # 当前响应内容
        self.pred_age = 0  # 存储预测年龄
        self.boxes = None
        self.dog_detected = False
        # 初始化文本编辑框
        self.ui.textEdit.clear()
        self.ui.textEdit.setWordWrapMode(QtGui.QTextOption.WordWrap)

        try:
            # 加载模型
            self.model = torch.load('../model/best_model_se_face4_19_5826.pth', map_location=self.device)
            # 加载训练好的 YOLO 模型
            yolo_model_path = '../model/dogFaceDetect.pt'
            self.yolo_model = YOLO(yolo_model_path)
            self.model.eval()

            # 加载统计量
            self.train_mean, self.train_std = age_mean_std('../annotations/train_correct.txt')
        except Exception as e:
            QMessageBox.critical(self, "初始化失败", f"无法加载模型或统计量：{str(e)}")
            self.close()



    # 读取图片,将路径存入txt文件中
    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;*.mp4;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.ui.label_5.width(), self.ui.label_5.height())
        self.ui.label_5.setPixmap(jpg)
        self.img_path = imgName

    def detect(self):
        if not self.img_path:
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return

        try:
            # # 图像预处理
            # img = Image.open(self.img_path).convert('RGB')
            # _, val_transform = get_transforms_crop()
            # tensor_img = val_transform(img).unsqueeze(0).to(self.device)
            _, val_transform = get_transforms_crop()
            results = self.yolo_model.predict(self.img_path)
            # 遍历检测结果
            for result in results:  # results 是一个列表，每个元素是一个检测结果
                self.boxes = result.boxes  # 获取 boxes 对象
                # 确保有检测结果
                if self.boxes is not None:
                    for i in range(len(self.boxes.xyxy)):
                        self.dog_detected = True
                        box = self.boxes.xyxy[i]
                        confidence = self.boxes.conf[i].cpu().item()
                        detected_class = int(self.boxes.cls[i].item())
                        print(detected_class, confidence)
                        # 检查检测到的类别是否为狗且置信度是否足够
                        if confidence >= 0.5:
                            x1, y1, x2, y2 = map(int, box[:4])

                            # 读取图像
                            image = cv2.imread(self.img_path)

                            # 裁剪图像
                            dog_face = image[y1:y2, x1:x2]
                            # cv2.imshow('dog_face', dog_face)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            # 将裁剪后的图像转换为 PIL 图像
                            dog_face_pil = Image.fromarray(cv2.cvtColor(dog_face, cv2.COLOR_BGR2RGB))

                            # 数据预处理
                            tensor_img = val_transform(dog_face_pil).unsqueeze(0).to(self.device)
                            # 执行推理
                            with torch.no_grad():
                                pred = self.model(tensor_img).item()
                                pred_age = pred * self.train_std + self.train_mean
                                pred_age = round(pred_age, 1)
                                # 显示结果（使用label_3显示结果）
                                self.ui.label_3.setStyleSheet("font-size: 12pt;")
                                self.ui.label_3.setText(f"{pred_age} 个月")
                                self.pred_age = pred_age
                        else:
                            QMessageBox.warning(self, "警告", "请上传狗的图片！")
                    if not self.dog_detected:
                        QMessageBox.warning(self, "警告", "没有检测到狗，请重新上传！")

        except Exception as e:
            QMessageBox.critical(self, "推理错误", f"推理过程出现异常：{str(e)}")
        finally:
            self.boxes = None
            self.dog_detected = False

    def interpret(self):
        if self.pred_age <= 0:
            QMessageBox.warning(self, "警告", "请先进行年龄预测！")
            return

            # 禁用按钮防止重复点击
        self.ui.pushButton_3.setEnabled(False)
        self.current_response = ""
        self.ui.textEdit.clear()
        # 构建提示语
        prompt = f"假如你是一名兽医，我有一只{self.pred_age}个月的卷毛比熊狗，请你从饮食、运动、健康管理、心理这四个方面给我提供护理指南"

        # 创建并启动线程
        self.response_thread = StreamResponseThread(prompt)
        self.response_thread.update_signal.connect(self.update_response)
        self.response_thread.complete_signal.connect(self.response_complete)
        self.response_thread.error_signal.connect(self.handle_error)
        self.response_thread.start()

    def update_response(self, new_text):
        """更新解释内容"""
        cursor = self.ui.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(new_text)

        # 自动滚动到底部
        self.ui.textEdit.ensureCursorVisible()
        self.ui.textEdit.repaint()  # 强制刷新

    def response_complete(self):
        """响应完成处理"""
        self.ui.pushButton_3.setEnabled(True)
        self.response_thread = None

    def handle_error(self, error_msg):
        """错误处理"""
        QMessageBox.critical(self, "错误", f"模型请求失败：{error_msg}")
        self.ui.pushButton_3.setEnabled(True)
        self.response_thread = None

    def go_to_main_1(self):
        self.win=UI_main_1()
        self.close()

    def img_show(self):
        print(self.img_path)
        # detect_main.process_image_simple(img)
        # img_path=(r'D:\my_homework\yolov5_infer\0001.jpg')
        # jpg = QtGui.QPixmap(img_path).scaled(self.ui.show_output.width(), self.ui.show_output.height())
        # self.ui.show_output.setPixmap(jpg)

    def update_frame(self):
        ret, image = self.cap.read()
        print(ret)
        if ret:
            # 图像转换及显示逻辑
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.ui.label_2.setPixmap(QPixmap(vedio_img))
            self.ui.label_2.setScaledContents(True)  # 自适应窗口
            self.ui.label_2.update()
        else:
            self.timer.stop()
            self.cap.release()

    def open_video(self):
        videoName, videoType = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;All Files(*)")
        self.videoName = videoName
        self.cap = cv2.VideoCapture(self.videoName)

    def video_show(self):
        self.timer.start()  # 假设每30毫秒更新一次，根据需要调整
        self.timer.timeout.connect(self.update_frame)

    def exit(self):
        self.win = UI_main_2()
        self.close()

if __name__ =="__main__":
    app =QApplication(sys.argv)
    win = UI_main_1()
    sys.exit(app.exec_())
