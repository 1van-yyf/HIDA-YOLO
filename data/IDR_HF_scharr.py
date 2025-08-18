import cv2
import os
import numpy as np
import tifffile as tiff

def process_images_to_4_channels(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder):
        # 构造完整路径
        input_path = os.path.join(input_folder, filename)

        # 确保是图像文件
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            continue

        # 读取图像 (BGR 格式)
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"Failed to read {input_path}. Skipping.")
            continue

        # 转换为 RGB 格式
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 转为灰度图
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # ---------------------------
        # 使用 Scharr 算子提取梯度
        # ---------------------------
        scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)  # X方向
        scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)  # Y方向

        # 计算梯度幅值
        scharr_magnitude = cv2.magnitude(scharr_x, scharr_y)

        # 转换为 8 位图像
        scharr_magnitude = cv2.convertScaleAbs(scharr_magnitude)

        # 扩展 Scharr 特征图的通道维度
        scharr_magnitude_expanded = np.expand_dims(scharr_magnitude, axis=-1)  # (H, W, 1)

        # 将 Scharr 特征图与 RGB 图像拼接为 4 通道图像
        image_with_4_channels = np.concatenate((image_rgb, scharr_magnitude_expanded), axis=-1)  # (H, W, 4)

        # 构造输出路径（建议根据需求修改输出文件夹命名）
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.tiff')

        # 保存 4 通道图像，指定 photometric='rgb' 以确保前三通道为 RGB
        tiff.imwrite(output_path, image_with_4_channels, photometric='rgb')
        print(f"Processed and saved: {output_path}")
#clipart VOC
# -------------------------------
# 设置输入文件夹和输出文件夹路径
# -------------------------------
input_folder = '/newHome/S1_YYF/YYF/yolov9-main/dataset/VOC/images/val'
output_folder = '/newHome/S1_YYF/YYF/yolov9-main/dataset/VOC_scharr/images/val'

# 调用函数
process_images_to_4_channels(input_folder, output_folder)
