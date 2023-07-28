import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def process_image(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 只处理图片文件
        if not os.path.isfile(input_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        # 调用find_stamp_area函数进行处理
        find_stamp_area(input_path, output_folder,filename)


def find_stamp_area(image_path,output_folder,output_filename):
#def find_stamp_area(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将图像转换为HSV空间
    lower_red = np.array([150, 0, 200])
    upper_red = np.array([255, 50, 255])
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)  # 创建红色掩膜
    enhanced_image = cv2.bitwise_and(image, image, mask=red_mask)
    kernel = np.ones((5, 5), np.uint8)  # 内核大小
    close_image = cv2.morphologyEx(enhanced_image, cv2.MORPH_CLOSE, kernel)  # 对颜色增强后的图像做闭运算
    open_image = cv2.morphologyEx(enhanced_image, cv2.MORPH_OPEN, kernel)  # 对颜色增强后的图像做开运算
    blurred=cv2.GaussianBlur(close_image,(5,5),0)#高斯滤波
    images = [image, enhanced_image, close_image,open_image ]
    # for i in range(4):
    #     plt.subplot(1, 4, i + 1)
    #     plt.imshow(images[i])
    #plt.show()
    gray=cv2.cvtColor(close_image,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(close_image,(5,5),0)
    edges = cv2.Canny(blurred, 100, 200)  # 边缘检测
    #plt.imshow(edges)
    #plt.show()
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("hierarchy", hierarchy)
    # 判断是否存在父子关系

    # 查找最大轮廓
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        #print("area",area)
        if area > max_area:
            max_area = area
            max_contour = contour

        # 绘制边界框
        if area>1000:
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #保存处理后的图像
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)

        # 显示结果图像
    # cv2.imshow("Stamp Area", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# 调用函数进行印章区域识别
input_folder="D:/python/patternrecognition/circle/circle"
output_folder="D:/python/patternrecognition/circle/process"
process_image(input_folder, output_folder)

#find_stamp_area("12.jpg")