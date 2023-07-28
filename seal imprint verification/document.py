import cv2
import numpy as np

def find_character_edges(image_path):
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 使用Canny边缘检测算法
    edges = cv2.Canny(img, 100, 200)  # 调整阈值来控制边缘检测结
    return edges

def find_character_positions(edges):
    # 找到边缘图像中的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 提取轮廓的边界框并获取字符的位置信息
    char_positions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_positions.append((x + w // 2, y + h // 2))  # 使用字符的中心点位置

        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(edges, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        # # 显示结果图像
        # cv2.imshow("Stamp Area", edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return char_positions

def calculate_variance(values):
    # 计算方差
    return np.var(values)

def is_document_image(image_path, threshold=100000):
    # 获取字符边缘图像
    edges = find_character_edges(image_path)
    # 获取字符位置信息
    char_positions = find_character_positions(edges)
    # 如果未找到字符轮廓，则返回False
    if not char_positions:
        return False
    # 提取字符位置信息的水平和垂直坐标
    x_positions, y_positions = zip(*char_positions)
    # 计算水平和垂直方向的方差
    var_x = calculate_variance(x_positions)
    var_y = calculate_variance(y_positions)
    print(var_x)
    print(var_y)

    # 判断是否为文档图像
    if var_x < threshold and var_y < threshold:
        return True
    else:
        return False

is_document=is_document_image("222.jpg")
if is_document:
    print("This image is likely a document.")
else:
    print("This image is likely not a document.")