########################################灰度化#########################################################################################
import cv2
import matplotlib.pyplot as plt

# path = 'D:/Network-sr/testpic.jpg'
#
# image = cv2.imread(path)
#
# # 将图像从BGR格式转换为RGB格式（因为OpenCV读取图像为BGR格式，而matplotlib显示为RGB格式）
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 显示图像
# plt.imshow(image)
# plt.axis('off')  # 取消坐标轴显示
# plt.show()

########################################直方图均衡化1#################################################################################
# import cv2 as cv
# import matplotlib.pyplot as plt
# # 直方图均衡化
# def img_histogram_balance(img):
#     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # cv.equalizeHist(src) 用于实现图像的直方图均衡化，其输入是灰度图像，输出的是直方图均衡化的图像。
#     result = cv.equalizeHist(img_gray)
#     plt.title("Origin")
#     plt.subplot(121)
#     plt.imshow(img_gray)
#     # 绘制原始直方图
#     plt.subplot(122)
#     plt.hist(img_gray.ravel(), 256)
#     plt.show()
#     plt.title("equalize")
#     plt.subplot(121)
#     plt.imshow(result)
#     # 绘制均衡化直方图
#     plt.subplot(122)
#     plt.hist(result.ravel(), 256)
#     plt.show()
# # 限制对比度的直方图均衡化
# def limit_histogram_balance(img):
#     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     '''
#     cv.createCLAHE(clipLimit,tileGridSize) 限制对比度的自适应直方图均衡化
#         clipLimit：颜色对比度的阈值
#         tileGridSize：均衡化的网格大小，即在多少网格下进行直方图的均衡化操作，常用大小是8×8的矩阵。
#     '''
#     clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
#     result = clahe.apply(img_gray)
#     plt.title("limit_equalize_img")
#     plt.subplot(121)
#     plt.imshow(result)
#     plt.subplot(122)
#     plt.hist(result.ravel(), 256)
#     plt.show()
# if __name__ == '__main__':
#     img = cv.imread('D:/Python_program_of_Network-sr/z_input_testpic/testpic5.jpg')
#     # img_histogram_balance(img)
#     limit_histogram_balance(img)

########################################直方图均衡化2####################################################################################
# import cv2
# import matplotlib.pyplot as plt
#
# 读取彩色图像
# img = cv2.imread('D:/Python_program_of_Network-sr/z_input_testpic/testpic.jpg')
#
# # 将图像从BGR颜色空间转换为HSV颜色空间
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# # 分离H、S、V通道
# h, s, v = cv2.split(hsv_img)
#
# # 对V通道进行直方图均衡化
# equalized_v = cv2.equalizeHist(v)
#
# # 合并处理后的通道
# equalized_hsv_img = cv2.merge([h, s, equalized_v])
#
# # 将图像从HSV颜色空间转换回BGR颜色空间
# equalized_img = cv2.cvtColor(equalized_hsv_img, cv2.COLOR_HSV2BGR)
#
# # 显示原始图像及其直方图
# plt.subplot(2, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.subplot(2, 2, 2)
# plt.hist(img.flatten(), bins=256, color='r', alpha=0.5)
# plt.title('Histogram of Original Image')
#
# # 显示均衡化后的图像及其直方图
# plt.subplot(2, 2, 3)
# plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
# plt.title('Equalized Image')
# plt.subplot(2, 2, 4)
# plt.hist(equalized_img.flatten(), bins=256, color='r', alpha=0.5)
# plt.title('Histogram of Equalized Image')
#
# plt.tight_layout()
# plt.show()
#########################################限制对比度直方图均衡化##############################################################################

# import cv2
#
# img = cv2.imread("D:/Python_program_of_Network-sr/z_input_testpic/testpic3.jpg", 0)
# img = cv2.resize(img, None, fx=1, fy=1)
# # 创建CLAHE对象
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
# # 限制对比度的自适应阈值均衡化
# dst = clahe.apply(img)  #限制对比度自适应直方图均衡化(CLAHE)
# # 使用全局直方图均衡化
# # equa = cv2.equalizeHist(img) #全局直方图均衡化(HE)
# # 分别显示原图，CLAHE，HE
# cv2.imshow("img", img)
# cv2.imshow("dst", dst)
# # cv2.imshow("equa", equa)
# cv2.waitKey()

#########################################锐化#########################################################################################
# import cv2
# import numpy as np
# # 读取图像
# image = cv2.imread('testpic.jpg')
#
# # 定义锐化核
# kernel = np.array([[0, -1, 0],
#                    [-1, 5.5,-1],
#                    [0, -1, 0]])
#
# # 进行卷积操作
# sharpened = cv2.filter2D(image, -1, kernel)
#
# # 显示结果图像
# cv2.imshow('Input', image)
# cv2.imshow('Sharpened', sharpened)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
###################################################阴影删除###################################################################################
# import cv2
# import numpy as np
#
# def max_filter(image,filter_size):
#     # padding操作，在最大滤波中需要在原图像周围填充（filter_size//2）个小的数字，一般取-1
#     # 先生成一个全为-1的矩阵，大小和padding后的图像相同
#     empty_image = np.full((image.shape[0] + (filter_size // 2) * 2, image.shape[1] + (filter_size // 2) * 2), -1)
#     # 将原图像填充进矩阵
#     empty_image[(filter_size // 2):empty_image.shape[0] - (filter_size // 2),
#     (filter_size // 2):empty_image.shape[1] - (filter_size // 2)] = image.copy()
#     # 创建结果矩阵，和原图像大小相同
#     result = np.full((image.shape[0], image.shape[1]), -1)
#
#     # 遍历原图像中的每个像素点，对于点，选取其周围（filter_size*filter_size）个像素中的最大值，作为结果矩阵中的对应位置值
#     for h in range(filter_size // 2, empty_image.shape[0]-filter_size // 2):
#         for w in range(filter_size // 2, empty_image.shape[1]-filter_size // 2):
#             filter = empty_image[h - (filter_size // 2):h + (filter_size // 2) + 1,
#                      w - (filter_size // 2):w + (filter_size // 2) + 1]
#             result[h-filter_size // 2, w-filter_size // 2] = np.amax(filter)
#     return result
#
# def min_filter(image,filter_size):
#     # padding操作，在最大滤波中需要在原图像周围填充（filter_size//2）个大的数字，一般取大于255的
#     # 先生成一个全为-1的矩阵，大小和padding后的图像相同
#     empty_image = np.full((image.shape[0] + (filter_size // 2) * 2, image.shape[1] + (filter_size // 2) * 2), 400)
#     # 将原图像填充进矩阵
#     empty_image[(filter_size // 2):empty_image.shape[0] - (filter_size // 2),
#     (filter_size // 2):empty_image.shape[1] - (filter_size // 2)] = image.copy()
#     # 创建结果矩阵，和原图像大小相同
#     result = np.full((image.shape[0], image.shape[1]), 400)
#
#     # 遍历原图像中的每个像素点，对于点，选取其周围（filter_size*filter_size）个像素中的最小值，作为结果矩阵中的对应位置值
#     for h in range(filter_size // 2, empty_image.shape[0]-filter_size // 2):
#         for w in range(filter_size // 2, empty_image.shape[1]-filter_size // 2):
#             filter = empty_image[h - (filter_size // 2):h + (filter_size // 2) + 1,
#                      w - (filter_size // 2):w + (filter_size // 2) + 1]
#             result[h-filter_size // 2, w-filter_size // 2] = np.amin(filter)
#     return result
# def remove_shadow(image_path):
#     image = cv2.imread(image_path, 0)
#
#     max_result=max_filter(image,30)
#     min_result=min_filter(max_result,30)
#     result=image-min_result
#     result=cv2.normalize(result, None, 0, 255, norm_type=cv2.NORM_MINMAX)
#     return result
#
#
# if __name__=="__main__":
#     result=remove_shadow('D:/Python_program_of_Network-sr/z_input_testpic/testpic3.jpg')
#     cv2.imwrite("D:/Python_program_of_Network-sr/z_output_outpic/out1.jpg", result)

#############################################阴影去除+限制对比度直方图均衡########################################################################################

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def max_filtering(N, I_temp):
#     wall = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
#     wall[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)] = I_temp.copy()
#     temp = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
#     for y in range(0,wall.shape[0]):
#         for x in range(0,wall.shape[1]):
#             if wall[y,x]!=-1:
#                 window = wall[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
#                 num = np.amax(window)
#                 temp[y,x] = num
#     A = temp[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)].copy()
#     return A
#
# def min_filtering(N, A):
#     wall_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
#     wall_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)] = A.copy()
#     temp_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
#     for y in range(0,wall_min.shape[0]):
#         for x in range(0,wall_min.shape[1]):
#             if wall_min[y,x]!=300:
#                 window_min = wall_min[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
#                 num_min = np.amin(window_min)
#                 temp_min[y,x] = num_min
#     B = temp_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)].copy()
#     return B
#
# def background_subtraction(I, B):
#     O = I - B
#     norm_img = cv2.normalize(O, None, 0,255, norm_type=cv2.NORM_MINMAX)
#     return norm_img
#
# def min_max_filtering(M, N, I):
#     if M == 0:
#         #max_filtering
#         A = max_filtering(N, I)
#         #min_filtering
#         B = min_filtering(N, A)
#         #subtraction
#         normalised_img = background_subtraction(I, B)
#     elif M == 1:
#         #min_filtering
#         A = min_filtering(N, I)
#         #max_filtering
#         B = max_filtering(N, A)
#         #subtraction
#         normalised_img = background_subtraction(I, B)
#     return normalised_img
#
# P = cv2.imread('C:/Users/94464/Desktop/Netsecurity-sr/Python_program_of_Network-sr/z_input_testpic/testpic3.jpg',0)
# cv2.imwrite("C:/Users/94464/Desktop/Netsecurity-sr/Python_program_of_Network-sr/z_output_outpic2/out_gray.jpg", P)
# #We can edit the N and M values here for P and C images
# plt.imshow(P,cmap='gray')
# plt.title("original image")
# plt.show()
#
# O_P = min_max_filtering(M = 0, N = 4, I = P)
# cv2.imwrite("C:/Users/94464/Desktop/Netsecurity-sr/Python_program_of_Network-sr/z_output_outpic2/out_mmf.jpg", O_P)
# # #Display final output
# # plt.imshow(O_P, cmap = 'gray')
# # plt.title("Final output")
# # plt.show()
#
# img = cv2.imread("C:/Users/94464/Desktop/Netsecurity-sr/Python_program_of_Network-sr/z_output_outpic2/out_mmf.jpg", 0)
# img = cv2.resize(img, None, fx=1, fy=1)
# # 创建CLAHE对象
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
# # 限制对比度的自适应阈值均衡化
# dst = clahe.apply(img)  #限制对比度自适应直方图均衡化(CLAHE)
# cv2.imwrite("C:/Users/94464/Desktop/Netsecurity-sr/Python_program_of_Network-sr/z_output_outpic2/out_dst.jpg",dst)
# # 使用全局直方图均衡化
# # equa = cv2.equalizeHist(img) #全局直方图均衡化(HE)
# # 分别显示原图，CLAHE，HE
# cv2.imshow("img", img)
# cv2.imshow("dst", dst)
# # cv2.imshow("equa", equa)
# cv2.waitKey()

#######################################################################################################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

def max_filtering(N, I_temp):
    wall = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    wall[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)] = I_temp.copy()
    temp = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    for y in range(0,wall.shape[0]):
        for x in range(0,wall.shape[1]):
            if wall[y,x]!=-1:
                window = wall[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num = np.amax(window)
                temp[y,x] = num
    A = temp[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)].copy()
    return A

def min_filtering(N, A):
    wall_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    wall_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)] = A.copy()
    temp_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    for y in range(0,wall_min.shape[0]):
        for x in range(0,wall_min.shape[1]):
            if wall_min[y,x]!=300:
                window_min = wall_min[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num_min = np.amin(window_min)
                temp_min[y,x] = num_min
    B = temp_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)].copy()
    return B

def background_subtraction(I, B):
    O = I - B
    norm_img = cv2.normalize(O, None, 0,255, norm_type=cv2.NORM_MINMAX)
    return norm_img

def min_max_filtering(M, N, I):
    if M == 0:
        #max_filtering
        A = max_filtering(N, I)
        #min_filtering
        B = min_filtering(N, A)
        #subtraction
        normalised_img = background_subtraction(I, B)
    elif M == 1:
        #min_filtering
        A = min_filtering(N, I)
        #max_filtering
        B = max_filtering(N, A)
        #subtraction
        normalised_img = background_subtraction(I, B)
    return normalised_img

P = cv2.imread('C:/Users/94464/Desktop/Netsecurity-sr/Python_program_of_Network-sr/z_input_testpic/testpic3.jpg',0)

cv2.imshow('original image',P)
cv2.waitKey()

O_P = min_max_filtering(M = 0, N = 4, I = P) #去除阴影
img = cv2.convertScaleAbs(O_P)               # 将处理后的32位图转换为8位，这里会不会影响，丢失信息？
cv2.imshow('mmf',img)
cv2.waitKey()

img = cv2.resize(img, None, fx=1, fy=1)
# 创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
# 限制对比度的自适应阈值均衡化
dst = clahe.apply(img)  #限制对比度自适应直方图均衡化(CLAHE)
cv2.imshow("dst", dst)
cv2.waitKey()

blurred=cv2.GaussianBlur(dst,(3,3),0) # 高斯滤波
cv2.imshow('blurred',blurred)
cv2.waitKey()

edges = cv2.Canny(blurred, 100, 200)  # 边缘检测
cv2.imshow("edges", edges)
cv2.waitKey()