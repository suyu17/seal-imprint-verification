import cv2
import numpy as np
import matplotlib.pyplot as plt

# #不同颜色通道比较
#
# img_BGR = cv2.imread('111.jpg')
#
# img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB) #BGR转换为RGB
#
# img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY) #灰度化处理
#
# img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV) #BGR转HSV
#
# img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb) #BGR转YCrCb
#
# img_HLS = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HLS) #BGR转HLS
#
# img_XYZ = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2XYZ) #BGR转XYZ
#
# img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB) #BGR转LAB
#
# img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV) #BGR转YUV
#
# #调用matplotlib显示处理结果
#
# titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
#
# images = [img_BGR, img_RGB, img_GRAY, img_HSV, img_YCrCb,img_HLS, img_XYZ, img_LAB, img_YUV]
#
# for i in range(9):
#   plt.subplot(3, 3, i+1)
#   plt.imshow(images[i], 'gray')
#   plt.title(titles[i])
#   plt.xticks([]),plt.yticks([])
# plt.show()
###做红色区域处理
##提取红色区域并增强
extract=cv2.imread('12.jpg')
kernel=np.ones((5,5),np.uint8)#3*3的内核
erosion=cv2.erode(extract,kernel)#腐蚀
dilation=cv2.dilate(extract,kernel)#膨胀
open=cv2.morphologyEx(extract,cv2.MORPH_OPEN,kernel)#开运算
close=cv2.morphologyEx(extract,cv2.MORPH_CLOSE,kernel)#闭运算
constarct1=np.hstack([extract,erosion,dilation])
cv2.imshow("image erosion dilation",constarct1)
cv2.waitKey(0)
constarct2=np.hstack([extract,open,close])
cv2.imshow("image open close",constarct2)
cv2.waitKey(0)
# #使用顶帽运算进行背景提取
# def tophat_demo(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)#二值化
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))#15*15内核
#     dst = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)#顶帽变换
#     cv2.imshow('sda', dst)
#     cimage = np.array(thresh.shape, np.uint8)
#     cimage = 120
#     dst = cv2.add(dst, cimage)
#     cv2.imshow('tophat', dst)
#     cv2.waitKey(0)
# tophat=tophat_demo(extract)

 #将图像转换为HSV空间
##确定所选区域的HSV值
# def getpos(event,x,y,flags,param):
#     if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
#         print(hsv_image[y,x])
#
# cv2.imshow("imageHSV",hsv_image)
# cv2.setMouseCallback("imageHSV",getpos)
# cv2.waitKey(0)
# #cv2.imshow('hsv',hsv_image)
# #cv2.waitKey(0)
lower_red=np.array([150,0,200])
upper_red=np.array([255,50,255])
red_mask = cv2.inRange(hsv_image, lower_red, upper_red)#创建红色掩膜
enhanced_image = cv2.bitwise_and(extract, extract, mask=red_mask)
close_image=cv2.morphologyEx(enhanced_image,cv2.MORPH_CLOSE,kernel)#对颜色增强后的图像做闭运算
images=[extract,enhanced_image,close_image]
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i])
plt.show()
edges=cv2.Canny(close_image,100,200)#边缘检测
plt.imshow(edges)
plt.show()
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 查找最大轮廓
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
            max_area = area
            max_contour = contour

    # 绘制边界框
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(extract, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果图像
cv2.imshow("Stamp Area", extract)
cv2.waitKey(0)
cv2.destroyAllWindows()

# lower=np.array([0,0,100])
# upper=np.array(([40,40,255])) 
# binary=cv2.inRange(extract,lower,upper)
# images=[extract,binary]
# for i in range(2):
#     plt.subplot(1,2,i+1)
#     plt.imshow(images[i])
# plt.show()


