import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from flask import Flask,jsonify
import random

# read image
def read_directory(path):
    array_of_img = []
    file_list = os.listdir(path)
    file_list.sort()
    for filename in file_list:
        img = cv2.imdecode(np.fromfile(path + filename,dtype=np.uint8),-1)
        array_of_img.append(img)

    return array_of_img

def read_name(path):
    file_list = os.listdir(path)
    file_list.sort()
    return file_list

def distance(center, point1):
    return np.sqrt(np.square(point1[0][0] - center[0]) + np.square(point1[0][1] - center[1]))


# 将四角按 左上、右上、左下、右下
def order_corner(corner):
    x = []
    y = []
    corner_order = []
    for i in range(len(corner)):
        x.append(corner[i][0][0])
        y.append(corner[i][0][1])
    index_x = np.argsort(np.array(x))

    # 先得到左上 和 右上
    index1 = index_x[0]
    index2 = index_x[1]
    if y[index1] > y[index2]:
        corner_order.append(corner[index2])
        corner_order.append(corner[index1])
    else:
        corner_order.append(corner[index1])
        corner_order.append(corner[index2])
    # 左下、右下
    index3 = index_x[2]
    index4 = index_x[3]
    if y[index3] > y[index4]:
        corner_order.append(corner[index4])
        corner_order.append(corner[index3])
    else:
        corner_order.append(corner[index3])
        corner_order.append(corner[index4])
    return corner_order


# 返回四角
def Get_corner(approxes):
    corner = []
    center = np.array([0, 0])
    for i in range(len(approxes)):
        center[0] += approxes[i][0][0][0]
        center[1] += approxes[i][0][0][1]
    center = center / len(approxes)

    # 根据距离选出四角 (num,1,2)
    index = []
    for approx in approxes:
        dist = []
        for i in range(approx.shape[0]):
            dist.append(distance(center, approx[i]))
        index.append(np.argmax(dist))

    for i in range(len(index)):
        corner.append(approxes[i][index[i]])

    corner = order_corner(corner)
    return corner


def calculate(mask, image,threshold_value):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 提取区域
    row = mask.shape[0]
    col = mask.shape[1]
    for i in range(row):
        for j in range(col):
            if not mask[i, j]:
                gray[i, j] = 0

    # 重新细化边缘
    mb_gray = cv2.medianBlur(gray, 9)
    edges = cv2.Canny(mb_gray, threshold1=50, threshold2=200)
    # 处理连通域
    for i in range(row):
        for j in range(col):
            if edges[i, j] > 0:
                gray[i - 1:i, j:j + 5] = 0

    # 二值化 求连通域
    _, binary_gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    num, mb_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_gray, connectivity=4)
    # 记录面积最大的几个区域
    sorted_id = sorted(range(len(stats)), key=lambda k: stats[k, 4], reverse=True)
    index = sorted_id[1]

    # 直方图均衡化

    count = 0
    sum = 0
    for i in range(row):
        for j in range(col):
            if mb_labels[i, j] == index:
                if gray[i, j] < threshold_value:
                    image[i, j] = (0, 0, 255)
                    count += 1
                # else:
                #     intersity_list = []
                #     for ii in [-1, 0, 1]:
                #         for jj in [-1, 0, 1]:
                #             if ii == 0 and jj == 0:
                #                 pass
                #             elif gray[i + ii, j + jj] >= threshold_value:
                #                 intersity_list.append(gray[i + ii, j + jj])
                #     if len(intersity_list) != 0:
                #         avg = np.mean(np.array(intersity_list))
                #         if avg - gray[i, j] > 30:
                #             image[i, j] = (0, 0, 255)
                #             count += 1
                sum += 1

    print("比例为：", count / sum * 100)
    return count / sum * 100,image


# 提取范围
def Segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 中值滤波
    mb = cv2.medianBlur(gray, 9)

    # 提取边缘
    edges = cv2.Canny(mb, threshold1=50, threshold2=200)

    # 使用Hough检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 360, 250)

    # 区分交点
    mb = cv2.add(mb, 10)

    # 绘制直线
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1800 * (-b))
        y1 = int(y0 + 1800 * a)
        x2 = int(x0 - 1800 * (-b))
        y2 = int(y0 - 1800 * a)
        cv2.line(mb, (x1, y1), (x2, y2), 0, 2)

    # 二值化 求连通域
    _, mb = cv2.threshold(mb, 10, 255, cv2.THRESH_BINARY)
    num, mb_labels, stats, centroids = cv2.connectedComponentsWithStats(mb)

    # 记录面积最大的几个区域
    sorted_id = sorted(range(len(stats)), key=lambda k: stats[k, 4], reverse=True)
    index = sorted_id[0:4]

    # 彩色显示
    # 定义随机颜色
    color = []
    for i in range(num):
        color.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img = np.zeros((mb_labels.shape[0], mb_labels.shape[1], 3), dtype=np.uint8)
    row = mb_labels.shape[0]
    col = mb_labels.shape[1]

    for i in range(row):
        for j in range(col):
            img[i, j, :] = color[mb_labels[i, j]]

    for i in range(len(index)):
        img = cv2.putText(img, str(index[i]), (int(centroids[index[i]][0]), int(centroids[index[i]][1])),
                          cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
    mask = (mb_labels == index[1])

    # 处理铁和洞
    for i in range(row):
        for j in range(col):
            if not mask[i, j]:
                gray[i, j] = 0
    gray = cv2.medianBlur(gray, 3)
    gray_edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # 处理连通域
    for i in range(row):
        for j in range(col):
            if gray_edges[i, j] > 0:
                gray[i - 1:i + 1, j - 1:j + 1] = 0

    # 二值化 求连通域
    _, binary_gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    num, mb_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_gray, connectivity=4)
    # 记录面积最大的几个区域
    sorted_id = sorted(range(len(stats)), key=lambda k: stats[k, 4], reverse=True)
    index = sorted_id[1]

    # 彩色显示
    # 定义随机颜色
    color = []
    for i in range(num):
        color.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img = np.zeros((mb_labels.shape[0], mb_labels.shape[1], 3), dtype=np.uint8)
    row = mb_labels.shape[0]
    col = mb_labels.shape[1]
    for i in range(row):
        for j in range(col):
            img[i, j, :] = color[mb_labels[i, j]]
    img = cv2.putText(img, str(index), (int(centroids[index][0]), int(centroids[index][1])),
                      cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)


    # 存在一些孔洞 进行闭操作
    mask = np.zeros((mb_labels.shape[0], mb_labels.shape[1]), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            if mb_labels[i, j] == index:
                mask[i, j] = 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    return mask

def process(folder_name):
    # 读取图像

    img_array = read_directory(folder_name)

    num_name = 0
    # 将检测结果绘制在图像上
    images = []

    for img in img_array:
        # img = cv2.blur(img, (5, 5))
        b = np.array(img[:, :, 0]).astype(np.float32)
        g = np.array(img[:, :, 1]).astype(np.float32)
        r = np.array(img[:, :, 2]).astype(np.float32)
        img_r = r*2-b-g
        img_r[img_r < 0.0] = 0.

        array_r = img_r.astype(np.uint8)
        # 红色提取并二值化
        array_r[array_r < 100] = 0
        array_r[array_r >= 100] = 255

        # 填充孔洞
        kernel = np.ones((10, 10), np.uint8)
        array_r = cv2.morphologyEx(array_r, cv2.MORPH_CLOSE, kernel)

        # 挑出四个大的连通的
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(array_r)
        # 记录面积最大的几个区域
        sorted_id = sorted(range(len(stats)), key=lambda k: stats[k, 4], reverse=True)
        index = np.array(sorted_id[1:5])

        row = labels.shape[0]
        col = labels.shape[1]
        for i in range(row):
            for j in range(col):
                if not (index == labels[i, j]).any():
                    array_r[i, j] = 0

        # 提取边缘
        contours, hierarchy = cv2.findContours(array_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        approxes = []
        for cnt in range(len(contours)):
            triangle = []
            # 轮廓逼近
            epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
            approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
            approxes.append(approx)

        corner = Get_corner(approxes)

        pts1 = np.float32([corner[0], corner[1], corner[2], corner[3]])  # 左上、右上、左下、右下
        pts2 = np.float32([[0, 1000], [0, 0], [750, 1000], [750, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (750, 1000))
        dst = cv2.flip(dst, 0)
        # label[num_name] = cv2.warpPerspective(label[num_name], M, (750, 1000))
        images.append(1)
        images[num_name]=dst
        # cv2.imwrite(os.path.join(output_file, str(num_name) + '.jpg'), dst)
        num_name += 1
    # cv2.imwrite("output/test.jpg",images[0])
    return images

def mask_maker(image,threshold):
    row=image.shape[0]
    col=image.shape[1]
    mask=np.zeros((row,col),np.uint8)
    img = cv2.blur(image, (5, 5))
    b = np.array(img[:, :, 0]).astype(np.float32)
    g = np.array(img[:, :, 1]).astype(np.float32)
    r = np.array(img[:, :, 2]).astype(np.float32)
    img_r = r * 2 - b - g
    img_r[img_r < 0.0] = 0.

    array_r = img_r.astype(np.uint8)
    # 红色提取并二值化
    array_r[array_r < 100] = 0
    array_r[array_r >= 100] = 255
    for i in range(row):
        for j in range(col):
            if array_r[i, j] == 0:
                mask[i, j] = 255
    # if threshold:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     for i in range(row):
    #         for j in range(col):
    #             if mask[i, j] == 0:
    #                 gray[i, j] = 0
    #     gray = cv2.medianBlur(gray, 3)
    #     gray_edges = cv2.Canny(gray, threshold1=100, threshold2=150)
    #     # 存在一些孔洞 进行闭操作
    #     kernel = np.ones((15, 15), np.uint8)
    #     gray_edges = cv2.morphologyEx(gray_edges, cv2.MORPH_CLOSE, kernel)
    #     # 处理连通域
    #     for i in range(row):
    #         for j in range(col):
    #             if gray_edges[i, j] > 0:
    #                 gray[i - 1:i + 1, j - 1:j + 1] = 0
    #     # cv2.imwrite("output/edges.jpg",gray_edges)
    #     # 二值化 求连通域
    #     _, binary_gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    #     num, mb_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_gray, connectivity=4)
    #     # 记录面积最大的几个区域
    #     sorted_id = sorted(range(len(stats)), key=lambda k: stats[k, 4], reverse=True)
    #     # img_test=np.zeros((mb_labels.shape[0],mb_labels.shape[1],3),np.uint8)
    #     # img_test[mb_labels==0]=(0,0,255)
    #     # img_test[mb_labels==1]=(0,255,0)
    #     # img_test[mb_labels==2]=(255,0,0)
    #     # cv2.imwrite("output/test_label.jpg",img_test)
    #     index = sorted_id[0]
    #     for i in range(row):
    #         for j in range(col):
    #             if mb_labels[i, j] != index:
    #                 mask[i, j] = 0

    return mask

def calculate_new(image,mask,threshold_value):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 提取区域
    row = image.shape[0]
    col = image.shape[1]

    label = np.zeros_like(gray)

    count = 0
    sum = 0

    label[np.logical_and(mask == 255, gray < threshold_value)] = 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label, connectivity=8)
    for k in range(1, num_labels):
        if stats[k, cv2.CC_STAT_AREA] >= 200:
            label[labels == k] = 0

    # cv2.imwrite("mask.jpg", label)
    for i in range(row):
        for j in range(col):
            if label[i, j]:
                image[i, j] = [0, 0, 255]
                count += 1
            sum += 1

    print("比例为：", count / sum * 100)
    return count / sum * 100, image

def calculate_threshold(image,mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    row = image.shape[0]
    col = image.shape[1]
    num = 0
    for i in range(row):
        for j in range(col):
            if mask[i,j]:
                num += 1
    pixel = np.zeros((num))
    num = 0
    for i in range(row):
        for j in range(col):
            if mask[i,j]:
                pixel[num] = gray[i, j]
                num += 1
    mean = np.mean(pixel)
    std = np.std(pixel)
    print(mean)
    print(std)
    thre_value = mean - 3 * std
    print("阈值为：", thre_value)
    return thre_value

def calculate_compare(base,change,mask,filename,threshold):
    row=base.shape[0]
    col=base.shape[1]
    base_gray=cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    change_gray=cv2.cvtColor(change, cv2.COLOR_BGR2GRAY)
    mean_d=0
    num=0
    for m in range(row):
        for n in range(col):
            if mask[m,n]:
                mean_d+=int(change_gray[m,n])-int(base_gray[m,n])
                num+=1
    mean_d=mean_d/num
    count_all=0
    count_decrease=0
    count_increase=0
    img_change=np.zeros((row,col),np.int16)
    # # 计算原图的直方图
    # hist_b = cv2.calcHist([base_gray], [0], None, [256], [0, 256])
    # # 计算霉菌图的直方图
    # hist_c = cv2.calcHist([change_gray], [0], None, [256], [0, 256])
    # # 绘制原始图像直方图
    # plt.subplot(222), plt.plot(hist_b), plt.title("(base)"), plt.xlabel("x"), plt.ylabel("y")
    # # 绘制原始图像直方图
    # plt.subplot(222), plt.plot(hist_c), plt.title("(change)"), plt.xlabel("x"), plt.ylabel("y")
    for m in range(row):
        for n in range(col):
            if mask[m, n]:
                img_change[m,n]=int(change_gray[m,n])-int(base_gray[m,n])-mean_d


    for i in range(row):
        for j in range(col):
            if mask[i,j]:
                if img_change[i,j]<(0-threshold):
                    count_increase+=1
                    change[i,j]=[0, 0, 255]
                elif img_change[i,j]>threshold:
                    count_decrease+=1

                count_all+=1
    print("增加的霉菌占比（%）：",count_increase/count_all*100)
    # change_new=cv2.flip(change,0)

    cv2.imwrite(filename+"_"+str(count_increase/count_all*100)+"%.jpg",change)
    return count_increase/count_all*100,(count_increase-count_decrease)/count_all*100, change


def batch_process():
    # 3.14test
    path = "data314/35_98/"
    file_list = read_name(path)
    img_corrective = process(path)
    base1 = img_corrective[0]
    base2 = img_corrective[1]
    base3 = img_corrective[2]
    change_list1 = []
    change_list2 = []
    change_list3 = []
    change_namelist1 = []
    change_namelist2 = []
    change_namelist3 = []

    for i in range(3, len(file_list)):
        if i % 3 == 0:
            change_list1.append(img_corrective[i])
            change_namelist1.append(file_list[i])
        elif i % 3 == 1:
            change_list2.append(img_corrective[i])
            change_namelist2.append(file_list[i])
        else:
            change_list3.append(img_corrective[i])
            change_namelist3.append(file_list[i])

    change_num = [len(change_list1), len(change_list2), len(change_list3)]
    assert change_num[0] == change_num[1] and change_num[0] == change_num[2], "changelist长度不一致"

    for j in range(len(change_namelist1)):
        # 第一组
        compare_mask1 = mask_maker(base1, False)
        rate1, _, predict1 = calculate_compare(base1, change_list1[j], compare_mask1, 'predict/compare', 35)
        print(change_namelist1[j] + ":" + str(rate1) + "\n")
        # 第二组
        compare_mask2 = mask_maker(base2, False)
        rate2, _, predict2 = calculate_compare(base2, change_list2[j], compare_mask2, 'predict/compare', 35)
        print(change_namelist2[j] + ":" + str(rate2) + "\n")
        # 第三组
        compare_mask3 = mask_maker(base3, False)
        rate3, _, predict3 = calculate_compare(base3, change_list3[j], compare_mask3, 'predict/compare', 35)
        print(change_namelist3[j] + ":" + str(rate3) + "\n")

if __name__ == '__main__':
    # #阈值法
    img_corrective = process('data/')
    threshold_mask1 = mask_maker(img_corrective[0], True)
    # threshold_mask2 = mask_maker(img_corrective[2], True)
    threshold1 = calculate_threshold(img_corrective[0], threshold_mask1)
    # threshold2 = calculate_threshold(img_corrective[3], threshold_mask2)
    threshold_rate1, threshold_predict1 = calculate_new(img_corrective[0], threshold_mask1, threshold1)
    # threshold_rate2, threshold_predict2 = calculate_new(img_corrective[3], threshold_mask2, threshold2)
    print("2-2d阈值法比例：", threshold_rate1)
    # print("3-2d阈值法比例：", threshold_rate2)
    image1 = cv2.flip(img_corrective[0], 0)
    # image2 = cv2.flip(img_corrective[3], 0)
    cv2.imwrite('output/image.tif', image1)
    # cv2.imwrite('image/image2.tif', image2)

    # threshold_predict1 = cv2.flip(threshold_predict1, 0)
    # threshold_predict2 = cv2.flip(threshold_predict2, 0)
    #
    #
    # cv2.imwrite("predict/2_2d_threshold.jpg", threshold_predict1)
    # cv2.imwrite("predict/3_2d_threshold.jpg", threshold_predict2)
    #
    #
    #
    # #对比法
    # base1 = img_corrective[0]
    # base2 = img_corrective[2]
    # change1 = img_corrective[1]
    # change2 = img_corrective[3]
    # compare_mask1 = mask_maker(img_corrective[0], False)
    # compare_mask2 = mask_maker(img_corrective[2], False)
    # compare_rate1, _, compare_predict1 = calculate_compare(base1, change1, compare_mask1, 'predict/2_2d_compare', 45)
    # compare_rate2, _, compare_predict2 = calculate_compare(base2, change2, compare_mask2, 'predict/3_2d_compare', 45)
    # print("2-2d对比法比例：", compare_rate1)
    # print("3-2d对比法比例：", compare_rate2)

    #标签处理
    # label = []
    # label.append(cv2.imread('2-2d.jpg'))
    # label.append(cv2.imread('3-2d.jpg'))
    # img_corrective = process('label/', label)
    # img_corrective[0] = cv2.flip(img_corrective[0], 0)
    # img_corrective[1] = cv2.flip(img_corrective[1], 0)
    # label[0] = cv2.flip(label[0], 0)
    # label[1] = cv2.flip(label[1], 0)
    # label1 = np.zeros((1000, 750), np.uint8)
    # label2 = np.zeros((1000, 750), np.uint8)
    # num1 = 0
    # num2 = 0
    # for i in range(1000):
    #     for j in range(750):
    #         if label[0][i, j, 2] > 240:
    #             num1 += 1
    #             label1[i, j] = 1
    #         if label[1][i, j, 2] > 240:
    #             num2 += 1
    #             label2[i, j] = 1
    # # for i in range(1000):
    # #     for j in range(750):
    # #         if img_corrective[0][i, j, 2] > 180 & img_corrective[0][i, j, 0] < 50 & img_corrective[0][i, j, 1] < 50:
    # #             label1[i, j] = 1
    # #             num1 += 1
    # #         if img_corrective[1][i, j, 2] > 180 & img_corrective[1][i, j, 0] < 50 & img_corrective[1][i, j, 1] < 50:
    # #             label2[i, j] = 1
    # #             num2 += 1
    # cv2.imwrite('label/label2_2d.tif', label1)
    # cv2.imwrite('label/label3_2d.tif', label2)
    # cv2.imwrite('label/label1.tif', label[0])
    # cv2.imwrite('label/label2.tif', label[1])
    # # cv2.imwrite('label/test2.jpg', img_corrective[1])
    # print(num1)
    # print(num2)
    #




    # #精度评定
    # label1 = cv2.imread('label/label2_2d.jpg', 0)
    # label2 = cv2.imread('label/label3_2d.jpg', 0)
    # threshold1 = threshold_predict1
    # threshold2 = threshold_predict2
    # compare1 = compare_predict1
    # compare2 = compare_predict2
    #
    # #阈值法
    #
    # threshold_label_count1 = 0
    #
    # threshold_label_count2 = 0
    # threshold_TP1 = 0
    # threshold_FP1 = 0
    # threshold_FN1 = 0
    # threshold_TP2 = 0
    # threshold_FP2 = 0
    # threshold_FN2 = 0
    # for i in range(235, 850):
    #     for j in range(750):
    #         if threshold1[i, j, 2] / 255 == label1[i, j] and label1[i, j]:
    #             threshold_TP1 += 1
    #             threshold_label_count1 += 1
    #         if threshold1[i, j, 2] / 255 != label1[i, j] and label1[i, j]:
    #             threshold_FP1 += 1
    #             threshold_label_count1 += 1
    #         elif threshold1[i, j, 2] / 255 != label1[i, j] and threshold1[i, j, 2] == 255:
    #             threshold_FN1 += 1
    # threshold_IOU1 = threshold_TP1/(threshold_TP1 + threshold_FN1 + threshold_FP1)
    # threshold_rate1 = threshold_TP1/(threshold_TP1 + threshold_FP1)
    #
    # for i in range(235, 850):
    #     for j in range(750):
    #         if threshold2[i, j, 2] / 255 == label2[i, j] and label2[i, j]:
    #             threshold_TP2 += 1
    #             threshold_label_count2 += 1
    #         if threshold2[i, j, 2] / 255 != label2[i, j] and label2[i, j]:
    #             threshold_FP2 += 1
    #             threshold_label_count2 += 1
    #         elif threshold2[i, j, 2] / 255 != label2[i, j] and threshold2[i, j, 2] == 255:
    #             threshold_FN2 += 1
    # threshold_IOU2 = threshold_TP2 / (threshold_TP2 + threshold_FN2 + threshold_FP2)
    # threshold_rate2 = threshold_TP2 / (threshold_TP2 + threshold_FP2)
    #
    #
    #
    # #对比法
    #
    # compare_label_count1 = 0
    #
    # compare_label_count2 = 0
    # compare_TP1 = 0
    # compare_FP1 = 0
    # compare_FN1 = 0
    # compare_TP2 = 0
    # compare_FP2 = 0
    # compare_FN2 = 0
    #
    #
    # for i in range(235, 850):
    #     for j in range(750):
    #         if compare1[i, j, 2] / 255 == label1[i, j] and label1[i, j]:
    #             compare_TP1 += 1
    #             compare_label_count1 += 1
    #         if compare1[i, j, 2] / 255 != label1[i, j] and label1[i, j]:
    #             compare_FP1 += 1
    #             compare_label_count1 += 1
    #         elif compare1[i, j, 2] / 255 != label1[i, j] and compare1[i, j, 2] == 255:
    #             compare_FN1 += 1
    # compare_IOU1 = compare_TP1/(compare_TP1 + compare_FN1 + compare_FP1)
    # compare_rate1 = compare_TP1/(compare_TP1 + compare_FP1)
    #
    # for i in range(235, 850):
    #     for j in range(750):
    #         if compare2[i, j, 2] / 255 == label2[i, j] and label2[i, j]:
    #             compare_TP2 += 1
    #             compare_label_count2 += 1
    #         if compare2[i, j, 2] / 255 != label2[i, j] and label2[i, j]:
    #             compare_FP2 += 1
    #             compare_label_count2 += 1
    #         elif compare2[i, j, 2] / 255 != label2[i, j] and compare2[i, j, 2] == 255:
    #             compare_FN2 += 1
    # compare_IOU2 = compare_TP2/(compare_TP2 + compare_FN2 + compare_FP2)
    # compare_rate2 = compare_TP2/(compare_TP2 + compare_FP2)
    #
    #
    # print('阈值法TP1:', threshold_TP1)
    # print('阈值法FP1:', threshold_FP1)
    # print('阈值法FN1:', threshold_FN1)
    # print('阈值法IOU1:', threshold_IOU1)
    # print('阈值法比例1:', threshold_rate1)
    # print('阈值法TP2:', threshold_TP2)
    # print('阈值法FP2:', threshold_FP2)
    # print('阈值法FN2:', threshold_FN2)
    # print('阈值法IOU2:', threshold_IOU2)
    # print('阈值法比例2:', threshold_rate2)
    # print('对比法TP1:', compare_TP1)
    # print('对比法FP1:', compare_FP1)
    # print('对比法FN1:', compare_FN1)
    # print('对比法IOU1:', compare_IOU1)
    # print('对比法比例1:', compare_rate1)
    # print('对比法TP2:', compare_TP2)
    # print('对比法FP2:', compare_FP2)
    # print('对比法FN2:', compare_FN2)
    # print('对比法IOU2:', compare_IOU2)
    # print('对比法比例2:', compare_rate2)

    # #精度评定2
    # TP = 0
    # FP = 0
    # FN = 0
    # TP_M = 0
    # predict = cv2.imread("predict/pre2.tif", 0)
    # label = cv2.imread("label/label3_2d.tif", 0)
    # row, col = predict.shape
    # for i in range(row):
    #     for j in range(col):
    #         if predict[i, j] == label[i, j]:
    #             TP += 1
    #             if label[i, j]:
    #                 TP_M += 1
    #         elif predict[i, j]:
    #             FN += 1
    #         else:
    #             FP += 1
    # print("TP:", TP)
    # print("TP_M:", TP_M)
    # print("FP:", FP)
    # print("FN:", FN)
    # print("IOU:", TP_M/(TP_M+FP+FN))
    # print("rate:", TP_M/(TP_M+FP))





