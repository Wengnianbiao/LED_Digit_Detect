#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2020/10/7 10:55
# @Author: Biao
# @File: main
import cv2
import imutils
import math
import numpy as np
from imutils import contours
from detect_code.utils import drawrect,crop_roi,gamma
import detect_code.utils as utils


def img_detect(frame):
    # 定义每一个数字对应的数码管字段
    # 具体的编码规则结合图sigal_digit_distribute.png图片
    # 根据长宽比判断1(长宽比小于0.5的话就是1)
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 1, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }
    # 裁剪特定区域内的数字
    # 先裁剪后对图像处理会提供性能
    img = utils.crop_roi(frame)
    # 对原图像先进行y变换
    img = utils.gamma(img)
    # 读取为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('./testimg/source.png',img)
    # 按照阈值190二值化图像
    ret,thresh = cv2.threshold(img,190,255,cv2.THRESH_BINARY)
    cv2.imwrite('./testimg/thresh.png',thresh)

    # 对原图进行膨胀
    # kerneld = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    # thresh = cv2.dilate(thresh,kerneld)
    # cv2.imwrite('testimg/dilated.png',thresh)

    # # 再进行腐蚀
    # kerneld = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))
    # erode = cv2.erode(thresh,kerneld)
    # cv2.imwrite('testimg/erode.png',erode)

    # 查找轮廓:在二值化的图像中查找轮廓，函数对原图像的要求->背景应当是黑色，数字为白色
    conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 针对不同版本，都只取轮廓，图像和层次不需要
    conts = imutils.grab_contours(conts)
    # canvas = np.ones((1000,1000, 1), dtype="uint8")
    # canvas[:] = 255
    # for j in range(len(conts)):
    #     for i in range(len(conts[j])):
    #         x,y = conts[j][i][0]
    #         canvas[y][x] = 0
    # cv2.imwrite('canvas.png',canvas)
    # 绘制矩阵并保存
    drawrect(thresh,conts,'./testimg/rect.png')
    digitCnts = []
    for c in conts:
        # 返回根据轮廓绘制后的矩形四个坐标点
        # 就是根据得到的轮廓点阵，分别找到4个顶点最边角的点进行绘制即可
        (x, y, w, h) = cv2.boundingRect(c)
        # 设置宽大于5，高大于20的合理边框加入候选区域
        if (w > 5) and (h > 20):
            digitCnts.append(c)
    # 从左到右对这些轮廓进行排序
    if not digitCnts:
        return None
    # 源码中就是按照边框的x进行排序
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    digits = []

    # 对获取的轮廓从左至右进行处理
    # print('一共有{}个数字：'.format(len(digitCnts)))
    for c in digitCnts:
        # 获取ROI区域
        (x, y, w, h) = cv2.boundingRect(c)
        # 计算长宽比用于区别1和其他数字
        rate = w/h
        # 如果长宽比小于0.5就是1直接返回即可
        if rate < 0.5:
            digits.append(str(1))
            continue

        roi = thresh[y:y + h+1, x:x + w+1]
        # 分别计算每一段的宽度和高度:这里的高度和宽度根据实际的数字大小来进行计算
        # 永杰的数字
        (roiH, roiW) = roi.shape
        # 水平方向晶体管:0,3,6中H:W = 5/6.
        (level_w, level_h) = (int(roiW * 0.33),int(roiH * 0.22))
        # 竖直方向晶体管:1,2,4,5中H:W = 1:1.
        (vertical_w, vertical_h) = (int(roiW * 0.28),int(roiH * 0.22))
        # 水平偏移量设置为宽度的5%
        OFFSET_W = int(1)
        # # 竖直偏移量设置为宽度的5%
        OFFSET_H = int(1)
        # 根据sigal_digit_distribute.png图片加上实际的边框定义每段数码管
        # cv2中读取图像顺序img[h][w]
        # 坐标格式:
        segments = [
            ((OFFSET_H,vertical_w),(level_h,w-vertical_w-OFFSET_W)),           # 上0
            ((level_h+OFFSET_H,OFFSET_W),(level_h+vertical_h+OFFSET_H,vertical_w+OFFSET_W)),   # 左上1
            ((level_h+OFFSET_H,w-vertical_w-OFFSET_W),(vertical_h+level_h+OFFSET_H,w-OFFSET_W)),         # 右上2
            ((vertical_h+level_h+OFFSET_H,vertical_w+OFFSET_W),(h-level_h-vertical_h-OFFSET_H, w-vertical_w-OFFSET_W)),  # 中间3
            ((h-vertical_h-level_h-OFFSET_H,OFFSET_W),(h-level_h-OFFSET_H,vertical_w+OFFSET_W)),            # 左下4
            ((h-vertical_h-level_h-OFFSET_H,w-vertical_w-2*OFFSET_W),(h-level_h-OFFSET_H,w-2*OFFSET_W)),        # 右下5
            ((h-level_h,vertical_w+OFFSET_W), (h,w-vertical_w-OFFSET_W))              # 下6
        ]

        on = [0] * len(segments)
        # 循环遍历数码管中的每一段
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # 将对于的数码管坐标截取
            segROI = roi[int(xA):int(xB), int(yA):int(yB)]
            # 统计其中非0的像素个数(即非黑),非0代表此数码管是亮的
            h = 1e-8
            total = cv2.countNonZero(segROI) + h
            # print(total)
            area = (xB - xA) * (yB - yA) + h
            # print(area)
            # print(total/area)

            # 如果非零区域的个数大于整个区域的一半，则认为该段是亮的
            if total / float(area) > 0.5:
                on[i] = 1

        # 进行数字查询并显示结果
        # 如果没有对应的标签直接跳过避免报错
        if DIGITS_LOOKUP.get(tuple(on)) is None:
            continue
        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(str(digit))

    ret = ''.join(digits)

    return ret


if __name__ == '__main__':
    f = cv2.imread('../digitalsets/46.png')
    rest = img_detect(f)
    print(rest)
