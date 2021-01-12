#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2020/10/7 10:53
# @Author: Biao
# @File: utils
"""
提供各种针对项目要求的数字图像处理函数
"""
import cv2
import numpy as np


def drawrect(image, cnts, savepath):
    """
    画出边框
    :param image:
    :param cnts:
    :param savepath:
    :return:
    """
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x - 1, y - 1), (x + w, y + h), (255, 255, 255), 1)

    cv2.imwrite(savepath, image)


def crop_roi(image):
    """
    裁剪roi区域即数字区域
    :param image:
    :return:
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    # 图像逆时针旋转固定角度的旋转矩阵
    rot_mat = cv2.getRotationMatrix2D(center, 14, 1)
    # 将旋转矩阵应用到仿射变换
    rotated = cv2.warpAffine(image, rot_mat, (w, h))
    cv2.imwrite('testimg/rotated.png', rotated)
    roi = rotated[750:824, 850:1104]

    return roi


def gamma(image, coe=3):
    """
    对图像使用ganmma矫正以增加对比度:coe为系数
    :param image:
    :param coe:
    :return:
    """
    res = np.power(image / float(np.max(image)), coe)
    res = 255 * res
    res = res.astype(np.uint8)

    return res


if __name__ == '__main__':
    img = cv2.imread('../digitalsets/46.png')
    img = crop_roi(img)
    cv2.imwrite('crop.png', img)
    img = gamma(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
    cv2.imwrite('roi.png',img)
