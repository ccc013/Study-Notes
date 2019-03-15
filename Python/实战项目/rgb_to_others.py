#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Copyright 2018 JD Inc.
# Author: Luocai

彩色图片转成卡通图
"""
import os
import cv2
import numpy as np
import time
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import matplotlib.pyplot as plt
from PIL import Image


def rgb_to_gray(image_name, output_name):
    img_rgb = cv2.imread(image_name)
    # 转换为灰度
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # 保存图片
    cv2.imwrite(output_name, img_gray)
    # 展示图片
    # cv2.imshow('original', img_rgb)
    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def rgb_to_cartoon(image_name, output_name):
    # 缩减像素采样的数目
    num_down = 2
    # 定义双边滤波的数目
    num_bilateral = 7

    img_rgb = cv2.imread(image_name)
    # 用高斯金字塔降低取样
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    # 重复使用小的双边滤波替代一个大的滤波
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9,
                                        sigmaColor=9, sigmaSpace=7)
    # 上采样图片到原始大小
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    # 转换为灰度并且使其产生中等的模糊
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # 采用 img_color 避免因为采样造成后续合成图片时候的尺寸不一致
    img_blur = cv2.medianBlur(img_gray, 7)
    # 检测边缘并且增强其效果
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize=9, C=2)
    # 转换回彩色图像
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    # 保存图片
    cv2.imwrite(output_name, img_cartoon)
    # 展示图片
    # cv2.imshow('original', img_rgb)
    # cv2.imshow('cartoonise', img_cartoon)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def dodgeNaive(image, mask):
    # determine the shape of the input image
    width, height = image.shape[:2]

    # prepare output argument with same size as image
    blend = np.zeros((width, height), np.uint8)

    for col in xrange(width):
        for row in xrange(height):
            # do for every pixel
            if mask[col, row] == 255:
                # avoid division by zero
                blend[col, row] = 255
            else:
                # shift image pixel value by 8 bits
                # divide by the inverse of the mask
                tmp = (image[col, row] << 8) / (255 - mask)

                # make sure resulting value stays within bounds
                if tmp > 255:
                    tmp = 255
                    blend[col, row] = tmp

    return blend


def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)


def rgb_to_sketch(image_name, output_name):
    img_rgb = cv2.imread(image_name)
    # convert image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # invert to obtain a negative
    img_gray_inv = 255 - img_gray
    # apply a Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
    # blend the grayscale image with the blurred negative
    img_blend = dodgeV2(img_gray, img_blur)
    # img_blend = burnV2(img_gray, img_blur)
    cv2.imshow('pencil sketch', img_blend)
    cv2.imwrite(output_name, img_blend)


def rgb_to_sketch_v2(image_name, output_name):
    img_rgb = cv2.imread(image_name)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
    img_blend = cv2.divide(img_gray, img_blur, scale=256)

    result = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('pencil sketch', result)
    cv2.imwrite(output_name, result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def rgb_to_sketch_lbp(image_name, output_name):
    # settings for LBP
    radius = 3.5
    n_points = 8 * radius

    img_rgb = cv2.imread(image_name)
    # 显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
    image1 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    # plt.subplot(131)
    # plt.imshow(image1)

    # 转换为灰度图显示
    image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # plt.subplot(132)
    # plt.imshow(image, cmap='gray')

    # 处理
    lbp = local_binary_pattern(image, n_points, radius)
    cv2.imwrite(output_name, lbp)
    # plt.subplot(133)
    # plt.imshow(lbp, cmap='gray')
    # plt.show()


def rgb_to_black_white(image_name, output_name):
    img_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    # Convert grayscale image to binary
    # thresh, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    img_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(output_name, img_bw)


def rgb_to_black_white_v2(image_name, output_name):
    img_rgb = cv2.imread(image_name)


def main(src_path, dst_path, operation='gray', is_rewrite=False):
    print 'transform rgb to %s' % operation
    count = 0
    for filename in os.listdir(src_path):
        origin_img_path = os.path.join(src_path, filename)
        dst_img_path = os.path.join(dst_path, filename)
        if os.path.exists(dst_img_path):
            if is_rewrite is False:
                continue
        print '%s to %s' % (origin_img_path, dst_img_path)
        if operation == 'gray':
            rgb_to_gray(origin_img_path, dst_img_path)
        elif operation == 'cartoon':
            rgb_to_cartoon(origin_img_path, dst_img_path)
        elif operation == 'blackwhite':
            rgb_to_black_white(origin_img_path, dst_img_path)
        elif operation == 'sketch':
            # rgb_to_sketch(origin_img_path, dst_img_path)
            # rgb_to_sketch_v2(origin_img_path, dst_img_path)
            rgb_to_sketch_lbp(origin_img_path, dst_img_path)
        else:
            raise ValueError('Invalid operation!')
        count += 1
        break
    print 'total transform %d images' % count


if __name__ == '__main__':
    print 'start'
    image_list = []
    cartoon_image_path = 'D:\Work\practiseCode\cartoon_images\\'
    origin_image_path = 'D:\Work\practiseCode\images\\'
    gray_image_path = 'D:\Work\practiseCode\gray_images\\'
    blackwhite_image_path = 'D:\Work\practiseCode\\blackwhite_images\\'
    sketch_image_path = 'D:\Work\practiseCode\\sketch_images\\'
    lbp_sketch_image_path = 'D:\Work\practiseCode\\lbp_sketch_images\\'
    t1 = time.time()
    main(origin_image_path, lbp_sketch_image_path, operation='sketch', is_rewrite=True)
    print 'finish, using %fs' % (time.time() - t1)
