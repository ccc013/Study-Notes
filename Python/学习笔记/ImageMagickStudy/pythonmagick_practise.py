#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Copyright 2018 JD Inc.
# Author: Luocai

"""
from __future__ import print_function
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
from wand.display import display
import os


# 画一个纯白背景，并保存
def draw_bg(width, height, filename):
    with Image(width=width, height=height, background=Color('white')) as img:
        img.save(filename=filename)


# 将图片 img 放在背景图片，并设置存放的位置
def composite(img_back, img, left, top, target):
    with Image(filename=img_back) as w:
        with Image(filename=img) as r:
            with Drawing() as draw:
                draw.composite(operator='atop',
                               left=left, top=top,
                               width=r.width,
                               height=r.height,
                               image=r)
                draw(w)
                w.save(filename=target)


# 旋转和调整图片大小
def resize_and_rotate(filename, resize_ratio, rotate_angle, target):
    with Image(filename=filename) as img:
        print('original size', img.size)
        names = os.path.splitext(target)
        save_name = names[0] + '_%d' + names[1]
        for r in range(1, 4):
            with img.clone() as i:
                i.resize(int(i.width * resize_ratio * r), int(i.height * resize_ratio * r))
                i.rotate(rotate_angle * r)
                i.save(filename=save_name % r)
                display(i)


def read_image():
    with Image(filename='front.png') as img:
        print('width=', img.width)
        print('height=', img.height)
        print('size=', img.size)


if __name__ == '__main__':
    # draw_bg(1000, 800, 'background.jpg')
    # composite('background.jpg', 'front.png', 100, 200, 'final2.jpg')
    # resize_and_rotate('front.png', 0.5, 90, 'resize_rotate.png')
    read_image()
