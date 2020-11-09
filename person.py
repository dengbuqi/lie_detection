#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : person.py
# @Time      : 2020/9/24 11:43
# @Author    : 陈嘉昕
# @Demand    : 构建人物对象类


class Person:

    def __init__(self):
        # 平均脸色rpg
        self.average_cheek_color = [0, 0, 0]
        # 眼睛高宽比阈值
        self.eye_aspect_ratio = 0
        # 嘴唇高宽比阈值
        self.lips_aspect_ratio = 0
        # 平均眨眼次数
        self.average_number_of_blinks = 0
        # 平均咧嘴次数
        self.average_number_of_lip_pursing = 0

    """设定个人眨眼的平均次数"""

    def set_average_number_of_blinks(self, num, seconds=1):
        if num > 0 and seconds > 0:
            self.average_number_of_blinks = num / seconds

    """设定个人脸颊变色的平均次数"""

    def set_average_cheek_color(self, cheek_color):
        self.average_cheek_color = cheek_color

    """计算平均颜色"""

    def calculate_average_color(self, color):
        if self.average_cheek_color[0] == 0 and self.average_cheek_color[1] == 0 and self.average_cheek_color[2] == 0:
            self.average_cheek_color = color
        else:
            self.average_cheek_color[2] = (self.average_cheek_color[2] + color[2]) / 2.0
            self.average_cheek_color[1] = (self.average_cheek_color[1] + color[1]) / 2.0
            self.average_cheek_color[0] = (self.average_cheek_color[0] + color[0]) / 2.0

    """设定撅嘴的平均次数"""

    def set_average_number_of_lip_pursing(self, num):
        self.average_number_of_lip_pursing = num

    """计算眼睛的长宽比阈值"""

    def calculate_average_eye_aspect_ratio(self, ratio):
        if self.eye_aspect_ratio == 0:
            self.eye_aspect_ratio = ratio
        else:
            self.eye_aspect_ratio = (self.eye_aspect_ratio + ratio) / 2.0

    """计算撅嘴的阈值"""

    def calculate_average_lips_aspect_ratio(self, ratio):
        if self.lips_aspect_ratio == 0:
            self.lips_aspect_ratio = ratio
        else:
            self.lips_aspect_ratio = (self.lips_aspect_ratio + ratio) / 2.0
