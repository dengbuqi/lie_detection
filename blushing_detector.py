#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : blushing_detector.py
# @Time      : 2020/9/25 9:29
# @Author    : 陈嘉昕
# @Demand    : 脸颊变化检测


import cv2
import numpy


class BlushingDetector:
    # 左脸颊坐标点
    right_cheek_idx = [1, 2, 3, 4, 48, 31, 36]

    # 右脸颊坐标点
    left_cheek_idx = [12, 13, 14, 15, 45, 35, 54]

    # RGB变化阈值
    RED_CHANGE_ALLOWANCE = 35
    GREEN_CHANGE_ALLOWANCE = 80
    BLUE_CHANGE_ALLOWANCE = 50

    # RGB总改变阈值
    RGB_CHANGE_ALLOWANCE = 35

    """初始化参数"""

    def __init__(self):
        # 在一帧内，发生脸红的次数
        self.blushing_frame_counter = 0
        # 脸颊状态保持的帧数
        self.BLUSHING_CONSECUTIVE_FRAMES = 10
        # 脸颊的平均RGB
        self.AVERAGE_CHEEK_COLOR = [0, 0, 0]
        # 发生脸红的次数
        self.blushing_occurred_counter = 0
        # 实时数据
        self.R = 0
        self.G = 0
        self.B = 0

    """开始检测"""

    def detect(self, frame, gray_frame, face_region):
        # 提取右脸颊和左脸颊的坐标
        right_cheek = face_region[self.right_cheek_idx]
        left_cheek = face_region[self.left_cheek_idx]

        # 画出帧中脸颊的范围
        self.draw_cheeks(frame, right_cheek, left_cheek)

        # 使用坐标计算脸颊的平均颜色
        cheeks_color = self.calculate_cheeks_color(frame, gray_frame, right_cheek, left_cheek)

        # 脸颊实时颜色
        self.R = cheeks_color[0]
        self.G = cheeks_color[1]
        self.B = cheeks_color[2]

        # 检查每一帧中脸红的发生，发生脸红，blushing_frame_counter + 1，标志""BLU ""表示观测者发生脸红的次数
        blushing = self.is_blushing(cheeks_color)
        if blushing:
            self.blushing_frame_counter += 1
            print("BLU " + str(self.blushing_frame_counter))

        # 在一定数量的帧中，被观察者一直保持脸红的状态，标志"BLUSHING"表示观测者一直脸红
        if self.blushing_frame_counter >= self.BLUSHING_CONSECUTIVE_FRAMES:
            # 打印脸颊的轮廓
            self.print_blushing(frame)

            # 发生脸红
            print("BLUSHING")

            # 将探测脸红的计时器置0
            self.blushing_frame_counter = 0
            self.blushing_occurred_counter += 1

    """判断脸红"""

    def is_blushing(self, temp_color):
        # 计算脸颊颜色变化，和平均脸颊颜色之间的红、绿、蓝三个方面的差
        red_change = abs(temp_color[2] - self.AVERAGE_CHEEK_COLOR[2])
        green_change = abs(temp_color[1] - self.AVERAGE_CHEEK_COLOR[1])
        blue_change = abs(temp_color[0] - self.AVERAGE_CHEEK_COLOR[0])

        # 计算总的该变量
        accumulated_change = red_change + green_change + blue_change

        # 返回判断结果
        return accumulated_change > self.RGB_CHANGE_ALLOWANCE \
               and 5 < red_change < self.RED_CHANGE_ALLOWANCE \
               and 0 < green_change < self.GREEN_CHANGE_ALLOWANCE \
               and 0 < blue_change < self.BLUE_CHANGE_ALLOWANCE

    """设定初始脸颊平均颜色"""

    def set_average_cheek_color(self, average_cheek_color):
        # 直接将一开始计算的平均脸颊颜色设定为判定标准
        self.R = average_cheek_color[0]
        self.G = average_cheek_color[1]
        self.B = average_cheek_color[2]
        self.AVERAGE_CHEEK_COLOR = average_cheek_color

    """获取脸颊变化的次数"""

    def get_number_of_blushing_occurred_and_reset(self):
        # 总共脸颊变化次数
        retVal = self.blushing_occurred_counter

        # 重置数据
        self.blushing_occurred_counter = 0
        self.blushing_frame_counter = 0

        # 返回脸颊变化次数
        return retVal

    """使用坐标计算出脸颊的平均颜色"""

    @staticmethod
    def calculate_cheeks_color(frame, gray_frame, right_cheek, left_cheek):
        # 一个脸颊的框架
        extracted_cheeks_frame = numpy.zeros(frame.shape, numpy.uint8)

        # 使用mask计算右边脸颊的颜色
        mask = numpy.zeros(gray_frame.shape, numpy.uint8)
        cv2.drawContours(mask, [right_cheek], -1, 255, -1)

        # 计算脸颊的颜色
        right_cheek_color = cv2.mean(frame, mask)
        cv2.drawContours(extracted_cheeks_frame, [right_cheek], -1, right_cheek_color, -1)

        # 使用mask计算左边脸颊的颜色
        mask = numpy.zeros(gray_frame.shape, numpy.uint8)
        cv2.drawContours(mask, [left_cheek], -1, 255, -1)

        # 计算脸颊的颜色
        leftCheekColor = cv2.mean(frame, mask)
        cv2.drawContours(extracted_cheeks_frame, [left_cheek], -1, leftCheekColor, -1)

        # 计算脸颊的平均颜色
        average_cheek_color = [(leftCheekColor[0] + right_cheek_color[0]) / 2,
                               (leftCheekColor[1] + right_cheek_color[1]) / 2,
                               (leftCheekColor[2] + right_cheek_color[2]) / 2]

        # 将脸颊的颜色变化打印在帧中
        cv2.putText(extracted_cheeks_frame, "BGR: {:.0f}".format(average_cheek_color[0])
                    + " {:.0f}".format(average_cheek_color[1])
                    + " {:.0f}".format(average_cheek_color[2]), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 答应连接的框架
        cv2.imshow('Cheeks', extracted_cheeks_frame)

        # 返回脸颊的平均颜色
        return average_cheek_color

    """画出脸颊在帧中的范围"""

    @staticmethod
    def draw_cheeks(frame, left_cheek, right_cheek):
        # 画出右脸颊
        cv2.drawContours(frame, [right_cheek], -1, (255, 0, 0))

        # 画出左脸颊
        cv2.drawContours(frame, [left_cheek], -1, (200, 0, 0))

    @staticmethod
    def print_blushing(frame):
        cv2.putText(frame, "BLUSHING", (150, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
