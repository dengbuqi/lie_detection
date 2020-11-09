#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : pursed_lips_detector.py
# @Time      : 2020/9/25 11:29
# @Author    : 陈嘉昕
# @Demand    : 嘴角咧嘴次数检测


import cv2
from imutils import face_utils
from scipy.spatial import distance


class PursedLipsDetector:
    # 获取嘴巴的位置坐标点
    (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    """参数初始化"""

    def __init__(self):
        # 每一帧中咧嘴的次数
        self.frame_pursed_counter = 0
        # 总共咧嘴的次数
        self.total_pursed_counter = 0
        # 口宽高比的阈值
        self.LIPS_ASPECT_RATIO_THRESHOLD = -1
        # 咧嘴持续的帧数阈值
        self.PURSED_LIPS_CONSECUTIVE_FRAMES = -1

    """开始检测"""

    def detect(self, frame, face_region):
        # 计算口宽高比
        LAR, mouth = self.lips_aspect_ratio(face_region, consider_smile=True)
        # 在帧中画出嘴巴的轮廓
        self.draw_mouth(frame, mouth)

        # 检查口宽高比是否低于阈值，如果是，增加咧嘴次数
        if LAR < self.LIPS_ASPECT_RATIO_THRESHOLD:
            self.frame_pursed_counter += 1
        else:
            # 在一定数量的帧中，被观测者的嘴一直处于咧嘴状态，就将总数+1
            if self.frame_pursed_counter >= self.PURSED_LIPS_CONSECUTIVE_FRAMES:
                self.total_pursed_counter += 1

            # 重置帧中咧嘴次数
            self.frame_pursed_counter = 0
            # 打印总共咧嘴的次数
            self.print_pursed_lips(frame, self.total_pursed_counter, LAR)

    """计算咧嘴宽高阈值"""

    def calculate_lips_aspect_ratio_threshold(self, lips_aspect_ratio):
        # 阈值按第一次计算的宽高比的0.8来计算
        self.LIPS_ASPECT_RATIO_THRESHOLD = lips_aspect_ratio * 0.8
        # 设置持续检测的帧数为4
        self.PURSED_LIPS_CONSECUTIVE_FRAMES = 4

    """获取咧嘴总数"""

    def get_and_reset_number_of_lip_pursing(self):
        # 咧嘴总数
        retVal = self.total_pursed_counter
        # 重置为0
        self.total_pursed_counter = 0
        self.frame_pursed_counter = 0
        # 返回咧嘴总数
        return retVal

    """计算嘴巴宽高比"""

    @staticmethod
    def lips_aspect_ratio(face_region, consider_smile=False):
        # 提取嘴部坐标，然后使用坐标计算嘴部长径比
        mouth = face_region[PursedLipsDetector.mouth_start:PursedLipsDetector.mouth_end]

        # 计算上嘴唇径长
        top_lip1 = distance.euclidean(mouth[2], mouth[13])
        top_lip2 = distance.euclidean(mouth[3], mouth[14])
        top_lip3 = distance.euclidean(mouth[4], mouth[15])

        # 计算下嘴唇径长
        bottom_lip1 = distance.euclidean(mouth[8], mouth[17])
        bottom_lip2 = distance.euclidean(mouth[9], mouth[18])
        bottom_lip3 = distance.euclidean(mouth[10], mouth[19])

        # 上唇和下唇中间的距离，以检测人是否在微笑（在这种情况下，嘴唇会显得更薄）
        smile = distance.euclidean(mouth[14], mouth[18])
        if consider_smile and smile > 3:
            return 5, mouth

        # 嘴巴宽度，检测是否咧嘴
        mouth_width = distance.euclidean(mouth[0], mouth[6])
        LAR = (top_lip1 + top_lip2 + top_lip3 + bottom_lip1 + bottom_lip2 + bottom_lip3) / (6.0 * mouth_width)
        return LAR, mouth

    """画出嘴巴的轮廓"""

    @staticmethod
    def draw_mouth(frame, mouth_contour):
        # 在帧中画出嘴巴的轮廓
        cv2.drawContours(frame, [mouth_contour], -1, (255, 255, 51), 1)

    """打印总共咧嘴的次数"""

    @staticmethod
    def print_pursed_lips(frame, number, MAR=-1):
        # 在帧上打印撅嘴的总数
        cv2.putText(frame, "Pursed lips: {}".format(number), (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        # 在帧上打印帧的唇高宽比
        cv2.putText(frame, "LAR: {:.4f}".format(MAR), (500, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
