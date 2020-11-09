#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : blink_detector.py
# @Time      : 2020/9/25 8:18
# @Author    : 陈嘉昕
# @Demand    : 眨眼次数检测


import cv2
from imutils import face_utils
from scipy.spatial import distance


class BlinkDetector:

    # 左眼特征点坐标
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

    # 右眼特征点坐标
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    """参数初始化"""

    def __init__(self):
        # 每帧眨眼的次数
        self.frame_blink_counter = 0
        # 总共眨眼的次数
        self.total_blink_counter = 0
        # 眼睛长宽比阈值，之后通过计算获取，拟取正常长宽比的0.7
        self.EYE_ASPECT_RATIO_THRESHOLD = -1
        # 一次眨眼持续的帧数
        self.BLINK_CONSECUTIVE_FRAMES = -1

    """开始检测"""

    def detect(self, frame, face_region):

        # 计算长宽比，左右眼坐标
        EAR, left_eye, right_eye = self.calculate_eye_aspect_ratio(face_region)

        # 画出帧上面的眼睛边框
        self.draw_eyes(frame, left_eye, right_eye)

        # 如果眼睛的长宽比低于设定的阈值，就表示这一帧中眼睛开始出现闭合，frame_blink_counter + 1
        if EAR < self.EYE_ASPECT_RATIO_THRESHOLD:
            self.frame_blink_counter += 1

        else:
            # 在一定数量的帧中，眼睛一直处于闭眼的状态，表示眼睛眨眼一次，total_blink_counter + 1
            if self.frame_blink_counter >= self.BLINK_CONSECUTIVE_FRAMES:
                self.total_blink_counter += 1

            # 将frame_blink_counter置0，进行下一次的计算
            self.frame_blink_counter = 0

        # 在帧中打印眼动数据
        self.print_blinks(frame, self.total_blink_counter, EAR)

    """计算眼睛长宽比阈值"""

    def calculate_eye_aspect_ratio_threshold(self, eye_aspect_ratio):
        # 将阈值设定为正常长宽比的0.7
        self.EYE_ASPECT_RATIO_THRESHOLD = eye_aspect_ratio * 0.7

        # 将一次眨眼持续帧数设定为1
        self.BLINK_CONSECUTIVE_FRAMES = 1

    """返回总共眨眼数，并重新设定眨眼数"""

    def get_and_reset_number_of_blinks(self):
        # 总共眨眼数
        retVal = self.total_blink_counter

        # 重置眨眼数
        self.frame_blink_counter = 0
        self.total_blink_counter = 0

        # 返回眨眼总数
        return retVal

    """实时计算眼睛的长宽比"""

    @staticmethod
    def calculate_eye_aspect_ratio(face_region):

        # 提取左右眼坐标
        left_eye = face_region[BlinkDetector.left_eye_start:BlinkDetector.left_eye_end]
        right_eye = face_region[BlinkDetector.right_eye_start:BlinkDetector.right_eye_end]

        # 计算左眼和右眼长宽比
        left_EAR = BlinkDetector.eye_aspect_ratio(left_eye)
        right_EAR = BlinkDetector.eye_aspect_ratio(right_eye)

        # 返回左右眼长宽比的均值，左眼坐标，右眼坐标
        return (left_EAR + right_EAR) / 2.0, left_eye, right_eye

    """计算眼睛长宽比"""

    @staticmethod
    def eye_aspect_ratio(eye):
        # 眼睛竖直欧式距离
        eye_height_1 = distance.euclidean(eye[1], eye[5])
        eye_height_2 = distance.euclidean(eye[2], eye[4])

        # 眼睛水平欧氏距离是一样的
        eye_width = distance.euclidean(eye[0], eye[3])

        # 计算眼睛长宽比
        EAR = (eye_height_1 + eye_height_2) / (2.0 * eye_width)

        # 返回眼睛长宽比
        return EAR

    """在帧中画出眼睛边框"""

    @staticmethod
    def draw_eyes(frame, left_eye, right_eye):
        # 计算左眼和右眼的凸出弧线
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)

        # 可视化左眼和右眼
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    """将眼动数据打印在帧中"""

    @staticmethod
    def print_blinks(frame, blinks, EAR=-1):
        # 打印眨眼的次数
        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 打印眼睛的实时长宽比
        cv2.putText(frame, "EAR: {:.4f}".format(EAR), (200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
