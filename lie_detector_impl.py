#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : lie_detector_impl.py
# @Time      : 2020/9/25 15:40
# @Author    : 陈嘉昕
# @Demand    : 测谎方法的实现


import os
import pandas as pd
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import blink_detector
import blushing_detector
import person
import pursed_lips_detector
import build_network
import time
import logging
import csv

# shape路径，用于寻找人物特征点
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# 帧,用于前期适应性检测
NUMBER_OF_FRAMES_TO_INSPECT = 200

# 用于前期眼部适应性检测
NUMBER_OF_FRAMES_TO_INSPECT_EYES = 25

# 系统日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')


class LieDetector:

    def __init__(self):
        # 初始化的人脸检测器(OG-based)
        self.detector = dlib.get_frontal_face_detector()

        # 创建人脸特征点检测器
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

        # 创建一个视频流
        self.video_stream = VideoStream(src=0).start()
        self.file_stream = False

        # 暂停1s
        time.sleep(1.0)

        # 初始化眨眼检测、咧嘴检测、脸色变化检测
        self.blink_detector = blink_detector.BlinkDetector()
        self.pursed_lips_detector = pursed_lips_detector.PursedLipsDetector()
        self.blushing_detector = blushing_detector.BlushingDetector()

        # 创建对象
        self.person = person.Person()

        # 正在检测的帧数
        self.frame_counter = 0

        # 测谎的次数
        self.questions_counter = 1

        # 计时器
        self.seconds = 0

    """开始测试"""

    def process(self, tag='7', file_name="data/video_data_for_lie_training.csv"):

        # 获取时间
        timeBefore = time.time()

        # 循环帧
        while True:

            # 检查是否有更多的帧在缓冲区中处理
            if self.file_stream and not self.video_stream.more():
                break

            # 从视频文件流获得帧
            frame = self.video_stream.read()

            # 调整大小
            frame = imutils.resize(frame, width=800)

            # 将其转换为灰度
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 帧数目自增
            self.frame_counter += 1

            # 在灰度帧中检测人脸
            faces = self.detector(gray_frame, 0)

            if faces and faces[0]:

                # 获取人脸68个特征点
                face = faces[0]

                # 确定人脸区域的人脸标志
                face_region = self.predictor(gray_frame, face)

                # 人脸标志(x, y)坐标转换为一个NumPy数组
                face_region = face_utils.shape_to_np(face_region)

                # 检查帧的数目有没有达到阈值
                if self.frame_counter < NUMBER_OF_FRAMES_TO_INSPECT:

                    if self.frame_counter < NUMBER_OF_FRAMES_TO_INSPECT_EYES:

                        # 通过前几帧计算眼睛的平均宽高比
                        self.calculate_eye_aspect_ratio(face_region)

                        # 通过前几帧计算嘴唇的平均宽高比
                        self.calculate_lips_aspect_ratio(face_region)

                    elif self.frame_counter < NUMBER_OF_FRAMES_TO_INSPECT_EYES + 3:

                        # 通过前几帧计算眼睛的平均宽高比阈值
                        self.blink_detector.calculate_eye_aspect_ratio_threshold(self.person.eye_aspect_ratio)

                        # 通过前几帧计算嘴唇的平均宽高比阈值
                        self.pursed_lips_detector.calculate_lips_aspect_ratio_threshold(self.person.lips_aspect_ratio)

                    else:
                        # 开始眨眼检测
                        self.blink_detector.detect(frame, face_region)

                        # 开始咧嘴微笑检测
                        self.pursed_lips_detector.detect(frame, face_region)

                    # 计算脸颊的平均颜色
                    self.calculate_average_cheek_color(frame, gray_frame, face_region)

                # 帧的数量达到了阈值
                elif self.frame_counter == NUMBER_OF_FRAMES_TO_INSPECT:

                    print("SET AVERAGE VALUES")

                    # 设置脸颊平均颜色
                    self.blushing_detector.set_average_cheek_color(self.person.average_cheek_color)

                    now = time.time()

                    # 设置平均眨眼次数
                    self.person.set_average_number_of_blinks(self.blink_detector.get_and_reset_number_of_blinks(),
                                                             now - timeBefore)

                    # 设置平均咧嘴次数
                    self.person.set_average_number_of_lip_pursing(
                        self.pursed_lips_detector.get_and_reset_number_of_lip_pursing())

                    # 测试的初始数据
                    print("average_cheek_color", self.person.average_cheek_color)
                    print("average_number_of_blinks", self.person.average_number_of_blinks)
                    print("average_number_of_lip_pursing", self.person.average_number_of_lip_pursing)

                # 开始检测声音、眨眼、咧嘴、脸颊
                else:
                    self.blink_detector.detect(frame, face_region)
                    self.pursed_lips_detector.detect(frame, face_region)
                    self.blushing_detector.detect(frame, gray_frame, face_region)

                # 打印阈值眼睛高宽比
                cv2.putText(frame, "A_EAR: {:.4f}".format(self.person.eye_aspect_ratio), (200, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 打印阈值嘴巴宽高比
                cv2.putText(frame, "A_LAR: {:.4f}".format(self.person.lips_aspect_ratio), (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 将帧展示出来
            cv2.imshow("Lie detector", frame)

            # 设置按钮
            key = cv2.waitKey(1) & 0xFF

            # 结束按钮
            if key == ord("q"):
                break

            # 第二次测试
            elif key == ord("n"):

                # 计算测试的时间
                now = time.time()
                self.seconds = now - timeBefore
                self.detect_if_lie(tag, file_name)
                self.questions_counter += 1
                timeBefore = time.time()

        # 着三句话是跳出循环执行的
        now = time.time()
        self.seconds = now - timeBefore
        self.detect_if_lie(tag, file_name)

    """计算脸颊的平均颜色"""

    def calculate_average_cheek_color(self, frame, gray_frame, face_region):
        # 左脸
        left_cheek = face_region[self.blushing_detector.left_cheek_idx]
        # 右脸
        right_cheek = face_region[self.blushing_detector.right_cheek_idx]
        # 计算脸颊平均颜色
        calculated_cheek_color = self.blushing_detector.calculate_cheeks_color(frame, gray_frame, right_cheek,
                                                                               left_cheek)
        # 赋值给对象
        self.person.calculate_average_color(calculated_cheek_color)

    """计算眼睛的高宽比"""

    def calculate_eye_aspect_ratio(self, face_region):
        # 使用blink类的计算方法计算眼睛坐标、眼睛高宽比
        EAR, left_eye, right_eye = self.blink_detector.calculate_eye_aspect_ratio(face_region)

        # 赋值给对象
        self.person.calculate_average_eye_aspect_ratio(EAR)

    """计算咧嘴微笑的高宽比"""

    def calculate_lips_aspect_ratio(self, face_region):
        # 使用lips类的计算方法计算嘴巴坐标、嘴巴高宽比
        LAR, mouth = self.pursed_lips_detector.lips_aspect_ratio(face_region, consider_smile=False)
        # 赋值给对象

        self.person.calculate_average_lips_aspect_ratio(LAR)

    def detect_if_lie(self, state='7', file_name="data/video_data_for_lie_training.csv"):
        # 获取检测结果
        R = self.blushing_detector.R
        G = self.blushing_detector.G
        B = self.blushing_detector.B
        R_change = abs(R - self.person.average_cheek_color[0])
        G_change = abs(G - self.person.average_cheek_color[1])
        B_change = abs(B - self.person.average_cheek_color[2])
        RGB_change = R_change + G_change + B_change
        number_of_blinks = self.blink_detector.get_and_reset_number_of_blinks()
        number_of_blushing_occurred = self.blushing_detector.get_number_of_blushing_occurred_and_reset()
        number_of_lip_pursing_occurred = self.pursed_lips_detector.get_and_reset_number_of_lip_pursing()

        # 平均眨眼数
        if self.seconds > 0:
            number_of_blinks_per_second = number_of_blinks / self.seconds
        else:
            number_of_blinks_per_second = number_of_blinks

        if state == '7':
            # 设置实验获取的数据，准备用来检测
            to_predict = [RGB_change, number_of_blinks_per_second,
                          abs(number_of_blinks_per_second - self.person.average_number_of_blinks),
                          number_of_blushing_occurred, number_of_lip_pursing_occurred]

            print("test data： ", to_predict)

            prediction = build_network.predict([to_predict], "model/video_lie_detect_model")[0]

            # 在控制台打印测试结果
            print("\n\n", str(self.questions_counter), ". Detected:")
            print("blinks detected: ", str(number_of_blinks))
            print("number of blinks per second: ", str(number_of_blinks_per_second))
            print("number of blushing occurred:  ", str(number_of_blushing_occurred))
            print("number of pursing occurred:  ", str(number_of_lip_pursing_occurred))
            print("prediction:  ", prediction[0])

            # 利用四舍五入，来判定0或1
            prediction_int = np.round(prediction[0])

            # 先设定预测结果为假
            result = "lie"

            # 如果预测结果为真
            if prediction_int:
                result = "truth"

            print("result:  " + result)

            # 将测试结果写入txt文件
            self.write_to_file(R, G, B, number_of_blinks, number_of_blushing_occurred, number_of_lip_pursing_occurred,
                               number_of_blinks_per_second, prediction[0], result)

            # 重置计时器
            self.seconds = 0

        elif state == '1':
            # 录入谎言数据
            self.write_to_csv_data(RGB_change,
                                   number_of_blinks_per_second,
                                   abs(number_of_blinks_per_second - self.person.average_number_of_blinks),
                                   number_of_blushing_occurred,
                                   number_of_lip_pursing_occurred,
                                   '1', file_name)

        elif state == '2':
            # 录入真话数据
            self.write_to_csv_data(RGB_change,
                                   number_of_blinks_per_second,
                                   abs(number_of_blinks_per_second - self.person.average_number_of_blinks),
                                   number_of_blushing_occurred,
                                   number_of_lip_pursing_occurred,
                                   '2', file_name)

    """将测试结果写入txt文件"""

    def write_to_file(self, R, G, B, number_of_blinks,
                      number_of_blushing_occurred, number_of_lip_pursing_occurred,
                      number_of_blinks_per_second, prediction, result):

        file = open("lie_detect_result.txt", "a")

        if self.questions_counter == 1:
            file.write("\n\n******************************************************\n")
            file.write("Person averaged")
            file.write("\n\tblinks: " + str(self.person.average_number_of_blinks))
            file.write("\n\tnumber of blinks per second: " + str(number_of_blinks_per_second))
            file.write("\n\tEAR: " + str(self.person.eye_aspect_ratio))
            file.write("\n\tLAR: " + str(self.person.lips_aspect_ratio))
            file.write("\n\tlip pursing: " + str(number_of_lip_pursing_occurred))
            file.write("\n\tcheek color: " + "{:0.0f}".format(self.person.average_cheek_color[2]) + ", "
                       + "{:0.0f}".format(self.person.average_cheek_color[1]) + ", "
                       + "{:0.0f}".format(self.person.average_cheek_color[0]))

        # 每一次测谎数据显示
        file.write("\n\n" + str(self.questions_counter) + ". Detected:")
        file.write("\n\tblinks detected: " + str(number_of_blinks))
        file.write("\n\tR: " + str(R))
        file.write("\n\tG: " + str(G))
        file.write("\n\tB: " + str(B))
        file.write("\n\tnumber of blinks per second: " + str(number_of_blinks_per_second))
        file.write("\n\tnumber of blushing occurred:  " + str(number_of_blushing_occurred))
        file.write("\n\tnumber of pursing occurred:  " + str(number_of_lip_pursing_occurred))
        file.write("\n\tPredicted:  " + str(prediction))
        file.write("\n\tResult:  " + result)

        # 关闭文件
        file.close()

    """将文件写入csv数据集"""

    def write_to_csv_data(self, rgb_change, number_of_blinks_per_second, per_second_blinks_change,
                          number_of_blushing_occurred, number_of_lip_pursing_occurred, my_index, file_name):

        if not os.path.exists(file_name):
            self.create_csv(file_name)

        # 记录结果
        answer = 0

        # 选项2的情况
        if my_index == '2':
            answer = 1

        data_frame = pd.DataFrame({
            'rgb_change': rgb_change,
            'number_of_blinks_per_second': number_of_blinks_per_second,
            'per_second_blinks_change': per_second_blinks_change,
            'number_of_blushing_occurred': number_of_blushing_occurred,
            'number_of_lip_pursing_occurred': number_of_lip_pursing_occurred,
            'answer': answer
        }, index=[0])

        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        data_frame.to_csv(file_name, index=False, header=False, sep=',', mode='a')

    """关闭摄像头"""

    def destroy(self):
        cv2.destroyAllWindows()
        self.video_stream.stop()

    @staticmethod
    def create_csv(file_name):
        with open(file_name, 'w') as f:
            csv_write = csv.writer(f)
            csv_head = ["rgb_change", "number_of_blinks_per_second", "per_second_blinks_change",
                        "number_of_blushing_occurred", "number_of_lip_pursing_occurred", "answer"]
            csv_write.writerow(csv_head)
