#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName   : implementation.py
# @DateTime   : 2020/9/30 14:46
# @Author     : 陈嘉昕
# @Demand     : 这是实现测谎的方法


import build_network
import lie_detector_impl
import csv
import os
import simple_recorder
import volume_analyzer
import time
import pandas as pd
from keras import optimizers
import numpy as np

"""用于实时收集video的测谎数据和测谎"""


def detect(tag, file_name):
    # 构建lieDetector对象
    lieDetector = lie_detector_impl.LieDetector()

    # 开始测谎
    lieDetector.process(tag, file_name)

    # 关闭摄像机
    lieDetector.destroy()


"""用于收集audio的测谎数据"""


def collection(file_name="data/audio_data_for_lie_training.csv", tag=False, flag=True):
    # 声音收集
    SR = simple_recorder.SimpleRecorder()

    # 声音分析
    VA = volume_analyzer.VolumeAnalyzer(rec_time=1)

    # 音频分析收集
    audio_detector_list = []

    # 使用多线程，一边收集声音，一边分析声音
    SR.register(VA)

    # 多线程开始
    SR.start()
    print("Audio Collection ......\n")

    # 计时器
    i = 0

    # 只能录取9秒的时间
    while i < 28:
        audio_detector_list.append(VA.get_volume())
        time.sleep(0.25)
        i += 1

    # 消除无用参数
    del audio_detector_list[0:6]

    # 最终结果
    print("test data： ", audio_detector_list)

    # 收集数据
    if flag:

        # 创建数据集文件csv
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                csv_write = csv.writer(f)
                csv_head = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                            "17", "18", "19", "20", "21", "22", "answer"]
                csv_write.writerow(csv_head)

        # 记录结果
        answer = 0

        # 如果说的是真话
        if tag:
            answer = 1

        # 对应列的数据
        data_frame = pd.DataFrame({
            '1': audio_detector_list[0],
            '2': audio_detector_list[1],
            '3': audio_detector_list[2],
            '4': audio_detector_list[3],
            '5': audio_detector_list[4],
            '6': audio_detector_list[5],
            '7': audio_detector_list[6],
            '8': audio_detector_list[7],
            '9': audio_detector_list[8],
            '10': audio_detector_list[9],
            '11': audio_detector_list[10],
            '12': audio_detector_list[11],
            '13': audio_detector_list[12],
            '14': audio_detector_list[13],
            '15': audio_detector_list[14],
            '16': audio_detector_list[15],
            '17': audio_detector_list[16],
            '18': audio_detector_list[17],
            '19': audio_detector_list[18],
            '20': audio_detector_list[19],
            '21': audio_detector_list[20],
            '22': audio_detector_list[21],
            'answer': answer
        }, index=[0])

        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        data_frame.to_csv(file_name, index=False, header=False, sep=',', mode='a')

        # 关闭线程
        SR.stop()
        SR.join()

    # 实时测谎
    else:
        prediction = build_network.predict([audio_detector_list], "model/audio_lie_detect_model")[0]
        print("\nDetect Finish")
        print("Predicted:  ", prediction[0])

        # 利用四舍五入，来判定0或1
        prediction_int = np.round(prediction[0])

        # 先设定预测结果为假
        result = "lie"

        # 如果预测结果为真
        if prediction_int:
            result = "truth"

        print("Result:  " + result)

        # 关闭线程
        SR.stop()
        SR.join()


def get_accuracy(data_name, model_name):
    # 获取测试数据集
    x_test, y_test = build_network.load_data(data_name, training=False)

    # 加载训练好的模型
    model = build_network.load_model(model_name)

    # 将模型编译，设置优化器、损失函数、学习率、评判标准
    model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 开始对模型训练的准确率进行测试
    score = model.evaluate(x_test, y_test)

    # 打印测试结果
    print("testing completed for ", model_name)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("\n\n")
