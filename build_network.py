#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : build_network.py
# @Time      : 2020/9/25 12:29
# @Author    : 陈嘉昕
# @Demand    : 搭建神经网络


import csv
import pandas
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
from sklearn import model_selection
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import confusion_matrix

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

"""导入数据"""


def load_data(file_name, training=False):
    # 获取csv数据集
    data = pandas.read_csv(file_name)

    # 将标签提取出来
    target = data['answer']

    # 将训练数据集提取出来
    data = data.drop('answer', axis=1)

    # 如果需要训练
    if training:

        # 将数据集区分开来，分为训练集和测试集
        model = model_selection.StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)

        # 分成训练集和标签
        gen = model.split(data, target)

        # 设定好训练集和测试集
        train_x, train_y, test_x, test_y = [], [], [], []

        # 循环读取数据
        for train_index, test_index in gen:
            train_x = data.loc[train_index]
            train_y = target.loc[train_index]
            test_x = data.loc[test_index]
            test_y = target.loc[test_index]

        # 返回训练所需的数据集
        return train_x, train_y, test_x, test_y

    # 不需要训练返回的数据集
    return data, target


"""训练"""


def fit_audio(x_train, y_train, dim):
    # 开始搭建神经网络
    model = Sequential()

    # 第一层全连接层
    model.add(Dense(128, input_dim=dim, init="uniform", activation="tanh"))

    # 第二层全连接层
    model.add(Dense(128, activation="tanh"))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    # 打印模型信息
    model.summary()

    # 画出模型结构并保存
    plot_model(model, to_file="image/audio_lie_detect_model.png", show_shapes=True)

    # 模型编译，优化器，设置学习率，损失函数
    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 开始训练，500个迭代
    history = model.fit(x_train, y_train, epochs=1500)

    # 获取loss和acc的变化图
    get_acc_and_loss_image(history, "image/acc_and_loss_analysis_for_audio.jpg")

    # 返回训练好的模型
    return model


def fit_video(x_train, y_train, dim):
    # 开始搭建神经网络
    model = Sequential()

    # 第一层全连接层
    model.add(Dense(128, input_dim=dim, init="uniform", activation="relu"))

    # 第二层全连接层
    model.add(Dense(128, activation="relu"))

    # 第三层全连接层
    model.add(Dense(128, activation="relu"))

    # 第四层全连接层
    model.add(Dense(128, activation="relu"))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    # 打印模型信息
    model.summary()

    # 画出模型结构并保存
    plot_model(model, to_file="image/video_lie_detect_model.png", show_shapes=True)

    # 模型编译，优化器，设置学习率，损失函数
    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 开始训练，500个迭代
    history = model.fit(x_train, y_train, epochs=500)

    # 获取loss和acc的变化图
    get_acc_and_loss_image(history, "image/acc_and_loss_analysis_for_video.jpg")

    # 返回训练好的模型
    return model


def get_acc_and_loss_image(history, file_name):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.title('Accuracy and Loss')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, loss, 'blue', label='Validation loss')
    plt.legend()
    plt.savefig(file_name)
    plt.show()


"""保存模型"""


def saving_model(model, file_name):
    # 保存模型的网络结构
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # 保存模型权重
    model.save_weights(file_name + ".h5")
    print("[INFO] Saved model to disk")


"""加载模型"""


def load_model(file_name):
    # 加载模型网络结构
    json_file = open(file_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # 加载模型权重
    loaded_model.load_weights(file_name + ".h5")
    print("[INFO] Loaded model from disk")

    # 返回加载的模型
    return loaded_model


"""预测"""


def predict(data, model_name):
    # 载入模型
    model = load_model(model_name)

    # 返回测试结果
    return model.predict(np.array(data))


"""训练模型主函数"""


def my_training(data_name, model_name, tag):
    # 加载数据集
    train_x, train_y, test_x, test_y = load_data(data_name, training=True)

    # 获取数据集向量的度
    with open(data_name) as op:
        rd = csv.reader(op)
        for raw in rd:
            dim = len(raw) - 1
            break
        op.close()

    # 开始训练
    if tag == "video":
        model = fit_video(train_x, train_y, dim)
    else:
        model = fit_audio(train_x, train_y, dim)

    # 保存模型
    saving_model(model, model_name)

    # 绘制混淆矩阵
    confusion_matrix.get_confusion_matrix(data_name, model_name, "image/" + tag + "_confusion_matrix.png")

    # 测试模型
    score = model.evaluate(test_x, test_y)

    # 打印测试数据
    print(model_name, " complete the training")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
