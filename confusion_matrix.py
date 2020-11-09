#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : confusion_matrix.py
# @Time      : 2020/10/2 15:29
# @Author    : 陈嘉昕
# @Demand    : 混淆矩阵


import build_network
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, image_name, title='Confusion matrix', chen_map=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=chen_map)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(image_name)
    plt.show()


def plot_confuse(test_model, x_val, y_val, image_name):
    predictions = test_model.predict_classes(x_val)
    conf_mat = confusion_matrix(y_true=y_val, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(y_val) + 1), image_name)


def get_confusion_matrix(file_name, model_name, image_name):
    # 导入测试数据
    x_train, y_train, x_test, y_test = build_network.load_data(file_name, training=True)
    # 导入训练模型
    model = build_network.load_model(model_name)
    plot_confuse(model, x_test, y_test, image_name)



