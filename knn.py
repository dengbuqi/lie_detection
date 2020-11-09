#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : knn.py
# @Time      : 2020/9/25 12:29
# @Author    : 陈嘉昕
# @Demand    : k-临近算法，和训练的模型作比较


import csv
import math
import operator
from random import shuffle
import matplotlib.pyplot as plt

# 数据集
training_set = []
# 测试集
test_set = []


def cross_validation(file_name):
    k_max = len(training_set) - 1
    k_scores = []
    for k in range(1, k_max):
        acc = 0
        for i in range(k_max):
            to_predict = training_set[i]
            training_set.remove(to_predict)
            predictions = fit([to_predict], training_set, k)
            acc += calculate_accuracy([to_predict], predictions)
            training_set.insert(i, to_predict)

        scores = acc / k_max
        print("k =", repr(k), "  accuracy = " + repr(scores))
        k_scores.append(scores)

    plt.plot(range(1, k_max), k_scores)
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.savefig(file_name)
    plt.show()
    max_index, max_number = max(enumerate(k_scores), key=operator.itemgetter(1))
    print("\n\nThe best results: k =", repr(max_index + 1), ", accuracy = " + repr(max_number) + "\n\n")


def load_dataset(filename, training=False):
    # 打开文件
    with open(filename, 'rt') as camile:
        next(csv.reader(camile))
        lines = csv.reader(camile)
        dataset = list(lines)

        shuffle(dataset)

        if training:
            split = 0.8 * len(dataset)
        else:
            split = len(dataset)

        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if len(training_set) <= split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])


def euclidean_distance(instance1, instance2, number_of_params):
    distance = 0
    for param in range(number_of_params):
        distance += pow((float(instance1[param]) - float(instance2[param])), 2)
    return math.sqrt(distance)


def get_neighbors(trainingSet, instance, k):
    distances = []
    length = len(instance) - 1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(instance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def calculate_votes(neighbors):
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def calculate_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def fit(to_predict, dataset, k):
    predictions = []
    for x in range(len(to_predict)):
        neighbors = get_neighbors(dataset, to_predict[x], k)
        result = calculate_votes(neighbors)
        predictions.append(result)

    return predictions


def predict(to_predict, data_set_path, k=12, training=False):
    if len(training_set) == 0:
        load_dataset(data_set_path, training)

    if training:
        predictions = fit(test_set, training_set, k)
        accuracy = calculate_accuracy(test_set, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
    else:
        predictions = fit(to_predict, training_set, k)

    return predictions


def run_one():
    training_set.clear()
    test_set.clear()
    load_dataset("data/video_data_for_lie_training.csv", True)
    cross_validation("image/knn_model_1.png")


def run_two():
    training_set.clear()
    test_set.clear()
    load_dataset("data/audio_data_for_lie_training.csv", True)
    cross_validation("image/knn_model_2.png")
