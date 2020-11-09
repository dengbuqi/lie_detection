# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName   : start_up.py
# @DateTime   : 2020/9/27 13:26
# @Author     : 陈嘉昕
# @Demand     : start_up.py是主函数，1是录入说谎面部数据，2是录入真相面部数据，3是录入说谎音频数据，4是录入真相音频数据，5是使用训练集训练面部微表情
#               测谎模型，6是使用训练集训练音频特征测谎模型，7是使用面部特征模型实时测谎，8是使用音频特征模型实时测谎，9是使用测试集检测面部
#               微表情测谎模型的准确性，10是使用测试集检测语音特征测谎模型的准确性，11是使用knn训练面部微表情测谎模型，12是使用knn训练语音
#               特征测谎模型，13是退出程序

import logging
import build_network
import implementation
import knn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

if __name__ == "__main__":

    while True:
        print("\n\n####################Lie Detect System####################")
        # 选项1 录入说谎面部特征
        print("1. Record Lying Facial Features")
        # 选项2 录入真话面部特征
        print("2. Record Truth Facial Features")
        # 选项3 录入说谎语音数据
        print("3. Record Lying Audio Features")
        # 选项4 录入真话语音数据
        print("4. Record Truth Audio Features")
        # 选项5 使用训练集训练面部微表情测谎模型
        print("5. Use The Training Set To Train Model One(Facial)")
        # 选项6 使用训练集训练音频特征测谎模型
        print("6. Use The Training Set To Train Model Two(Audio)")
        # 选项7 使用面部特征模型实时测谎
        print("7. Real-Time Lie Detection Using Model One(Facial)")
        # 选项8 使用音频特征模型实时测谎
        print("8. Real-Time Lie Detection Using Model Two(Audio)")
        # 选项9 使用测试集检测面部微表情测谎模型的准确性
        print("9. Use The Test Set To Evaluate Model One(Facial)")
        # 选项10 使用测试集检测语音特征测谎模型的准确性
        print("10. Use The Test Set To Evaluate Model Two(Audio)")
        # 选项11 使用knn训练面部微表情测谎模型
        print("11. Use Knn To Train Model One(Facial)")
        # 选项12 使用knn训练语音特征测谎模型
        print("12. Use Knn To Train Model Two(Audio)")
        # 选项13 退出程序
        print("13. Exit The Program\n")

        # 输入选择
        x = input("Please enter value: ")
        print("")
        if x == '1':
            implementation.detect(tag='1', file_name="data/video_data_for_lie_training.csv")

        elif x == '2':
            implementation.detect(tag='2', file_name="data/video_data_for_lie_training.csv")

        elif x == '3':
            implementation.collection(file_name="data/audio_data_for_lie_training.csv", tag=False, flag=True)

        elif x == '4':
            implementation.collection(file_name="data/audio_data_for_lie_training.csv", tag=True, flag=True)

        elif x == '5':
            build_network.my_training(data_name="data/video_data_for_lie_training.csv",
                                      model_name="model/video_lie_detect_model",
                                      tag="video")
        elif x == '6':
            build_network.my_training(data_name="data/audio_data_for_lie_training.csv",
                                      model_name="model/audio_lie_detect_model",
                                      tag="audio")
        elif x == '7':
            implementation.detect(tag='7', file_name="data/video_data_for_lie_training.csv")

        elif x == '8':
            implementation.collection(file_name="data/audio_data_for_lie_training.csv", tag=False, flag=False)

        elif x == '9':
            implementation.get_accuracy(data_name="data/video_data_for_lie_training.csv",
                                        model_name="model/video_lie_detect_model")
        elif x == '10':
            implementation.get_accuracy(data_name="data/audio_data_for_lie_training.csv",
                                        model_name="model/audio_lie_detect_model")
        elif x == '11':
            knn.run_one()

        elif x == '12':
            knn.run_two()

        else:
            exit(0)
