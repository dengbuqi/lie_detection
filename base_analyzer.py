#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : base_analyzer.py
# @Time      : 2020/9/25 12:29
# @Author    : 陈嘉昕
# @Demand    : 声音基本分析


import threading
import logging


class BaseAnalyzer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.audio_data = None
        self.sr = None
        self.running = threading.Event()
        self.running.set()
        self.logger = logging.getLogger(str(self.__class__))

    def stop(self):
        self.logger.debug("Stop signal received")
        self.running.clear()

    def register_recorder(self, recorder):
        self.audio_data = recorder.audio_data
        self.sr = recorder.sr
        self.recorder = recorder

    def is_running(self):
        return self.running.isSet()
