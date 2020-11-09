#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : simple_recorder.py
# @Time      : 2020/9/25 12:29
# @Author    : 陈嘉昕
# @Demand    : 声音的简单记录


from core_recorder import CoreRecorder
import threading
import logging
import time


class SimpleRecorder(threading.Thread):
    def __init__(self,
                 sr=2000,  # Sample Rate
                 ):
        threading.Thread.__init__(self)
        self.audio_data = []
        self.audio_lock = threading.Lock()
        self.logger = logging.getLogger(__name__ + ".SimpleWatcher")
        self.recorder = CoreRecorder(sr=sr)
        self.analyzers = []
        self.sr = sr
        self.start_time = None
        # 线程停止的标志
        self.__running = threading.Event()
        self.__running.set()

    def register(self, analyzer):
        self.analyzers.append(analyzer)
        analyzer.register_recorder(self)

    def run(self):
        self.recorder.start()
        for analyzer in self.analyzers:
            analyzer.start()
        while self.__running.isSet():
            self.start_time = self.recorder.start_time
            if self.start_time is not None:
                break
            time.sleep(.05)

        while self.__running.isSet():
            while not self.recorder.buffer.empty():
                v = self.recorder.buffer.get()
                self.audio_data.append(v)

        for analyzer in self.analyzers:
            analyzer.stop()
            analyzer.join()
        self.recorder.stop()
        self.recorder.join()

    # 退出线程
    def stop(self):
        self.logger.warning("Stop Signal Received!")
        self.__running.clear()
