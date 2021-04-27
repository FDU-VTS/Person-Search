from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
# import numpy as np
from utils.timer import Timer
import glob
import PIL.Image as Image
import random
from interface import compare_vector, Detect, Reid
import cv2
# import multiprocessing
from multiprocessing import Process, Queue
import threading
import time
# import matplotlib.pyplot as plt
import logging
import uuid
import numpy as np
import sys
import signal
import time
from mysql import *


_t = {'detect': Timer(), 'reid': Timer()}


def quit(signum, frame):
    for i in processes:
        os.kill(i.pid, signal.SIGTERM)
    sys.exit()




class ReceiveProcess(Process):
    def __init__(self, ip, frame):
        super(ReceiveProcess, self).__init__()
        self.ip = ip
        self.frame = frame

    def run(self):
        cap = cv2.VideoCapture(self.ip)
        ret, frame = cap.read()
        if self.frame.full():
            self.frame.get()
        self.frame.put((frame, ret, datetime.now()))
        while True:
            for _ in range(25):
                ret, frame = cap.read()
                while frame is None:
                    logging.warning("Missing frame from ", self.ip)
                    cap = cv2.VideoCapture(self.ip)
                    ret, frame = cap.read()

            ret, frame = cap.read()
            if self.frame.full():
                self.frame.get()
            self.frame.put((frame, ret, datetime.now()))
            while frame is None:
                logging.warning("Missing frame from ", self.ip)
                cap = cv2.VideoCapture(self.ip)
                ret, frame = cap.read()
                if self.frame.full():
                    self.frame.get()
                self.frame.put((frame, ret, datetime.now()))


# class ReceiveThread(threading.Thread):
#     def __init__(self, ip, frame):
#         threading.Thread.__init__(self)
#         self.ip = ip
#         self.frame = frame

#     def run(self):
#         cap = cv2.VideoCapture(self.ip)
#         ret, frame = cap.read()
#         self.frame.append((frame, ret))
#         while True:
#             ret, frame = cap.read()
#             self.frame[0] = (frame, ret)
#             while frame is None:
#                 print("Missing frame from ", self.ip)
#                 cap = cv2.VideoCapture(self.ip)
#                 ret, frame = cap.read()
#                 self.frame[0] = (frame, ret)


def predict(frame):
    img_raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = frame

    ## 计时器，tic开始计时，toc结束计时
    _t['detect'].tic()

    ## 检测测试帧中的行人，返回bbox
    bbox = detector.get_bbox(img)

    ## 结束行人检测的计时器，开启重识别计时器
    _t['detect'].toc()


    ## 临时变量
    # img_for_vis = np.asarray(img_raw)
    # img_for_vis = cv2.cvtColor(img_for_vis, cv2.COLOR_BGR2RGB)


    ## 对当前帧中的所有进行进行重识别
    for b in bbox:
        ## 获取识别到的行人的特征向量
        img = img_raw.crop((b[0], b[1], b[2], b[3]))
        if (b[3] - b[1]) / (b[2] - b[0]) < 1:
            continue
        # _t['reid'].tic()
        # f = reidor.generate_feature(img)
        # _t['reid'].toc()

        img_for_vis = np.asarray(img)
        img_for_vis = cv2.cvtColor(img_for_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./raw_data/'+str(time.time())+'.jpg', img_for_vis)
        print("Detect!")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    device = torch.device("cpu" if False else "cuda")

    ## 日志文件
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename='recognition.log',
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )

    host, user, password, database, charset = get_mysql_info()


    if not os.path.exists('./raw_data/'):
        os.mkdir('./raw_data/')

    # mysql = Mysql(host, user, password, database, charset)


    # 摄像头地址
    #     ipcamera = ["rtsp://admin:admin12345@10.141.5.141/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1"]
    file_name = 'ip.txt'
    fo = open(file_name, "r")
    ipcamera_all = fo.readlines()
    ipcamera = []
    ipcamera_ip = []
    ipcamera_channel = []

    for i in range(len(ipcamera_all)):
        temp = ipcamera_all[i].replace('\n', '').split(' ')
        ipcamera.append(temp[0])
        ipcamera_ip.append(temp[1])
        ipcamera_channel.append(temp[2])

    ## 初始化检测模型和重识别模型
    detector = Detect('./weights/Final_FaceBoxes.pth', device)
    reidor = Reid('./weights/model_best.pth', device)

    ## 开启监督线程
    processes = []
    frames_list = []
    # print(ipcamera)
    for i, ip in enumerate(ipcamera):
        frames = Queue(5)
        frames_list.append(frames)
        process = ReceiveProcess(ip, frames)
        processes.append(process)

    for process in processes:
        process.start()

    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)

    while True:
        for cid_idx, frames in enumerate(frames_list):
            frame, ret, time_create = frames.get()
            predict(frame)

