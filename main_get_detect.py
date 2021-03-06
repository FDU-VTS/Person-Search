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

    ## ????????????tic???????????????toc????????????
    _t['detect'].tic()

    ## ????????????????????????????????????bbox
    bbox = detector.get_bbox(img)

    ## ?????????????????????????????????????????????????????????
    _t['detect'].toc()


    ## ????????????
    # img_for_vis = np.asarray(img_raw)
    # img_for_vis = cv2.cvtColor(img_for_vis, cv2.COLOR_BGR2RGB)


    ## ?????????????????????????????????????????????
    for b in bbox:
        ## ???????????????????????????????????????
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

    ## ????????????
    logging.basicConfig(level=logging.INFO,  # ??????????????????????????????
                        filename='recognition.log',
                        filemode='a',  ##????????????w???a???w?????????????????????????????????????????????????????????????????????
                        # a???????????????????????????????????????????????????????????????
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # ????????????
                        )

    host, user, password, database, charset = get_mysql_info()


    if not os.path.exists('./raw_data/'):
        os.mkdir('./raw_data/')

    # mysql = Mysql(host, user, password, database, charset)


    # ???????????????
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

    ## ???????????????????????????????????????
    detector = Detect('./weights/Final_FaceBoxes.pth', device)
    reidor = Reid('./weights/model_best.pth', device)

    ## ??????????????????
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

