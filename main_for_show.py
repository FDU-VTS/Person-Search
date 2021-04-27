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
import matplotlib.pyplot as plt
import logging
import uuid
import numpy as np
import sys
import signal
from mysql import *


_t = {'detect': Timer(), 'reid': Timer()}


def quit(signum, frame):
    mysql.close()
    gallery_thread.stop()
    for i in processes:
        os.kill(i.pid, signal.SIGTERM)
    sys.exit()

class Update_gallery(threading.Thread):
    def __init__(self, gallery_path, mysql):
        super(Update_gallery, self).__init__()
        self.gallery_path = gallery_path

        host, user, password, database, charset = get_mysql_info()

        self.mysql = Mysql(host, user, password, database, charset)

        self.toStop = False

        # self.mysql = mysql

    def run(self):
        while True and not self.toStop:
            # paths = glob.glob("./gallery/*.png")
            # self.mysql.conn.ping(reconnect=True)
            paths = self.mysql.get_reid_info()
            if len(paths) != len(self.gallery_path):
                mutex.acquire()
                self.gallery_path.clear()
                self.gallery_path += paths
                mutex.release()
            time.sleep(1)

    def stop(self):
        self.toStop = True


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


def predict(frame, cid_idx, time_create, gellery_fs):
    img_raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = frame

    ## 计时器，tic开始计时，toc结束计时
    _t['detect'].tic()

    ## 检测测试帧中的行人，返回bbox
    bbox = detector.get_bbox(img)

    ## 结束行人检测的计时器，开启重识别计时器
    _t['detect'].toc()



    ## 临时变量
    img_for_vis = np.asarray(img_raw)
    img_for_vis = cv2.cvtColor(img_for_vis, cv2.COLOR_BGR2RGB)


    ## 对当前帧中的所有进行进行重识别
    for b in bbox:
        ## 获取识别到的行人的特征向量
        img = img_raw.crop((b[0], b[1], b[2], b[3]))
        if (b[3] - b[1]) / (b[2] - b[0]) < 1:
            continue
        _t['reid'].tic()
        f = reidor.generate_feature(img)
        _t['reid'].toc()

        scores = []

        mutex.acquire()
        ## 将此人向量和待重识别的向量进行比对，阈值0.45，大于这个相似度即为重识别成功，否则并非要重识别的人，放弃识别
        for i, gellery_f in enumerate(gellery_fs):
            score = compare_vector(gellery_f, f)
            scores.append(score)

        if len(scores) != 0:
            score = max(scores)
            i = scores.index(score)
            # print(scores)
            if score > 0.6:
                # plt.imshow(img_raw)
                # plt.show()
                # print("The current scene")
                # plt.imshow(img)
                # plt.show()

                cv2.rectangle(img_for_vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
                cv2.putText(img_for_vis, gallery_path[i][4], (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

                # mysql.conn.ping(reconnect=True)

                if ipcamera_channel[cid_idx] != '-1':

                    mysql.insert_recognition(id=uuid.uuid4(), create_time=time_create, isdelete=gallery_path[i][2], version=gallery_path[i][3], portait_id=gallery_path[i][4],
                                         ipc_ip=None, nvr_ip=ipcamera_ip[cid_idx],passtime=datetime.now(), nvr_channel=ipcamera_channel[cid_idx], p_score=score[0].data)

                else:
                    mysql.insert_recognition(id=uuid.uuid4(), create_time=time_create, isdelete=gallery_path[i][2], version=gallery_path[i][3], portait_id=gallery_path[i][4],
                                         ipc_ip=ipcamera_ip[cid_idx], nvr_ip=None,passtime=datetime.now(), nvr_channel=None, p_score=score[0].data)

                logging.info('camera_id: {} person_id: {} score: {} detect_time: {:.4f}s reid: {:.4f}s'.format(ipcamera[cid_idx], gallery_path[i][4],
                                                                                                score[0].data, _t[
                                                                                                            'detect'].average_time,
                                                                                                        _t[
                                                                                                            'reid'].average_time))
                print('camera_id: {} person_id: {} score: {:.4f} detect_time: {:.4f}s reid: {:.4f}s'.format(ipcamera[cid_idx],
                                                                                                               gallery_path[
                                                                                                        i][
                                                                                                                   4],
                                                                                                               score[0].data,
                                                                                                               _t['detect'].average_time,
                                                                                                               _t['reid'].average_time))
        mutex.release()

    cv2.imshow("demo"+str(cid_idx), img_for_vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit(signal.SIGTERM, 0)
        exit(0)


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

    mysql = Mysql(host, user, password, database, charset)


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

    ## 初始化gallery
    # gallery_path = glob.glob("./gallery/*.png")
    # mysql.conn.ping(reconnect=True)
    gallery_path = mysql.get_reid_info()

    mutex = threading.Lock()
    gallery_thread = Update_gallery(gallery_path, get_mysql_info())
    gallery_thread.start()

    ## 获取需要重识别的行人的特征向量
    gellery_fs = []
    check_time = time.time()
    for i in gallery_path:
        # gellery_fs.append(reidor.generate_feature(Image.open(i).convert("RGB")))
        # print(i[5].size)
        gellery_fs.append(reidor.generate_feature(i[5].convert("RGB")))

    current_gallery_num = len(gallery_path)

    # for i, path in enumerate(gallery_path):
    #     plt.imshow(path[5])
    #     plt.show()
    #     print(path[0])
    #     logging.info("The No.{} gallery person".format(i + 1))

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
            if time.time() - check_time > 10:
                mutex.acquire()
                if current_gallery_num != len(gallery_path):
                    # for i, path in enumerate(gallery_path):
                    #     plt.imshow(Image.open(path))
                    #     plt.show()
                    #     print("The No.{} gallery person".format(i + 1))
                    gellery_fs = []
                    for i in gallery_path:
                        gellery_fs.append(reidor.generate_feature(i[5].convert("RGB")))
                    current_gallery_num = len(gallery_path)
                mutex.release()
                check_time = time.time()
            frame, ret, time_create = frames.get()
            predict(frame, cid_idx, time_create, gellery_fs)

