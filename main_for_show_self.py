from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.timer import Timer
import glob
import PIL.Image as Image
import random
from interface import compare_vector, Detect, Reid
import cv2
import multiprocessing
from multiprocessing import Process, Queue
import threading
from threading import Lock
import time
import zmq
import sys
import signal
import datetime

from configparser import ConfigParser
# import matplotlib.pyplot as plt


_t = {'detect': Timer(), 'reid': Timer()}


r'''
    port: 11111 to receive message from system
    port: 11112 to send the result of recognition
'''

def quit(signum, frame):
    zmq_process.stop()
    # os.kill(zmq_process.pid, signal.SIGTERM)
    for i in processes:
        os.kill(i.pid, signal.SIGTERM)
    sys.exit()


class ZMQReID(threading.Thread):

    def __init__(self, gallery_path, processes, frames_list):
        super(ZMQReID, self).__init__()
        self.gallery_path = gallery_path
        self.processes = processes
        self.frames_list = frames_list
        self.toStop = False
        # self._lock = Lock()

    def run(self) -> None:
        # self.add_camera("/home/pp/re-id/hahaha.mp4")
        # with self._lock:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:11111")
        while True and not self.toStop:
            message = socket.recv_pyobj()
            print(message)
            if message["head"] == "add_camera":
                ip = message["ip"]
                self.add_camera(ip.replace('\t', ''))
                print("new_ip", ip.replace('\t', ''))
                socket.send_pyobj(dict(head="add_camera", result="successful"))
            elif message["head"] == "delete_camera":
                ip = message["ip"]
                self.delete_camera(ip.replace('\t', ''))
                socket.send_pyobj(dict(head="delete_camera", result="successful"))
            elif message["head"] == "add_person":
                name = message["name"]
                sex = message["sex"]
                department = message["department"]
                img = message["img"]
                print(name, sex, department)
                self.gallery_path.append((img, name, sex, department))
                socket.send_pyobj(dict(head="add_person", result="successful"))
            elif message["head"] == "delete_person":
                name = message["name"]
                sex = message["sex"]
                department = message["department"]
                print(name, sex, department)
                for i, temp in enumerate(self.gallery_path):
                    if name == temp[1] and sex == temp[2] and department == temp[3]:
                        self.gallery_path.pop(i)
                        socket.send_pyobj(dict(head="delete_person", result="successful"))
                        print("delete_person successful")
                    if i == len(self.gallery_path)-1:
                        socket.send_pyobj(dict(head="delete_person", result="not find"))
                        print("not find")
                if len(self.gallery_path) == 0:
                    socket.send_pyobj(dict(head="delete_person", result="not find"))
                    print("zero")
        socket.close()
            # context.term()

    def stop(self):
        self.toStop = True

    def add_camera(self, ip):
        frames = Queue(5)
        self.frames_list.append(frames)
        process = ReceiveProcess(ip, frames)
        self.processes.append(process)
        process.start()
        print("zmq,", len(self.frames_list), len(self.processes))

    def delete_camera(self, ip):
        for process in self.processes:
            if process.ip == ip:
                os.kill(process.pid, signal.SIGTERM)

def send_recognition(img, name, sex, department, server_ip, ori_ip):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://{0}:3002".format(server_ip))
    message = dict(head="recognition_reid", image=img, name=name, sex=sex, department=department, ip=ori_ip)
    print("already sead")
    socket.send_pyobj(message)


class Update_gallery(threading.Thread):
    def __init__(self, gallery_path):
        super(Update_gallery, self).__init__()
        self.gallery_path = gallery_path

    def run(self):
        while True:
            paths = glob.glob("./gallery/*.jpg")
            if len(paths) != len(self.gallery_path):
                self.gallery_path.clear()
                self.gallery_path += paths
            time.sleep(10)


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
        self.frame.put((frame, ret, self.ip))
        while True:
            for _ in range(25):
                ret, frame = cap.read()
                while frame is None:
                    print("Missing frame from ", self.ip)
                    cap = cv2.VideoCapture(self.ip)
                    ret, frame = cap.read()

            ret, frame = cap.read()
            if self.frame.full():
                self.frame.get()
            if frame is not None:
                self.frame.put((frame, ret, self.ip))
            while frame is None:
                print("Missing frame from ", self.ip)
                cap = cv2.VideoCapture(self.ip)
                ret, frame = cap.read()
                if self.frame.full():
                    self.frame.get()
                self.frame.put((frame, ret, self.ip))


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




def predict(frame, cid, gellery_fs):
    img_raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = frame

    ## 计时器，tic开始计时，toc结束计时
    _t['detect'].tic()

    ## 检测测试帧中的行人，返回bbox
    bbox = detector.get_bbox(img)

    ## 结束行人检测的计时器，开启重识别计时器
    _t['detect'].toc()

    ## 对当前帧中的所有进行进行重识别
    for b in bbox:
        ## 获取识别到的行人的特征向量
        img = img_raw.crop((b[0], b[1], b[2], b[3]))
        if (b[3] - b[1]) / (b[2] - b[0]) < 1:
            continue
        _t['reid'].tic()
        f = reidor.generate_feature(img)
        _t['reid'].toc()

        ## 将此人向量和待重识别的向量进行比对，阈值0.45，大于这个相似度即为重识别成功，否则并非要重识别的人，放弃识别
        for i, gellery_f in enumerate(gellery_fs):
            score = compare_vector(gellery_f, f)
            if score > 0.5:
                send_recognition(np.asarray(img_raw), gallery_path[i][1],gallery_path[i][2],gallery_path[i][3], server_ip, cid)
                # cv2.imwrite("./result/"+"ip"+"_"+gallery_path[i].split('/')[-1].replace('.jpg', '')+'_'+str(datetime.datetime.now())+'.jpg', cv2.cvtColor(np.asarray(img_raw),cv2.COLOR_RGB2BGR))
                # plt.imshow(img_raw)
                # plt.show()
                print("The current scene")
                # plt.imshow(img)
                # plt.show()
                print('camera_id: {} person_id: {} score: {} detect_time: {:.4f}s reid: {:.4f}s'.format(cid, i + 1,
                                                                                                        score.data, _t[
                                                                                                            'detect'].average_time,
                                                                                                        _t[
                                                                                            'reid'].average_time))


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    device = torch.device("cpu" if False else "cuda")
    # print(device)


    cf = ConfigParser()
    f = open('server.ini')
    cf.read_file(f)
    server_ip = cf.get("default", "server_ip")



    # 摄像头地址
    #     ipcamera = ["rtsp://admin:admin12345@10.141.5.141/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1",
    #                "rtsp://admin:admin12345@10.141.5.142/Streaming/Channels/1"]
    # file_name = 'ip.txt'
    # fo = open(file_name, "r")
    # ipcamera = fo.readlines()
    # for i in range(len(ipcamera)):
    #     ipcamera[i] = ipcamera[i].replace('\n', '')

    #     ipcamera = ["/root/reid"]

    ## 初始化检测模型和重识别模型
    detector = Detect('./weights/Final_FaceBoxes.pth', device)
    reidor = Reid('./weights/model_best.pth', device)

    ## 初始化gallery
    # gallery_path = glob.glob("./gallery/*.jpg")
    gallery_path = []

    # gallery_thread = Update_gallery(gallery_path)
    # gallery_thread.start()

    ## 获取需要重识别的行人的特征向量
    gellery_fs = []
    check_time = time.time()
    for i in gallery_path:
        gellery_fs.append(reidor.generate_feature(i[0]))

    current_gallery_num = len(gallery_path)

    for i, path in enumerate(gallery_path):
        # plt.imshow(Image.open(path))
        # plt.show()
        print("The No.{} gallery person".format(i + 1))



    ## 开启监督线程
    processes = []
    frames_list = []

    zmq_process = ZMQReID(gallery_path, processes, frames_list)
    zmq_process.start()
    # for i, ip in enumerate(ipcamera):
    #     frames = Queue(5)
    #     frames_list.append(frames)
    #     process = ReceiveProcess(ip, frames)
    #     processes.append(process)

    for process in processes:
        process.start()


    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)

    print("gallery count: ", current_gallery_num)
    print("camera count: ", len(frames_list))

    while True:
        for frames in frames_list:
            if time.time() - check_time > 1:
                # print("camera count: ", len(frames_list))
                if current_gallery_num != len(gallery_path):
                    print("gallery count: ", current_gallery_num)
                    gellery_fs = []
                    for i in gallery_path:
                        gellery_fs.append(reidor.generate_feature(Image.fromarray(i[0])))
                    current_gallery_num = len(gallery_path)
                check_time = time.time()
            if frames.empty():
                continue
            frame, ret, ip = frames.get()
            predict(frame, ip, gellery_fs)

