# import threading
import time
import multiprocessing
import threading

class ReceiveThread(multiprocessing.Process):
    def __init__(self, ip, frame):
        super(ReceiveThread, self).__init__()
        self.ip = ip
        self.frame = queue
        self.num = 1
    def run(self):
        print("run")
        while True:
            self.num += 1
            if queue.full():
                self.frame.get()
            self.frame.put(self.num)
            print(self.num, self.ip)

class test(threading.Thread):
    def __init__(self):
        super(test, self).__init__()

        self.toStop = False

    def run(self):
        while True and not self.toStop:

            time.sleep(1)

    def stop(self):
        self.toStop = True

if __name__ == "__main__":
    thread = test()
    thread.start()
    time.sleep(3)
    thread.stop()
    thread.join()
    # queue = multiprocessing.Queue(5)
    #
    #
    # thread1 = ReceiveThread("1",queue)
    # thread2 = ReceiveThread("2",queue)
    # # thread2 = ReceiveThread()
    #
    # thread1.start()
    # thread2.start()
    #
    # while True:
    #     # time.sleep(1)
    #     print(queue.get())
    #
    # detector = Detect('./weights/Final_FaceBoxes.pth', device)
    # reidor = Reid('./weights/model_best.pth', device)
    #
    #
    # ## 需要一个frame, cv2直接获取的即可
    # img_raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # img = frame
    #
    # ## 检测测试帧中的行人，返回bbox
    # bbox = detector.get_bbox(img)
    #
    # ## 对当前帧中的所有进行进行重识别
    # for b in bbox:
    #     ## 获取识别到的行人的特征向量
    #     img = img_raw.crop((b[0], b[1], b[2], b[3]))
    #     if (b[3] - b[1]) / (b[2] - b[0]) < 1:
    #         continue
    #
    #     f = reidor.generate_feature(img)
    #
    #     ## 将此人向量和待重识别的向量进行比对，阈值0.45，大于这个相似度即为重识别成功，否则并非要重识别的人，放弃识别
    #     for i, gellery_f in enumerate(gellery_fs):
    #         score = compare_vector(gellery_f, f)
    #         if score > 0.6:
    #             plt.imshow(img_raw)
    #             plt.show()
    #             print("The current scene")
    #             plt.imshow(img)
    #             plt.show()
    #
    # file_name = 'ip.txt'
    # fo = open(file_name, "r")
    # ipcamera_all = fo.readlines()
    # ipcamera = []
    # ipcamera_ip = []
    # ipcamera_channel = []
    # print(ipcamera_all)
    # for i in range(len(ipcamera_all)):
    #     temp = ipcamera_all[i].replace('\n', '').split(' ')
    #     ipcamera.append(temp[0])
    #     ipcamera_ip.append(temp[1])
    #     ipcamera_channel.append(temp[2])
    #
    # print(ipcamera)
    # print(ipcamera_ip)
    # print(ipcamera_channel)