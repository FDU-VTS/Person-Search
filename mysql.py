import pymysql
from datetime import datetime
import PIL.Image as Image
from io import BytesIO
import base64
# import numpy as np
import re
from configparser import ConfigParser
# import matplotlib.pyplot as plt
import logging


class Mysql():
    def __init__(self, host='172.0.16.5', user='root', password='shinowit', database='locusdb', charset='utf8'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        try:
            self.conn = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset)
            self.cursor = self.conn.cursor()
            logging.warning('Ready connect mysql')
            print('Ready connect mysql')

        # 出现异常重新连接
        except:
            logging.warning("Failed to connect mysql")
            print("Failed to connect mysql, please check mysql.ini")
            exit(0)

    def reconnect(self):
        self.conn = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database, charset=self.charset)
        self.cursor = self.conn.cursor()

    # def insert_image(self,id, time, isdelete, version, portait_id, portrait_image):
    #     sql = "INSERT INTO t_p_portrait_image(ID, CREATE_TIME, ISDELETE, VERSION, PORTRAIT_ID, ) \
    #            VALUES ('%s', '%s',  '%s', '%s',  '%s', '%s')" % \
    #           (id, time, isdelete, version, portait_id, portrait_image)
    #     try:
    #         self.conn.ping(reconnect=True)
    #         self.cursor.execute(sql)
    #         self.conn.commit()
    #     except Exception as e:
    #         logging.warning("Exception:", e)
    #         self.reconnect()
    #         # self.conn.rollback()
    #         # self.conn.ping(reconnect=True)
    #         self.cursor.execute(sql)
    #         self.conn.commit()

    def insert_recognition(self, id, create_time, isdelete, version, portait_id, ipc_ip, nvr_ip, passtime, nvr_channel, nvr_ip_channel, p_score):
        sql = "INSERT INTO t_p_portrait_recognition(ID, CREATE_TIME, ISDELETE, VERSION, PORTRAIT_ID, IPC_IP, NVR_IP, PASSTIME, NVR_CHANNEL, NVR_IP_CHANNEL, P_SOCRE) \
               VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % \
              (id, create_time, isdelete, version, portait_id, ipc_ip, nvr_ip, passtime, nvr_channel, nvr_ip_channel, p_score)

        try:
            self.conn.ping(reconnect=True)
            self.cursor.execute(sql)
            self.conn.commit()
        except Exception as e:
            logging.warning("Exception:", e)
            # self.conn.rollback()
            self.conn.ping(reconnect=True)
            self.cursor.execute(sql)
            self.conn.commit()

    def get_reid_info(self):
        sql = "SELECT * from t_p_portrait_image"
        try:
            self.conn.ping(reconnect=True)
            self.cursor.execute(sql)
            self.conn.commit()
            all = self.cursor.fetchall()
            gallery = []
            # print(len(all))
            for i in all:
                id, time, isdelete, version, portait_id, portrait_image = i
                if isdelete == 1:
                    continue
                # print(id, time, isdelete, version, portait_id, portrait_image)
                portrait_image = base64_to_pil(portrait_image)
                # print(time)
                gallery.append((id, time, isdelete, version, portait_id, portrait_image))
                # print(f.size)
            # self.conn.commit()
            # print(gallery)
            return gallery
        except Exception as e:
            logging.warning(e)
            # self.conn.rollback()




    def close(self):
        self.cursor.close()
        self.conn.close()


def pil_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def base64_to_pil(base64_str):
    base64_data = re.sub(b'^data:image/.+;base64,', b'', base64_str)
    # base64_data = base64_data.encode(encoding="utf-8")

    missing_padding = len(base64_data) % 4
    if missing_padding != 0:
        base64_data += b'=' * (4 - missing_padding)

    base64_data = base64.b64decode(base64_data)
    image_data = BytesIO(base64_data)
    return Image.open(image_data)


def get_mysql_info():
    cf = ConfigParser()
    f = open('mysql.ini')
    cf.read_file(f)
    host = cf.get("default", "host")
    user = cf.get("default", "user")
    password=cf.get("default", "password")
    database=cf.get("default", "database")
    charset=cf.get("default", "charset")
    return (host,user,password,database,charset)


# def insertReidPerson(ip, id):
#     mysql.insert_recognition(id=temp[0], create_time=temp[1], isdelete=temp[2], version=temp[3], portait_id=temp[4], ipc_ip=6, nvr_ip=7,
#                        passtime=datetime.now(), nvr_channel=nvr_channel[ip])

if __name__ == "__main__":
    mysql = Mysql()

    reid_gallery = Mysql.get_reid_info()
    gallery_imgs = list(map(lambda x: x[5], reid_gallery))


    # mysql.insert_recognition(id=15, create_time=datetime.now(), isdelete=0, version=1, portait_id=5, ipc_ip=6, nvr_ip=7,
    #                    passtime=datetime.now(), nvr_channel=9)
    # mysql.insert_recognition(id=temp[0], create_time=temp[1], isdelete=temp[2], version=temp[3], portait_id=temp[4], ipc_ip=6, nvr_ip=7,
    #                    passtime=datetime.now(), nvr_channel=9)

    # mysql.get_result_info()


    mysql.close()



