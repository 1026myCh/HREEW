import configparser
import os
import socket
import struct
import SimpleLogger as logger
from datetime import datetime
import numpy as np
import traceback
from EEW_Single import EEW_Single, CallPag, CallInit
import queue, threading, time

if __name__ == '__main__':
    # 创建 socket 对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '0.0.0.0'
    port = 9800
    EEW_Params = configparser.ConfigParser()
    current_path = os.path.dirname(os.path.abspath(__file__))
    EEW_Params.read(os.path.join(current_path, 'EEW_Params.ini'), encoding='UTF-8')
    if EEW_Params.has_option("EEW_Single","listen_port"):
        port = int(EEW_Params["EEW_Single"]["listen_port"])
    # 设置服务器的 IP 地址和端口号
    server_address = (host, port)
    # 绑定服务器地址和端口号
    server_socket.bind(server_address)
    # 监听客户端连接请求
    server_socket.listen(1)
    logger.product('Server start at: {0}:{1}'.format(host, port), 1, True)
    FEATURE_SIZE = 0

    eew_que = queue.Queue(10000)


    def alg_consumer():
        while True:
            try:
                times = []
                datas = []
                while not eew_que.empty() and len(datas) < 75:  # 算法暂时无法处理更多数据
                    data_time = eew_que.get(block=False)
                    times.append(data_time[0])
                    datas.append(data_time[1])
                if len(datas) > 0 and client_socket:
                    # 调用算法
                    dt11 = datetime.now().timestamp()
                    callData = np.concatenate(datas, axis=0)
                    [NewInfo, StationInfo, Sta_vars_ret1, Sta_vars_ret2, StartT, AlarmS] = EEW_Single(
                        callData, times[0])
                    if NewInfo == 0:
                        continue
                    dt22 = datetime.now().timestamp()
                    if dt22 - dt11 > 0.5:
                        logger.product(
                            "计算用时过大，数据点数:{0},用时:{1}".format(callData.shape[0],
                                                                        dt22 - dt11), 1, True)
                    # 回复包头
                    returnHead = bytes((0xfa, 0xfa, 0xf1, 0x2e, 0x00, 0x00, 0x00))
                    # 回复正文
                    statusBuffer = struct.pack("2B", NewInfo, 1)  # newinfo,isevent
                    # client_socket.sendall(statusBuffer)
                    returnBuffer = struct.pack("2d7f", Sta_vars_ret1.P_time, Sta_vars_ret1.S_time,
                                               Sta_vars_ret1.Magnitude, Sta_vars_ret1.Azimuth,
                                               Sta_vars_ret1.Distance_Pred,
                                               Sta_vars_ret1.PGA_Pred, StationInfo.PGA_Curr, Sta_vars_ret1.PGV_Curr,
                                               Sta_vars_ret1.PGD_Curr)  #
                    client_socket.sendall(returnHead + statusBuffer + returnBuffer + bytes((0xfb, 0xfb)))


            except Exception as e:
                logger.product(traceback.format_exc(), 2, True)
                try:
                    CallInit()
                except:
                    print("CallInit Error!")
            finally:
                time.sleep(0.001)


    threading.Thread(target=alg_consumer).start()


    def Bytes2Float32String(feature):
        num = int(FEATURE_SIZE / 4)
        x = np.empty(num, dtype=float)
        for i in range(num):
            data = feature[i * 4: (i * 4) + 4]
            x[i] = struct.unpack('f', data)[0]
        return x


    def count_pga(data):
        NewInfo, Sta_vars_ret1, pga = CallPag(data)
        if NewInfo == 2:
            # 回复包头
            returnHead = bytes((0xfa, 0xfa, 0xf1, 0x2e, 0x00, 0x00, 0x00))
            # 回复正文
            statusBuffer = struct.pack("2B", NewInfo, 1)  # newinfo,isevent
            # client_socket.sendall(statusBuffer)
            returnBuffer = struct.pack("2d7f", Sta_vars_ret1.P_time, Sta_vars_ret1.S_time,
                                       Sta_vars_ret1.Magnitude, Sta_vars_ret1.Azimuth,
                                       Sta_vars_ret1.Distance_Pred,
                                       Sta_vars_ret1.PGA_Pred, pga, Sta_vars_ret1.PGV_Curr,
                                       Sta_vars_ret1.PGD_Curr)  #
            client_socket.sendall(returnHead + statusBuffer + returnBuffer + bytes((0xfb, 0xfb)))


    # 持续接收客户端连接，监听端口永不关闭
    client_socket = socket
    while True:
        try:
            # 等待客户端连接
            client_socket, client_address = server_socket.accept()
            client_socket.settimeout(60)
            logger.product('Client connected:' + str(client_address), 1, True)
            # 循环接收客户端发送的数据包
            while True:
                # 接收数据包头
                header = client_socket.recv(7)
                # 判断是否接收到了数据包头
                if len(header) == 7 and header[0:3] == b'\xfa\xfa\x01':
                    # 接收协议类型
                    protocol_type = header[0:2]
                    # 接收数据长度
                    data_length = (header[6] << 24) + (header[5] << 16) + (header[4] << 8) + header[3]
                    # 接收数据正文
                    data = client_socket.recv(data_length)
                    if len(data) <= 0:
                        raise Exception("socket连接已断开，等待重连...")
                        # 正文长度
                    while len(data) < data_length:
                        packet = client_socket.recv(data_length - len(data))
                        if len(packet) <= 0:
                            break
                        data += packet
                    # 接收数据包尾
                    footer = client_socket.recv(2)
                    while len(footer) < 2:
                        packet = client_socket.recv(2 - len(footer))
                        if len(packet) <= 0:
                            break
                        footer += packet
                    # 判断是否接收到了数据包尾
                    if len(footer) == 2 and footer == b'\xfb\xfb':
                        # 数据包解析成功，进行处理
                        # 时间戳解析
                        s_time = struct.unpack('d', data[0:8])[0]
                        # 通道数据解析
                        FEATURE_SIZE = data_length - 8
                        newarr = Bytes2Float32String(data[8:])
                        # 一维数组转多维数组
                        data_now = newarr.reshape(-1, 6)
                        # ####ls
                        # with open('data918new1sock1.txt', 'a+') as file:
                        #     for i in data_now:
                        #         file.write(str(i[0])+'\t'+str(i[1])+'\t'+str(i[2])+'\t'+str(i[3])+'\t'+str(i[4])+'\t'+str(i[5])+'\t'+'\n')
                        #     # file.write(str(data_now))

                        dt1 = datetime.now().timestamp()
                        if dt1 - s_time > 1:
                            print("%s ！！！！！！收到延迟过大的数据:%.3f, currT:%.3f, startT:%.3f" % (
                                str(datetime.now()), dt1 - s_time, dt1, s_time))
                        # 为了提升阈值报警的效率
                        count_pga(data_now)
                        eew_que.put((s_time, data_now), timeout=1.)
                    else:
                        # 数据包尾错误，断开连接
                        client_socket.close()
                        raise Exception("数据包尾错误，断开连接，等待重连...")
                else:
                    # 数据包头错误，断开连接
                    client_socket.close()
                    raise Exception("数据包头错误，断开连接，等待重连...")
        except Exception as e:
            try:
                logger.product(traceback.format_exc(), 2, True)
            except:
                traceback.print_exc()
            try:
                client_socket.close()
            except:
                traceback.print_exc()
            client_socket = None
