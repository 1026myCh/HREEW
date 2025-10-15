# 胡承东 编写 2024.3.22
import os
import queue, threading, time
from datetime import datetime

log_que = queue.Queue(10000)


class LogModel:
    def __init__(self, log: str, logtype: int, isoutput: bool):
        self.log = log
        self.logtype = logtype
        self.isoutput = isoutput


def consumer():
    while True:
        try:
            if log_que.empty():
                continue

            logmodel = log_que.get(block=False)
            if os.path.exists(os.path.join(os.getcwd(), 'EEW_Params.ini')):
                logpath = os.getcwd() + "/Log"
            else:
                logpath = "/usr/eew/Log"
            logTip = "[Info]"
            if logmodel.logtype == 0:
                logpath += "/Debug"
                logTip = "[Debug]"
            elif logmodel.logtype == 1:
                logpath += "/Info"
                logTip = "[Info]"
            elif logmodel.logtype == 2:
                logpath += "/Error"
                logTip = "[Error]"

            now = datetime.now()
            logpath += "/" + str(now.year) + "/" + str(now.month) + "/" + str(now.day)
            if not os.path.exists(logpath):
                os.makedirs(logpath)
            logpath += "/" + str(now.hour) + ".log"
            writelog = logTip + str(now) + " " + logmodel.log
            with open(logpath, "a", encoding="utf-8") as file:
                file.write(writelog + "\n")
            if logmodel.isoutput:
                print(writelog)
        except Exception as e:
            print("consumer出错: %s" % str(e))
        finally:
            time.sleep(0.001)


threading.Thread(target=consumer).start()
print(str(datetime.now()) + " SimpleLogger日志记录模块启动成功！")


def product(log: str, logtype: int, isoutput: bool = False) -> int:
    """
    记录各种类型的日志
    :param log: 日志内容，字符串格式
    :param logtype: 日志类型,整型  0代表Debug 1代表INFO  2代表ERROR
    :param isoutput: bool型，是否输出到界面
    :return: 0正常 1出错
    """
    try:
        log_que.put(LogModel(log, logtype, isoutput), timeout=1.)
        return 0
    except Exception as e:
        print("product出错: %s" % str(e))
    return -1
