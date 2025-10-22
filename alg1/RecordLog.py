#算法日志创建与读写
import os
from datetime import datetime
def checkisExist():
    now = datetime.today()
    year = str(now.year)
    month = str(now.month)
    day = str(now.day)
    # hour=str(now.hour)+'.txt'
    base_dir=os.getcwd().replace('\\','/')
    weitepath = os.path.join(base_dir, 'Log')
    yearpath = os.path.join(weitepath, year)
    monthpath = os.path.join(yearpath, month)
    daypath = os.path.join(monthpath, day)
    logpath = os.path.join(daypath, 'Info')
    ErorPath = os.path.join(daypath, 'Error')
    if not os.path.isdir(logpath):
        os.makedirs(logpath)
    if not os.path.isdir(ErorPath):
        os.makedirs(ErorPath)

    # Versionpath=os.path.join(daypath, 'VersionNo.txt')
    # file1 = open(Versionpath, "w", encoding='utf-8')
    # file1.close()
    # EEWsinglepath=os.path.join(logpath,'LOG_EEW_Single')
    # os.makedirs(EEWsinglepath)
    # singleTxtpath=os.path.join(logpath,hour)
    # file = open(singleTxtpath, "w", encoding='utf-8')
    # file.close()
    return daypath

# 算法每次重启或初始化时，记录算法版本信息
def WriteVersionNo(strinfo):
    daypath=checkisExist()
    Versionpath=os.path.join(daypath, 'VersionNo.txt')
    file1 = open(Versionpath, "a+", encoding='utf-8')
    file1.write(strinfo+'\n')
    file1.close()

#记录算法Debug输出信息，socket连接信息
def WriteInfo(strinfo):
    now = datetime.today()
    daypath = checkisExist()
    logpath = os.path.join(daypath, 'Info')
    hour = str(now.hour) + '.txt'
    Versionpath = os.path.join(logpath, hour)
    file1 = open(Versionpath, "a+", encoding='utf-8')
    file1.write(strinfo+'\n')
    file1.close()

#记录全局算法错误异常信息
def WriteError(strinfo):
    now = datetime.today()
    daypath = checkisExist()
    logpath = os.path.join(daypath, 'Error')
    hour = str(now.hour) + '.txt'
    Versionpath = os.path.join(logpath, hour)
    file1 = open(Versionpath, "a+", encoding='utf-8')
    file1.write(strinfo+'\n')
    file1.close()