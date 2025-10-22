
# from mat_Read import *
import pathlib

from class_obj import objmat
from EEW_Test_Massive import *
# from EEW_Test_Massive import counter
import os
import datetime
import time as Time

dirpath = r'D:\测试\监控单元测试数据集\日本数据2025下载——发波文件_fs200\小于等于20台'
filepath = os.listdir(dirpath)
filesize = len(filepath)
excel_path = os.getcwd()
dtime = datetime.datetime.now()
strdate = dtime.strftime("%Y%m%d")
exceldir = excel_path + "\\小于等于20台" + strdate + str(dtime.hour) + str(dtime.minute) + "test.xlsx"  # excel 输出
# print(filesize)
# obj: objmat = objmat()
for i in range(filesize):
    localtime = Time.asctime(Time.localtime(Time.time()))
    if os.path.isdir(filepath[i]):
        continue
    print(localtime)

    dir1 = dirpath + '\\' + filepath[i]
    filepath1 = os.listdir(dir1)
    filesize1 = len(filepath1)
    for j in range(filesize1):
        print('testall2, '+'i: '+str(i)+', j: '+str(j))  #
        if os.path.isdir(filepath1[j]):
            continue
        print(filepath1[j])
        filedir2 = dir1 + '\\' + filepath1[j]
        EEW_Test_Massvie(filedir2, exceldir, i)
    # os.system("pause")
# obj=mat_Read(fileaddress)
# z=obj.epiLon
# print(z)
