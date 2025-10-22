
# from mat_Read import *
import pathlib

from class_obj import objmat
from EEW_Test_Massive import *
# from eew_multi_eq_multi_sta11 import run_all_earthquakes
# from EEW_Test_Massive import counter
import os
import datetime
import time as Time

# dirpath = r'C:\Users\lihui\Desktop\20250909地震发波文件_主\铁发0909发波数据_sta7'
dirpath = r'D:\地震20210916_20250920_txt6单台（报警台站+部分当时未报警台站）'
# dirpath = r'C:\Users\lihui\Desktop\临时数据\地震20210916_20250920_txt6单台（报警台站+部分当时未报警台站）'
# dirpath = r'C:\Users\lihui\Desktop\临时数据\干扰366组'
filepath = os.listdir(dirpath)
filesize = len(filepath)
excel_path=os.getcwd()
dtime=datetime.datetime.now()
strdate = dtime.strftime("%Y%m%d")
exceldir = excel_path+"\\地震20210916_20250920_txt6单台-nature1-thresh20-0.55gal-nature1-"+strdate+str(dtime.hour)+str(dtime.minute)+"ata_txt6.xlsx" # excel 输出

# dtime = datetime.datetime.now()
# currtime = datetime.datetime.now()
# min_time = np.round(currtime.timestamp(), 3)33333333333333333333


# for i in range(0,filesize):  # filesize
for i in range(0,filesize):  # filesize
    print(i)
    print(filepath[i])
    localtime = Time.asctime(Time.localtime(Time.time()))
    if os.path.isdir(filepath[i]):
        continue
    print(localtime)
    filedir = dirpath + '\\' + filepath[i]
    EEW_Test_Massvie(filedir, exceldir, i)
    # run_all_earthquakes(filedir, exceldir,i)
    # os.system("pause")
# obj=mat_Read(fileaddress)
# z=obj.epiLon
print('test end!!!')


