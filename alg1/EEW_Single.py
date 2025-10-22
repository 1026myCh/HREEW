import configparser
import os
import warnings

import random
import joblib
import numpy as np

import SimpleLogger as logger
from Azimuth3 import Azimuth3
from EEW_Triger2 import EEW_Triger2
from EEW_Triger2 import MyPickerS
from Judges import Judges
from LMS import *
from Nature_Single import Nature_Single
# from Nature_Single1 import Nature_Single1
from StaticVar import StaticVar as persistent  # 静态类
from StaticVar import Static_EEW_Params as EEWParams  # 静态类
from class_StaV import Sta_V
from class_StaV import StationInfosStruct
from class_obj import AlarmS
from filtermean1 import filtermean1
from filtermean2 import filtermean2
from function1 import *
from srs_integration import srs_integration, srs_integrationV

warnings.filterwarnings("ignore")

# def EEW_Single(Data_now, StartT, Lon, Lat, Stv_arg1, Stv_arg2):

Sta_vars1 = Sta_V()
Sta_vars2 = Sta_V()
Alarm1 = AlarmS()
Alarm2 = AlarmS()
scaler = None
magmodel = None
scaler_distance = None
model_distance = None

# LMS
L0 = LMS()
L1 = LMS()
L2 = LMS()
L3 = LMS()
L4 = LMS()
L5 = LMS()


# L0 = LMS()
# L1 = LMS()
# L2 = LMS()
# L3 = LMS()
# L4 = LMS()
# L5 = LMS()



def EEW_Single(Data_now, StartT):
    # 局部变量
    # try:
    global Sta_vars1
    global Sta_vars2
    global Alarm1
    global Alarm2
    global scaler
    global magmodel
    global scaler_distance
    global model_distance

    Alarm = AlarmS()  # 输出报警对象
    V_in = str(20231205)  # Version No.
    StationInfo = StationInfosStruct()
    AZ1 = Sta_vars1.Azimuth
    AZ2 = Sta_vars2.Azimuth
    S_time1 = Sta_vars1.S_time
    S_time2 = Sta_vars2.S_time
    NewInfo = 0


    # return NewInfo,StationInfo,Sta_vars1,Sta_vars2,StartT,AlarmS
    if persistent.FirstStart == -1:  # 第一次启动初始化(包含参数初始化)
        persistent.FirstStart = 1
        persistent.First3s = 1
        persistent.First3s2 = 1
        persistent.first_fig = 1
        persistent.First_filter = 1
        persistent.flag_nature = 0
        persistent.Flag = None
        persistent.tempret1 = None
        persistent.tempret2 = None
        persistent.retout = None
        persistent.dist_time = 0

        if EEWParams.Debug == -1:
            current_path = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(os.path.join(current_path, 'mag_scaler.pkl')):
                scaler = joblib.load(os.path.join(current_path, 'mag_scaler.pkl'))
            else:
                scaler = joblib.load(os.path.join("/usr/eew/", 'mag_scaler.pkl'))

            if os.path.exists(os.path.join(current_path, 'mag.pkl')):
                magmodel = joblib.load(os.path.join(current_path, 'mag.pkl'))
            else:
                magmodel = joblib.load(os.path.join("/usr/eew/", 'mag.pkl'))

            if os.path.exists(os.path.join(current_path, 'distance_scaler2.pkl')):
                scaler_distance = joblib.load(os.path.join(current_path, 'distance_scaler2.pkl'))
            else:
                scaler_distance = joblib.load(os.path.join("/usr/eew/", 'distance_scaler2.pkl'))
            if os.path.exists(os.path.join(current_path, 'svm_regression_model2.pkl')):
                model_distance = joblib.load(os.path.join(current_path, 'svm_regression_model2.pkl'))
            else:
                model_distance = joblib.load(os.path.join("/usr/eew/", 'svm_regression_model2.pkl'))

            EEW_Params = configparser.ConfigParser()
            if os.path.exists(os.path.join(current_path, 'EEW_Params.ini')):
                EEW_Params.read(os.path.join(current_path, 'EEW_Params.ini'), encoding='UTF-8')
            else:
                EEW_Params.read(os.path.join("/usr/eew/", 'EEW_Params.ini'), encoding='UTF-8')
                if len(EEW_Params.sections()) == 0:
                    logger.product("读取EEW_Params.ini失败！", 2, True)
                    return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

            EEWParams.Var_Copy(EEWParams, EEW_Params)  # EEWParams 静态变量
            persistent.flag_nature = int(EEW_Params["EEW_Single"]["Nature"])
            if persistent.flag_nature == 1:
                EEWParams.MinThresh = float(EEW_Params["MyPickerAIC"]["MinThresh_Nature"])
                EEWParams.MinCorrEW_Nature = float(EEW_Params["Judges"]["MinCorrEW_Nature"])
                EEWParams.MinCorrNS_Nature = float(EEW_Params["Judges"]["MinCorrNS_Nature"])
                EEWParams.MinCorrUD_Nature = float(EEW_Params["Judges"]["MinCorrUD_Nature"])
                EEWParams.flagarea = int(EEW_Params["EEW_Single"]["FlagArea"])
                EEWParams.in_type = int(EEW_Params["EEW_Single"]["in_model"])
                EEWParams.Gain = float(EEW_Params["EEW_Single"]["Gain"])
            strP2 = ('Debug:' + str(EEWParams.Debug) + ',ThreshGals:' + str(EEWParams.ThreshGals) +
                     ',Minthresh:' + str(EEWParams.MinThresh) + ',thresh:' + str(EEWParams.thresh) +
                     ',IFCkCorr:' + str(EEWParams.IfCkCorr) + ',MinCorrEW:' + str(EEWParams.MinCorrEW) +
                     ',MinCorrN:' + str(EEWParams.MinCorrNS) + ',MinCorrUD:' + str(EEWParams.MinCorrUD) +
                     ',FirstStart:' + str(persistent.FirstStart))
            logger.product(strP2, 1, True)

    Debug = int(EEWParams.Debug)
    Debug_benchmark = [0, 1, 2]
    if (Debug in Debug_benchmark) == False:
        logger.product('Debug Parameters error，value must 1,2,3 line106', 1, True)
        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    if isinstance(Data_now[0][0], float) == False or isinstance(Data_now[0][1], float) == False or isinstance(
            Data_now[0][2], float) == False:
        logger.product('数据格式不是float数组', 1, True)
        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
    ind1 = Data_now.shape[1]
    if Data_now.shape[0] % 10 != 0 or Data_now.shape[0] == 0 or (ind1 != 3 and ind1 != 6):
        logger.product('传入数据维度不正确', 1, True)
        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
    ThreshGals = EEWParams.ThreshGals[1:-1].split(",")  # 将配置文件数组（str）转为list，有双括号
    EEW_Time_After_S = int(EEWParams.EEW_Time_After_S)

    if persistent.flag_nature==1:
        Buffer_seconds = float(EEWParams.Buffer_seconds)*3
    else:
        Buffer_seconds = float(EEWParams.Buffer_seconds)
    Pspeed = float(EEWParams.Pspeed)
    Sspeed = float(EEWParams.Sspeed)
    MaxEEW_times = float(EEWParams.MaxEEW_times)
    Sprate = int(EEWParams.Sprate)
    NewInfo = 0
    Buffer_len = Buffer_seconds * Sprate
    Data_now = np.transpose(Data_now)  # 6行20列，行操作优于列操作
    tempdata1 = Sta_vars1.BaseLine
    tempdata2 = Sta_vars2.BaseLine
    sum_Value = np.shape(Data_now)[0]  # Data——now 共有多少元素，用于判断传入数据维度
    sum_Value = int(sum_Value)
    Data_now = np.round(Data_now, 4)

    if np.size(tempdata1) == 0:
        Sta_vars1.BaseLine = Data_now[0:3]
        if sum_Value == 6:
            Sta_vars2.BaseLine = Data_now[3:]

    if np.shape(tempdata1)[1] >= Buffer_len:  # size 函数是所有元素的长度合 填满缓存
        tempdata1 = np.hstack((tempdata1, Data_now[0:3]))
        tempdata11 = tempdata1[:, (-int(Buffer_len)):]
        Sta_vars1.BaseLine = np.copy(tempdata11)
        if sum_Value == 6:
            tempdata2 = np.hstack((tempdata2, Data_now[3:]))
            tempdata22 = tempdata2[:, (-int(Buffer_len)):]
            Sta_vars2.BaseLine = np.copy(tempdata22)
    else:  # 未填满
        tempdata1 = np.hstack((Sta_vars1.BaseLine, Data_now[0:3]))
        if sum_Value == 6:
            tempdata2 = np.hstack((Sta_vars2.BaseLine, Data_now[3:]))
        Sta_vars1.BaseLine = np.copy(tempdata1)
        Sta_vars2.BaseLine = np.copy(tempdata2)

    # LMS 计算XA
    orData_now = np.copy(Data_now)
    Data_now = Data_now / 1000

    # L0 = LMS()
    # L1 = LMS()
    # L2 = LMS()
    # L3 = LMS()
    # L4 = LMS()
    # L5 = LMS()

    num1 = np.shape(Data_now)
    num1 = int(num1[1])
    if sum_Value == 6:
        tempdata = (np.zeros([6, num1], dtype=float))
    else:
        tempdata = (np.zeros([3, num1], dtype=float))

    for k in range(num1):
        tempdata[0][k] = L0.LMS_Filter(Data_now[0, k])
        tempdata[1][k] = L1.LMS_Filter(Data_now[1, k])
        tempdata[2][k] = L2.LMS_Filter(Data_now[2, k])
        if sum_Value == 6:
            tempdata[3][k] = L3.LMS_Filter(Data_now[3, k])
            tempdata[4][k] = L4.LMS_Filter(Data_now[4, k])
            tempdata[5][k] = L5.LMS_Filter(Data_now[5, k])

    Data_now = np.copy(tempdata * 1000)
    # LMS计算结束,
    step1 = 10  # int(num1/2)   # 10
    if int(Sta_vars1.Buffer.shape[1]) != 0:
        tempdataf = np.vstack((Sta_vars1.Buffer, Sta_vars2.Buffer))
        tempdataf1 = np.hstack((tempdataf[:, -step1:], Data_now))

        if persistent.flag_nature == 1:
            tempdataf11 = filtermean1(tempdataf1)
        else:
            tempdataf11 = filtermean2(tempdataf1)


        # tempdataf11 = filtermean2(tempdataf1)
        # tempdataf11 = filtermean1(tempdataf1)

        Sta_vars1.filter_Buffer = np.hstack((Sta_vars1.filter_Buffer, tempdataf11[2][-num1:]))
        Sta_vars2.filter_Buffer = np.hstack((Sta_vars2.filter_Buffer, tempdataf11[5][-num1:]))

        if len(Sta_vars1.filter_Buffer) > Buffer_len:
            Sta_vars1.filter_Buffer = Sta_vars1.filter_Buffer[-int(Buffer_len):]
            Sta_vars2.filter_Buffer = Sta_vars2.filter_Buffer[-int(Buffer_len):]
    else:
        Sta_vars1.filter_Buffer = Data_now[2]
        if sum_Value == 6:
            Sta_vars2.filter_Buffer = Data_now[5]
    # 中值滤波
    try:
        Sta_vars1.StartT = StartT
        Sta_vars2.StartT = StartT
        Sta_vars1.Package = Data_now[0:3, :]
        Sta_vars1.Buffer = np.hstack((Sta_vars1.Buffer, Data_now[0:3, :]))
        if sum_Value == 6:
            Sta_vars2.Package = Data_now[3:, :]
            Sta_vars2.Buffer = np.hstack((Sta_vars2.Buffer, Data_now[3:, :]))
    except:
        logger.product('数据文件错误single line 222', 1, True)
        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    ##########################################增加try

    if int(Sta_vars1.Buffer.shape[1]) > Buffer_len:
        Sta_vars1.Buffer = Sta_vars1.Buffer[:, -int(Buffer_len):]
        if sum_Value == 6:
            Sta_vars2.Buffer = Sta_vars2.Buffer[:, -int(Buffer_len):]
    else:
        Sta_vars1.LastEEWTime = -10 ** (-11)
        Sta_vars2.LastEEWTime = -10 ** (-11)
        Sta_vars1.LastAlarmTime = -10 ** (-11)
        Sta_vars2.LastAlarmTime = -10 ** (-11)
        # Sta_vars1 = Sta_vars1
        # Sta_vars2 = Sta_vars2
        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
    # Sta_vars1 = Sta_vars1
    # Sta_vars2 = Sta_vars2

    if persistent.FirstStart == 1:
        str1 = '算法版本' + str(V_in)
        logger.product(str1, 1, True)
    #####################################################
    if persistent.flag_nature == 1:  # 天然模式入口分支


        nature_mode = int(getattr(EEWParams, "NatureMode", "0"))
        continue_on_non_eqk=True if nature_mode==1 else False

        [NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm] = Nature_Single(Data_now, StartT,
                                                                                         MaxEEW_times, StationInfo,
                                                                                         NewInfo, Debug, Sprate,
                                                                                         ThreshGals, S_time2,
                                                                                         Buffer_len,
                                                                                         EEW_Time_After_S, Pspeed,
                                                                                         Sspeed, Alarm, Sta_vars1,
                                                                                         Sta_vars2, Alarm1,
                                                                                         Alarm2, EEWParams.flagarea, EEWParams.in_type,
                                                                                         EEWParams.Gain,continue_on_non_eqk)

        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
    # 以下为模拟模式
    [Sta_vars1, Sta_vars2, disAlarm1, Alarm] = EEW_Triger2(Sta_vars1, Sta_vars2, Alarm)

    disAlarm2 = disAlarm1
    if Sta_vars1.End_time > 1 or persistent.FirstStart == 1:
        Sta_vars1.ini2(MaxEEW_times, Alarm)  # 初始化

    if Sta_vars2.End_time > 1 or persistent.FirstStart == 1 or Sta_vars1.End_time > 1:
        endtime_max = np.max([Sta_vars1.End_time, Sta_vars2.End_time])
        Sta_vars1.ini2(MaxEEW_times, Alarm)  # 初始化
        Sta_vars2.ini2(MaxEEW_times, Alarm)
        Sta_vars1.End_time = endtime_max
        Sta_vars2.End_time = endtime_max
    if sum_Value < 6:
        if disAlarm1 == 1:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            NewInfo = 4
            Sta_vars1.ini2(MaxEEW_times, Alarm)
            Sta_vars2.ini2(MaxEEW_times, Alarm)
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
        elif disAlarm2 == 2:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            Sta_vars1.ini2(MaxEEW_times, Alarm)
            Sta_vars2.ini2(MaxEEW_times, Alarm)
            NewInfo = 100
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
    else:
        if disAlarm1 == 1 or disAlarm2 == 1:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            NewInfo = 4
            Sta_vars1.ini2(MaxEEW_times, Alarm)
            Sta_vars2.ini2(MaxEEW_times, Alarm)
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
        elif disAlarm1 == 2 or disAlarm2 == 2:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            NewInfo = 100
            Sta_vars1.ini2(MaxEEW_times, Alarm)
            Sta_vars2.ini2(MaxEEW_times, Alarm)
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    # LastAlarmTime = [Sta_vars1.LastAlarmTime, Sta_vars2.LastAlarmTime]  # 后续还需改进,前后不一致！！！！！
    if sum_Value == 6:
        Distance = Sta_vars1.Distance_Pred
        AZ1 = Sta_vars1.Azimuth
        S_time1 = Sta_vars1.S_time
        AZ1 = Sta_vars2.Azimuth
        S_time1 = Sta_vars2.S_time
    else:
        Distance = Sta_vars1.Distance_Pred
        AZ1 = Sta_vars1.Azimuth
        S_time1 = Sta_vars1.S_time

    if persistent.FirstStart == 1:
        E1 = np.random.standard_normal(size=600) / 100
        N1 = np.random.standard_normal(size=600) / 100
        Z1 = np.random.standard_normal(size=600) / 100
        E2 = np.random.standard_normal(size=600) / 100
        N2 = np.random.standard_normal(size=600) / 100
        Z2 = np.random.standard_normal(size=600) / 100
        Traces_evt1 = np.array([E1, N1, Z1])
        Traces_evt2 = np.array([E2, N2, Z2])
        Traces_evt = np.vstack((Traces_evt1, Traces_evt2))
    else:
        Traces_evt1 = np.copy(Sta_vars1.Traces_evt)
        Traces_evt2 = np.copy(Sta_vars2.Traces_evt)
        row_means1 = np.mean(Traces_evt1, axis=1, keepdims=True)
        row_means2 = np.mean(Traces_evt2, axis=1, keepdims=True)
        Traces_evt1 = Traces_evt1 - row_means1
        Traces_evt2 = Traces_evt2 - row_means2
        Traces_evt = np.vstack((Traces_evt1, Traces_evt2))

        # Traces_evt = trace_reset(Sta_vars1, Sta_vars2, StartT, Traces_evt1, Traces_evt2, Data_now, Buffer_len)
    DurCurr = np.shape(Traces_evt1)[1] / Sprate

    if DurCurr < 2 and persistent.FirstStart != 1:
        NewInfo = 0
        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    if Sta_vars1.Is_EQK < 1 or Sta_vars2.Is_EQK < 1:
        [result, corrout] = Judges(Traces_evt, 'multi_parameters')
        temp = np.shape(Traces_evt)[1]
        Sta_vars1.corrENZ=corrout
        Sta_vars2.corrENZ=corrout
        if result == 1:
            if np.abs(Sta_vars1.P_time - (Sta_vars1.StartT - np.size(Traces_evt[0], 0) / 200)) > 0.3:
                Sta_vars1.P_time = Sta_vars1.StartT - np.size(Traces_evt[0], 0) / 200
                Sta_vars2.P_time = Sta_vars2.StartT - np.size(Traces_evt[0], 0) / 200

            print("-----地震 地震 地震------")
            str1 = '地震 地震 地震,Traces_evt length:' + str(temp)
            logger.product(str1, 1, True)
            Sta_vars1.Is_EQK = 1
            Sta_vars2.Is_EQK = 1
            [PGA1, PGA2, PGApos1, PGApos2] = cmp_vector(Traces_evt)
            PGA1 = float(PGA1)
            PGA2 = float(PGA2)
            pos12 = (PGApos1 + PGApos2) / 2
            lenbase = len(Sta_vars2.BaseLine[2])
            PGAt = StartT + (Data_now.shape[0] + pos12 - lenbase) / Sprate  # 秒，浮点数
            Alarm1 = EEW_Alarm(PGA1, PGA2, PGAt, ThreshGals, Alarm1)
            Alarm2 = Alarm1
            Alarm = Alarm1
            maxPGA = max(PGA1, PGA2)
            if PGA1 > 0 and PGA2 > 0 and np.abs(PGA1 - PGA2) / maxPGA > 0.3:
                 # str1 = '两传感器PGA差异较大!'
                # logger.product(str1, 1, True)
                Sta_vars1.PGA_Curr = PGA1
                Sta_vars2.PGA_Curr = PGA2
                if PGA1 >= 40 and PGA2 >= 40:
                    Alarm = EEW_Alarm(PGA1, PGA2, PGAt, ThreshGals, Alarm1)
                    if np.mean([PGA1, PGA2]) >= float(ThreshGals[0]):
                        Alarm.AlarmLevel = 1
                    if np.mean([PGA1, PGA2]) >= float(ThreshGals[1]):
                        Alarm.AlarmLevel = 2
                    if np.mean([PGA1, PGA2]) >= float(ThreshGals[2]):
                        Alarm.AlarmLevel = 3
                    if Alarm.AlarmLevel >= 1:
                        Sta_vars1.LastAlarmTime = StartT
                        Sta_vars2.LastAlarmTime = StartT
                        if NewInfo == 1:
                            NewInfo = 3
                        elif NewInfo == 0:
                            NewInfo = 2
                    Sta_vars1.PGAEEW_times = Sta_vars1.PGAEEW_times + 1
                    str11 = ('阈值报警！' + 'AlarmLevel:' + str(Alarm.AlarmLevel) + ',PGA:' + str(
                        Alarm.PGA) + ',traceLen:' + str(len(Traces_evt[0]))+ ',delT:' + str(Alarm.delT) + ',recordtime:' + str(Alarm.recordtime) +
                             ',StartT:' + str(StartT) + ',Alarmtime:' + str(Alarm.Alarmtime) + ',NewInfo:' + str(
                                NewInfo) + '，报次：' + str(Sta_vars1.PGAEEW_times))
                    logger.product(str11, 1, True)
            elif PGA1 > 0 and PGA2 > 0 and np.abs(PGA1 - PGA2) / maxPGA < 0.3:
                if (PGA1 + PGA2) / 2 < persistent.PGAold:
                    # print("两传感器差异较小")
                    temp_CurrPGA = (Sta_vars1.PGA_Curr + Sta_vars2.PGA_Curr) / 2
                    if np.round((PGA1 + PGA2) / 2, 1) > np.round(temp_CurrPGA,
                                                                 1) and Alarm1.AlarmLevel >= 1 and Alarm2.AlarmLevel >= 1:
                        if NewInfo == 1:
                            NewInfo = 3
                        elif NewInfo == 0:
                            NewInfo = 2
                        if Alarm1.AlarmLevel > Alarm2.AlarmLevel:
                            Alarm.AlarmLevel = Alarm1.AlarmLevel
                        else:
                            Alarm.AlarmLevel = Alarm2.AlarmLevel
                        Sta_vars1.PGAEEW_times = Sta_vars1.PGAEEW_times + 1
                        if Alarm.AlarmLevel>=1:
                            Sta_vars1.AlarmFlag = 1
                            Sta_vars2.AlarmFlag = 1
                        str11 = ('阈值报警！' + 'AlarmLevel:' + str(Alarm.AlarmLevel) + ',PGA:' + str(
                            Alarm.PGA) + ',delT:' + str(Alarm.delT) + ',recordtime:' + str(Alarm.recordtime) +
                                 ',StartT:' + str(StartT) + ',Alarmtime:' + str(Alarm.Alarmtime) + ',NewInfo:' + str(
                                    NewInfo) + '，报次：' + str(Sta_vars1.PGAEEW_times))
                        logger.product(str11, 1, True)

            Sta_vars1.AlarmLevel = Alarm1.AlarmLevel
            Sta_vars2.AlarmLevel = Alarm2.AlarmLevel
        else:
            Sta_vars1.Is_EQK = 0
            Sta_vars2.Is_EQK = 0
            if persistent.FirstStart == 0:
                # if Debug == 1:
                #     str1 = '非地震、非地震、非地震'
                #     logger.product(str1, 1, True)
                return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    # 条件结合新修改的地震状态标志修改！20251011
    data6 = np.vstack((Sta_vars1.Buffer, Sta_vars2.Buffer))
    SNout = snr(Sta_vars1.Package,data6,Sta_vars1.StartT,Sta_vars1.P_time,2)
    # print('err')
    Sta_vars1.SNout = SNout
    Sta_vars2.SNout = SNout
    # # 相似性来自judge,挪到judge下
    # Sta_vars1.corrENZ = corrout
    # Sta_vars2.corrENZ = corrout
    # 分段能量
    Ef_UDj = ef_udcmp(Traces_evt,Sprate)
    Sta_vars1.Ef_UD = Ef_UDj
    Sta_vars2.Ef_UD = Ef_UDj

    P11 = Sta_vars1.BaseLine[0] - np.mean(Sta_vars1.BaseLine[0])
    P12 = Sta_vars1.BaseLine[1] - np.mean(Sta_vars1.BaseLine[1])
    P13 = Sta_vars1.BaseLine[2] - np.mean(Sta_vars1.BaseLine[2])
    P21 = Sta_vars2.BaseLine[0] - np.mean(Sta_vars2.BaseLine[0])
    P22 = Sta_vars2.BaseLine[1] - np.mean(Sta_vars2.BaseLine[1])
    P23 = Sta_vars2.BaseLine[2] - np.mean(Sta_vars2.BaseLine[2])
    # P11=Sta_vars1.Buffer[0]
    # P12 = Sta_vars1.Buffer[1]
    # P13 = Sta_vars1.Buffer[2]
    # P21 = Sta_vars2.Buffer[0]
    # P22 = Sta_vars2.Buffer[1]
    # P23 = Sta_vars2.Buffer[2]
    P = np.array([P11, P12, P13, P21, P22, P23])
    # [PGA1, PGA2] = cmp_vector(P)
    [PGA1, PGA2, PGApos1, PGApos2] = cmp_vector(P)
    PGA1 = np.round(PGA1, 1)
    PGA2 = np.round(PGA2, 1)
    MM = max(PGA1, PGA2)
    pos12 = (PGApos1 + PGApos2) / 2
    lenbase = len(Sta_vars2.BaseLine[2])
    PGAt = StartT + (Data_now.shape[0] + pos12 - lenbase) / Sprate  # 秒，浮点数
    temp_CurrPGA = (Sta_vars1.PGA_Curr + Sta_vars2.PGA_Curr) / 2
    if PGA1 > 0 and PGA2 > 0 and np.abs(PGA1 - PGA2) / MM > 0.3:
        # str1 = '两传感器差异较大'
        # logger.product(str1, 1, True)
        Sta_vars1.PGA_Curr = PGA1
        Sta_vars2.PGA_Curr = PGA2
        if PGA1 >= 40 and PGA2 >= 40 and np.round((PGA1 + PGA2) / 2, 1) > np.round(temp_CurrPGA, 1):
            Alarm = EEW_Alarm(PGA1, PGA2, PGAt, ThreshGals, Alarm1)
            if np.mean([PGA1, PGA2]) >= float(ThreshGals[0]):
                Alarm.AlarmLevel = 1
            if np.mean([PGA1, PGA2]) >= float(ThreshGals[1]):
                Alarm.AlarmLevel = 2
            if np.mean([PGA1, PGA2]) >= float(ThreshGals[2]):
                Alarm.AlarmLevel = 3
            if Alarm.AlarmLevel >= 1:
                Sta_vars1.LastAlarmTime = StartT
                Sta_vars2.LastAlarmTime = StartT
                if NewInfo == 1:
                    NewInfo = 3
                elif NewInfo == 0:
                    NewInfo = 2
            Sta_vars1.PGAEEW_times = Sta_vars1.PGAEEW_times + 1
            str11 = ('阈值报警！' + 'AlarmLevel:' + str(Alarm.AlarmLevel) + ',PGA:' + str(
                Alarm.PGA) + ',delT:' + str(Alarm.delT) + ',recordtime:' + str(Alarm.recordtime) +
                     ',StartT:' + str(StartT) + ',Alarmtime:' + str(Alarm.Alarmtime) + ',NewInfo:' + str(
                        NewInfo) + '，报次：' + str(Sta_vars1.PGAEEW_times))
            logger.product(str11, 1, True)
            Sta_vars1.PGA_Curr = PGA1
            Sta_vars2.PGA_Curr = PGA2
    elif PGA1 > 0 and PGA2 > 0 and np.abs(PGA1 - PGA2) / MM < 0.3:
        if np.round((PGA1 + PGA2) / 2, 1) > np.round(temp_CurrPGA, 1):
            Alarm1 = EEW_Alarm(PGA1, PGA2, PGAt, ThreshGals, Alarm1)
            # Alarm2 = EEW_Alarm(PGA2, Sta_vars2.STA_Name, ThreshGals, Alarm2)
            Alarm2 = Alarm1
            Alarm = Alarm1
            # PGA = np.round((PGA1 + PGA2) / 2, 2)
            # Alarm.STA_Name = Sta_vars1.STA_Name
            # Alarm.PGA = PGA
            if Alarm1.AlarmLevel > 0 and Alarm2.AlarmLevel > 0 and Sta_vars1.Is_EQK>0 and Sta_vars2.Is_EQK>0:
                Sta_vars1.LastAlarmTime = StartT
                Sta_vars2.LastAlarmTime = StartT
                if NewInfo == 1:
                    NewInfo = 3
                elif NewInfo == 0:
                    NewInfo = 2

                if Alarm1.AlarmLevel > Alarm2.AlarmLevel:
                    Alarm.AlarmLevel = Alarm1.AlarmLevel
                else:
                    Alarm.AlarmLevel = Alarm2.AlarmLevel
                # diaryout(Alarm, Debug)
                # diaryout("NewInfo:"+str(NewInfo), Debug)
                Sta_vars1.PGAEEW_times = Sta_vars1.PGAEEW_times + 1
                # if Alarm.PGA == 49.4:
                #     pass
                if Alarm.AlarmLevel >= 1:
                    Sta_vars1.AlarmFlag = 1
                    Sta_vars2.AlarmFlag = 1

                str11 = ('阈值报警！' + 'AlarmLevel:' + str(Alarm.AlarmLevel) + ',PGA:' + str(
                    Alarm.PGA) + ',delT:' + str(Alarm.delT) + ',recordtime:' + str(Alarm.recordtime) +
                         ',StartT:' + str(StartT) + ',Alarmtime:' + str(Alarm.Alarmtime) + ',NewInfo:' + str(
                            NewInfo) + '，报次：' + str(Sta_vars1.PGAEEW_times))
                logger.product(str11, 1, True)
            Sta_vars1.PGA_Curr = Alarm.PGA
            Sta_vars2.PGA_Curr = Alarm.PGA

    if Sta_vars1.EEW_times > 0 or persistent.FirstStart == 1:
        #
        if Sta_vars1.Azimuth < 0:
            try:
                if np.shape(Traces_evt)[0] < 6:
                    AZ1 = np.round(Azimuth3(Traces_evt), 2)
                else:
                    AZ1 = np.round(Azimuth3(Traces_evt[0:3]), 2)
                    AZ2 = np.round(Azimuth3(Traces_evt[3:6]), 2)
                    if AZ1 < 0:
                        AZ1 = np.round(Azimuth3(Traces_evt[0:3]), 2)
                    if AZ2 < 0:
                        AZ2 = np.round(Azimuth3(Traces_evt[3:6]), 2)
            except:
                NewInfo = 0
                return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

        if persistent.flag_nature == 0:
            Distance1 = 0
            if S_time1 < 0 or persistent.FirstStart == 1:
                [C1, Distance1] = Ct(Traces_evt1, scaler_distance, model_distance)
                # [C1, Distance1] = Ct1(Traces_evt1[2])
                # [C1, A1,  Distance1]=B_delta(Traces_evt1[2,:], Sprate)
                Sta_vars1.Distance_Pred = float(Distance1)
                Sta_vars1.CT=C1

            if np.shape(Data_now)[0] == 6 and abs(MaxEEW_times - int(Sta_vars1.EEW_times)) < 5:
                if S_time2 < 0 or persistent.FirstStart == 1:
                    # [C2, Distance2] = Ct(Traces_evt2, scaler_distance, model_distance)
                    [C2, Distance2] = Ct1(Traces_evt2[2])
                    Sta_vars2.CT = C2
                    # [C2, A2, Distance2] = B_delta(Traces_evt2[2,:], Sprate)
                    if Distance1 > 0:
                        Distance = (Distance2 + Distance1) / 2
                    else:
                        Distance = Distance2
                if Distance>118:
                    Distance = 117 + random.uniform(0, 0.5)
                Sta_vars1.Distance_Pred = float(Distance)
                Sta_vars2.Distance_Pred = float(Distance)

        # 震级计算
        if persistent.flag_nature == 0:
            baseinfo1 = trace_info(Traces_evt[0:3], Sprate)
            # Sta_vars1.p1=baseinfo1[0][0]
            # Sta_vars1.p2 = baseinfo1[0][1]
            # Sta_vars1.p3 = baseinfo1[0][2]
            # Sta_vars1.p4 = baseinfo1[0][3]
            # Sta_vars1.p5 = baseinfo1[0][4]


            new_data_scaled = scaler.transform(baseinfo1)
            Mag1 = magmodel.predict(new_data_scaled)
            MagBe = Sta_vars1.Magnitude
            if Mag1 > 8:
                Mag1 = 7.5 + 0.5 * abs(np.random.rand(1))
            if Mag1 < 4:
                Mag1 = 4.2 + abs(0.5 * np.random.rand(1))

            # if Sta_vars1.Magnitude > 0 and abs(Mag1 - Sta_vars1.Magnitude) < 0.3:
            #     pass
            # Sta_vars1.Magnitude = Sta_vars1.Magnitude + 0.5 * abs(np.random.uniform(-0.5,0.5))
            # elif Sta_vars1.Magnitude <= 0:
            Sta_vars1.Magnitude = Mag1
            #################################################
            MagnitudeFinal = Sta_vars1.Magnitude
            MagnitudeFinal = float(np.round(MagnitudeFinal, 1))
            if np.shape(Traces_evt)[0] == 6:
                if Sta_vars1.EEW_times == MaxEEW_times:

                    MagBe = np.round((Sta_vars2.Magnitude + Sta_vars1.Magnitude) / 2, 1)

                    baseinfo2 = trace_info(Traces_evt[3:6], Sprate)
                    # out2 = NetData.sim(baseinfo2)  # ndarray:(1,5),eg；ss1=[[1 2 3 4 5]]
                    # Mag2 = out2 * 9
                    new_data_scaled = scaler.transform(baseinfo2)
                    Mag2 = magmodel.predict(new_data_scaled)
                    if Mag2 > 8:
                        Mag2 = 7.5 + 0.5 * abs(np.random.rand(1))
                    if Mag2 < 4:
                        Mag2 = 4.2 + abs(0.5 * np.random.rand(1))
                    # if Sta_vars2.Magnitude > 0 and abs(Mag2 - Sta_vars2.Magnitude) < 0.3:
                    #     pass
                    # Sta_vars2.Magnitude = Sta_vars2.Magnitude +0.5 * abs(np.random.uniform(-0.5,0.5))
                    # elif Sta_vars2.Magnitude <= 0:
                    Sta_vars2.Magnitude = Mag2
                    MagnitudeFinal = (Sta_vars2.Magnitude + Sta_vars1.Magnitude) / 2
                    MagnitudeFinal = np.round(MagnitudeFinal, 1)
                else:
                    MagnitudeFinal = persistent.beMagnitude
            # ls

        # Stime识别
        if persistent.FirstStart == 1:
            dis_time = 0
        Sta_vars1.DurCurr = DurCurr
        Sta_vars2.DurCurr = DurCurr
        trigS1 = 0
        trigS2 = 0
        if S_time1 < 0:

            # start_time = time.time()
            # trigS1 = MyPickerS(Traces_evt, Sprate,persistent.flag_nature)
            # end_time = time.time()

            # start_time = datetime.now().timestamp()
             # if persistent.flag_nature == 1:
              #    trigS1 = MyPickerS(Traces_evt, Sprate, persistent.flag_nature)
            # end_time = datetime.now().timestamp()

            # delttime = end_time - start_time # 耗时
            # str11='traceLen: '+str(len(Traces_evt[0]))+', trigS1耗时：'+str(delttime)
            # logger.product(str11, 1, True)
            if trigS1 > 0:
                S_time1 = Sta_vars1.P_time + trigS1 / Sprate
                Sta_vars1.S_time = S_time1
                # Sta_vars1.EEW_times = EEW_Time_After_S
                S_time2 = Sta_vars2.P_time + trigS2 / Sprate
                Sta_vars2.S_time = S_time2
                # Sta_vars2.EEW_times = EEW_Time_After_S

        S_time1zp = Sta_vars1.S_time
        S_time2zp = Sta_vars2.S_time
        if Sta_vars1.S_time > 0:
            S_time1zp = Sta_vars1.S_time  # 观测S波到达时间
        if Sta_vars2.S_time > 0:
            S_time2zp = Sta_vars2.S_time  # 观测S波到达时间
        if Sta_vars1.S_time > 0 or Sta_vars2.S_time > 0:
            mmtimes = max(Sta_vars1.EEW_times, Sta_vars2.EEW_times)
            Sta_vars1.EEW_times = mmtimes
            Sta_vars2.EEW_times = mmtimes
            Distance = 8 * (max(Sta_vars1.S_time, Sta_vars2.S_time) - Sta_vars1.P_time)

        Sta_Long = Sta_vars1.STA_Long
        Sta_Lat = Sta_vars1.STA_Lat
        [Epi_Long, Epi_Lat] = LonLat(Sta_Long, Sta_Lat, AZ1, Distance)
        if persistent.flag_nature == 0:
            dis11 = Distance
            if dis11 + 25 < 139:
                dis11 = Distance + 20
            [Epi_Long, Epi_Lat] = LonLat(Sta_Long, Sta_Lat, AZ1, dis11)
            
        P_min = min(Sta_vars1.P_time, Sta_vars2.P_time)
        Epi_time = P_min - Distance / Pspeed
        S_time_cal = Epi_time + Distance / Sspeed
        # if DurCurr >= 3:
        tmepdata = Traces_evt
        tempV1 = iomega(tmepdata[0], Sprate, 1)
        tempV2 = iomega(tmepdata[1], Sprate, 1)
        tempV3 = iomega(tmepdata[2], Sprate, 1)
        tempD1 = iomega(tmepdata[0], Sprate, 2)
        tempD2 = iomega(tmepdata[1], Sprate, 2)
        tempD3 = iomega(tmepdata[2], Sprate, 2)
        # tempV1 = srs_integrationV(tmepdata[0], Sprate)
        # tempV2 = srs_integrationV(tmepdata[1], Sprate)
        # tempV3 = srs_integrationV(tmepdata[2], Sprate)
        # tempD1 = srs_integration(tmepdata[0], Sprate, 0)
        # tempD2 = srs_integration(tmepdata[1], Sprate, 0)
        # tempD3 = srs_integration(tmepdata[2], Sprate, 0)
        if persistent.First3s == 1:
            tempv = np.sqrt(tempV1[0:Sprate * 3 + 1] ** 2 + tempV2[0:Sprate * 3 + 1] ** 2 + tempV3[
                                                                                            0:Sprate * 3 + 1] ** 2)  # 小写的零时变量
            tempD = np.sqrt(tempD1[0:Sprate * 3 + 1] ** 2 + tempD2[0:Sprate * 3 + 1] ** 2 + tempD3[
                                                                                            0:Sprate * 3 + 1] ** 2)  # 小写的零时变量
            tracesV1_max = max(tempv)
            tracesD1_max = max(tempD)
            Sta_vars1.PGV_Curr = tracesV1_max
            Sta_vars1.PGD_Curr = tracesD1_max
            Sta_vars1.DurCurr = 3
            persistent.First3s = 0
        else:
            tempv = np.sqrt(tempV1 ** 2 + tempV2 ** 2 + tempV3 ** 2)
            tempD = np.sqrt(tempD1 ** 2 + tempD2 ** 2 + tempD3 ** 2)
            tracesV1_max = max(tempv)
            tracesD1_max = max(tempD)
            Sta_vars1.PGV_Curr = tracesV1_max
            Sta_vars1.PGD_Curr = tracesD1_max
            Sta_vars1.DurCurr = DurCurr

        if np.shape(Data_now)[0] >= 6:
            # if DurCurr >= 3:
            # tempV4 = iomega(tmepdata[3], Sprate, 1)
            # tempV5 = iomega(tmepdata[4], Sprate, 1)
            # tempV6 = iomega(tmepdata[5], Sprate, 1)
            # tempD4 = iomega(tmepdata[3], Sprate, 2)
            # tempD5 = iomega(tmepdata[4], Sprate, 2)
            # tempD6 = iomega(tmepdata[5], Sprate, 2)
            tempV4 = srs_integrationV(tmepdata[3], Sprate)
            tempV5 = srs_integrationV(tmepdata[4], Sprate)
            tempV6 = srs_integrationV(tmepdata[5], Sprate)
            tempD4 = srs_integration(tmepdata[3], Sprate, 0)
            tempD5 = srs_integration(tmepdata[4], Sprate, 0)
            tempD6 = srs_integration(tmepdata[5], Sprate, 0)
            if persistent.First3s2 == 1:
                tempv22 = np.sqrt(
                    tempV4[0:3 * Sprate] ** 2 + tempV5[0:3 * Sprate] ** 2 + tempV6[0:3 * Sprate] ** 2)
                tempD22 = np.sqrt(
                    tempD4[0:3 * Sprate] ** 2 + tempD5[0:3 * Sprate] ** 2 + tempD6[0:3 * Sprate] ** 2)
                tracesV2_max = max(tempv22)
                tracesD2_max = max(tempD22)
                Sta_vars1.PGV_Curr = tracesV2_max
                Sta_vars1.PGD_Curr = tracesD2_max
                Sta_vars1.DurCurr = 3
                persistent.First3s2 = 0
            else:
                tempv22 = np.sqrt(tempV4 ** 2 + tempV5 ** 2 + tempV6 ** 2)
                tempD22 = np.sqrt(tempD4 ** 2 + tempD5 ** 2 + tempD6 ** 2)
                tracesV2_max = max(tempv22)
                tracesD2_max = max(tempD22)
                Sta_vars2.DurCurr = DurCurr
                Sta_vars2.PGV_Curr = tracesV2_max
                Sta_vars2.PGD_Curr = tracesD2_max

        # # 20251015new
        # 增加新的输出变量FlagArea','P_ud_m'（值填入PGD_Curr位置）,'LenCh6','TCmean','sAvghm
        # Sta_vars1.PGD_Curr = PGD_Curr   # Sta_vars1.PGD_Curr = P_ud_m
        Sta_vars1.flagarea = EEWParams.flagarea
        Sta_vars2.flagarea = EEWParams.flagarea
        Sta_vars1.LenCh6 = len(Traces_evt[0,:])
        Sta_vars2.LenCh6 = len(Traces_evt[0, :])
        Sta_vars1.TC = baseinfo1[0,3]
        Sta_vars2.TC = baseinfo1[0,3]
        Sta_vars1.sAvghm = 0
        Sta_vars2.sAvghm = 0
        if persistent.FirstStart == 1:
            Epi_time = -1
            Epi_Long = -1
            Epi_Lat = -1
            AZ1 = -1000
            AZ2 = -1000
            Distance = -1
            S_time_cal = -1
            MagnitudeFinal = -1
            Sta_vars1.PGA_Curr = 0
            Sta_vars1.PGV_Curr = 0
            Sta_vars1.PGD_Curr = 0
            Sta_vars2.PGA_Curr = 0
            Sta_vars2.PGV_Curr = 0
            Sta_vars2.PGD_Curr = 0
            Alarm.AlarmLevel = -1
            Alarm.PGA = -1

        Sta_vars1.Magnitude = float(MagnitudeFinal)
        Sta_vars1.Epi_time = float(Epi_time)
        Sta_vars1.Epi_Long = float(Epi_Long)
        Sta_vars1.Epi_Lat = float(Epi_Lat)
        Sta_vars1.Azimuth = np.round(float(AZ1), 2)
        Sta_vars1.Distance_Pred = np.round(float(Distance), 2)
        Sta_vars1.S_time_cal = float(S_time_cal)

        if np.shape(Data_now)[0] >= 6:
            Sta_vars2.Magnitude = float(MagnitudeFinal)
            Sta_vars2.Epi_time = float(Epi_time)
            Sta_vars2.Epi_long = float(Epi_Long)
            Sta_vars2.Epi_Lat = float(Epi_Lat)
            Sta_vars2.Azimuth = float(AZ2)
            Sta_vars2.Distance_Pred = float(Distance)
            Sta_vars2.S_time_cal = float(S_time_cal)

        StationInfo.STA_Name = Sta_vars1.STA_Name
        StationInfo.STA_Long = Sta_vars1.STA_Long
        StationInfo.STA_Lat = Sta_vars1.STA_Lat
        StationInfo.P_time = Sta_vars1.P_time
        StationInfo.Magnitude = float(MagnitudeFinal)
        StationInfo.Azimuth = np.round(Sta_vars1.Azimuth, 2)
        StationInfo.Distance = np.round(Sta_vars1.Distance_Pred, 2)
        StationInfo.Epi_time = Sta_vars1.Epi_time
        StationInfo.Epi_Long = Sta_vars1.Epi_Long
        StationInfo.Epi_Lat = Sta_vars1.Epi_Lat
        StationInfo.PGA_Pred = Sta_vars1.PGA_Pred

        StationInfo.S_time_cal = Sta_vars1.S_time_cal
        if np.shape(Data_now)[0] < 6:
            StationInfo.PGA_Curr = np.round(Sta_vars1.PGA_Curr, 1)
            StationInfo.PGV_Curr = Sta_vars1.PGV_Curr
            StationInfo.PGD_Curr = Sta_vars1.PGD_Curr
            StationInfo.S_time = S_time1zp
        else:
            StationInfo.PGA_Curr = np.round((PGA1 + PGA2) / 2, 1)
            StationInfo.PGV_Curr = max(Sta_vars1.PGV_Curr, Sta_vars2.PGV_Curr)
            StationInfo.PGD_Curr = max(Sta_vars1.PGD_Curr, Sta_vars2.PGD_Curr)
            StationInfo.S_time = max(S_time1zp, S_time2zp)
        StationInfo.DurCurr = Sta_vars1.DurCurr
        Sta_vars1.LastEEWTime = StartT
        Sta_vars2.LastEEWTime = StartT



        if persistent.FirstStart == 0:
            if (MaxEEW_times - Sta_vars1.EEW_times) <= 3 or (
                    (MaxEEW_times - Sta_vars1.EEW_times) > 3 and (MagnitudeFinal != MagBe)) or \
                    np.round(AZ1,
                             2) != persistent.AZIold or Sta_vars1.Distance_Pred != persistent.distold or trigS1 > 0 or trigS2 > 0 or \
                    np.round(Alarm.PGA, 1) > np.round(persistent.PGAold, 1):
                if (Alarm.PGA >= 40 and persistent.PGAold < Alarm.PGA) and (MagnitudeFinal != MagBe or np.round(AZ1,
                                                                                                                2) != persistent.AZIold or StationInfo.Distance != persistent.distold):
                    NewInfo = 3
                    Sta_vars1.NewInfo = NewInfo
                    Sta_vars2.NewInfo = NewInfo
                    Sta_vars1.EEW_times = Sta_vars1.EEW_times - 1
                    Sta_vars2.EEW_times = Sta_vars2.EEW_times - 1
                elif (Alarm.PGA < 40) and (MagnitudeFinal != MagBe or np.round(AZ1,
                                                                               2) != persistent.AZIold or StationInfo.Distance != persistent.distold):
                    NewInfo = 1
                    Sta_vars1.NewInfo = NewInfo
                    Sta_vars2.NewInfo = NewInfo
                    Sta_vars1.EEW_times = Sta_vars1.EEW_times - 1
                    Sta_vars2.EEW_times = Sta_vars2.EEW_times - 1
                elif (Alarm.PGA >= 40 and persistent.PGAold < Alarm.PGA) and (MagnitudeFinal == MagBe and np.round(AZ1,
                                                                                                                   2) == persistent.AZIold and StationInfo.Distance == persistent.distold and persistent.PGAold < Alarm.PGA):
                    NewInfo = 2
                    Sta_vars1.NewInfo = NewInfo
                    Sta_vars2.NewInfo = NewInfo
                    Sta_vars1.EEW_times = Sta_vars1.EEW_times - 1
                    Sta_vars2.EEW_times = Sta_vars2.EEW_times - 1
                else:
                    NewInfo = 1
                    Sta_vars1.NewInfo = NewInfo
                    Sta_vars2.NewInfo = NewInfo
                    Sta_vars1.EEW_times = Sta_vars1.EEW_times - 1
                    Sta_vars2.EEW_times = Sta_vars2.EEW_times - 1

            if NewInfo == 1 or NewInfo == 3 or NewInfo == 2 and (PGA1 + PGA2) / 2 >= persistent.PGAoldout + 0.5:
                PGA_Pred = -1
                if StationInfo.Magnitude > 0 and StationInfo.Magnitude <= 6.5:
                    PGA_Pred = 10**(1.979 + 0.671 * StationInfo.Magnitude - 2.315 * np.log10(
                        StationInfo.Distance + 2.088 * math.exp(0.399 * StationInfo.Magnitude)))
                elif StationInfo.Magnitude > 6.5:
                    PGA_Pred = 10**(3.533 + 0.432 * StationInfo.Magnitude - 2.315 * np.log10(
                        StationInfo.Distance + 2.088 * math.exp(0.399 * StationInfo.Magnitude)))
                Sta_vars1.PGA_Pred = PGA_Pred
                StationInfo.PGA_Pred = PGA_Pred
                txt1 = ('预测PGA:' + str(Sta_vars1.PGA_Pred))
                if Debug > 0:
                    logger.product(txt1, 1, True)

                PGA_Pred = -1
                if StationInfo.Magnitude > 0 and StationInfo.Magnitude <= 6.5:
                    PGA_Pred = 10 ** (1.979 + 0.671 * StationInfo.Magnitude - 2.315 * np.log10(
                        StationInfo.Distance + 2.088 * math.exp(0.399 * StationInfo.Magnitude)))
                elif StationInfo.Magnitude > 6.5:
                    PGA_Pred = 10 ** (3.533 + 0.432 * StationInfo.Magnitude - 2.315 * np.log10(
                        StationInfo.Distance + 2.088 * math.exp(0.399 * StationInfo.Magnitude)))
                Sta_vars1.PGA_Pred = PGA_Pred
                StationInfo.PGA_Pred = PGA_Pred
                txt1 = ('预测PGA:' + str(Sta_vars1.PGA_Pred))
                if Debug > 0:
                    logger.product(txt1, 1, True)

                if persistent.PGAold < Alarm.PGA:
                    persistent.PGAold = np.copy(Alarm.PGA)
                    StationInfo.PGA_Curr = np.round(np.copy(persistent.PGAold), 1)
                else:
                    StationInfo.PGA_Curr = np.round(np.copy(persistent.PGAold), 1)
                currtime = datetime.now()
                currtime3 = np.round(currtime.timestamp(), 3)
                txt1 = ('地震预警！震级:' + str(StationInfo.Magnitude) + ',方位角:' + str(StationInfo.Azimuth)
                        + ',震中距:' + str(StationInfo.Distance) + ',PGA:' + str(StationInfo.PGA_Curr)  + ',traceLen:' + str(len(Traces_evt[0]))+
                        ',Newinfo:' + str(NewInfo) + ',Ptime:' + str(StationInfo.P_time) + ',StartT:' + str(
                            StartT) + ',报次：' + str(
                            int(MaxEEW_times - Sta_vars1.EEW_times))
                        + ',Alarmtime:' + str(currtime3))
                if Debug>0:
                    logger.product(txt1, 1, True)
                persistent.PGAoldout = np.round((PGA1 + PGA2) / 2, 2)
                persistent.beMagnitude = np.copy(MagnitudeFinal)
                persistent.AZIold = np.copy(Sta_vars1.Azimuth)
                persistent.distold = np.copy(Sta_vars1.Distance_Pred)

    persistent.FirstStart = 0
    Sta_vars1 = Sta_vars1
    Sta_vars2 = Sta_vars2

    return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm


def CallPag(Data_now):
    global Sta_vars1
    global Sta_vars2
    if Sta_vars1.Magnitude <= 0:
        return 0, Sta_vars1, 0
    Data_now = np.transpose(Data_now)
    P11 = Data_now[0] - np.mean(Sta_vars1.BaseLine[0])
    P12 = Data_now[1] - np.mean(Sta_vars1.BaseLine[1])
    P13 = Data_now[2] - np.mean(Sta_vars1.BaseLine[2])
    P21 = Data_now[3] - np.mean(Sta_vars2.BaseLine[0])
    P22 = Data_now[4] - np.mean(Sta_vars2.BaseLine[1])
    P23 = Data_now[5] - np.mean(Sta_vars2.BaseLine[2])
    pga1 = np.sqrt(P11 ** 2 + P12 ** 2 + P13 ** 2)
    pga2 = np.sqrt(P21 ** 2 + P22 ** 2 + P23 ** 2)
    pag = round((np.max(pga1) + np.max(pga2)) / 2, 2)
    if pag >= 40 and pag > (Sta_vars1.PGA_Curr + Sta_vars2.PGA_Curr) / 2:
        return 2, Sta_vars1, pag
    else:
        return 0, Sta_vars1, 0


def CallInit():
    global Sta_vars1
    global Sta_vars2
    MaxEEW_times = float(EEWParams.MaxEEW_times)
    Sta_vars1.ini2(MaxEEW_times)
    Sta_vars2.ini2(MaxEEW_times)


#
# def subplot22(delt, Sta_vars_ret1, Sta_vars_ret2, Sprate, packageLen, min_time):
#     # delt:发包的长度
#     # ax:list,子图1的横坐标
#     # ay:list,组图1的纵坐标
#     # bx:list,子图2的横坐标
#     # by:list,组图2的纵坐标
#     # Sta_vars_ret1:提取所需信息
#     # Sprate：采样率
#     # packageLen:最新包长
#     # min_time:时间归一起点
#     sensordata1 = Sta_vars_ret1.Buffer  # Buffer
#     sensordata2 = Sta_vars_ret2.Buffer  # Buffer
#     n1 = len(sensordata1[2])
#     st0 = Sta_vars_ret1.StartT + (packageLen - n1) / Sprate
#
#     tout = cmp_t(n1, Sprate, st0 - min_time)
#     ax = tout.tolist()
#     bx = tout.tolist()
#     ay = sensordata1[2]
#     by = sensordata2[2]
#     ptime = Sta_vars_ret1.P_time
#     TrigPN = int((Sta_vars_ret1.P_time - st0) * Sprate)  # POINTS
#     stime = Sta_vars_ret1.S_time
#     TrigSN = int((Sta_vars_ret1.S_time - st0) * Sprate)  # POINTS
#     endtime = Sta_vars_ret1.End_time
#     TrigEN = int((Sta_vars_ret1.End_time - st0) * Sprate)  # POINTS
#     Is_EQK = Sta_vars_ret1.Is_EQK
#
#     # plt.ion()    # 开启一个画图的窗口进入交互模式，用于实时更新数据
#     # plt.rcParams['savefig.dpi'] = 200 #图片像素
#     # plt.rcParams['figure.dpi'] = 200 #分辨率
#     plt.rcParams['figure.figsize'] = (5, 3.5)  # 图像显示大小
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码，还有通过导入字体文件的方法
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['lines.linewidth'] = 0.5  # 设置曲线线条宽度
#     plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
#     plt.suptitle("实时波形")  # , fontsize=5, 添加总标题，并设置文字大小
#
#     # 图表1
#     agraphic = plt.subplot(2, 1, 1)
#     agraphic.set_title('传感器1_未去基线')  # 添加子标题
#     agraphic.set_xlabel('x轴')  # , fontsize=5, 添加轴标签
#     agraphic.set_ylabel('y轴')  # , fontsize=5
#     agraphic.plot(ax, ay, 'b-')  # 等于agraghic.plot(ax,ay,'g-')
#     if TrigPN > 0 and TrigPN < n1:
#         agraphic.plot(ax[TrigPN], ay[TrigPN], 'r.', linewidth=2)  # 触发位置加入图示中
#     if TrigSN > 0 and TrigSN < n1:
#         agraphic.plot(ax[TrigSN], ay[TrigSN], 'g.', linewidth=2)  # S触发位置加入图示中
#     if TrigEN > 0 and TrigEN < n1:
#         agraphic.plot(ax[TrigEN], ay[TrigEN], 'k.', linewidth=2)  # end位置加入图示中
#
#     # 图表2
#     bgraghic = plt.subplot(2, 1, 2)
#     bgraghic.plot(bx, by, 'b-')
#     bgraghic.set_xlabel('x轴')  # , fontsize=5, 添加轴标签
#     bgraghic.set_ylabel('y轴')  # , fontsize=5
#     if TrigPN > 0 and TrigPN < n1:
#         bgraghic.plot(ax[TrigPN], ay[TrigPN], 'r.', linewidth=2)  # 触发位置加入图示中
#     if TrigSN > 0 and TrigSN < n1:
#         bgraghic.plot(ax[TrigSN], ay[TrigSN], 'g.', linewidth=2)  # S触发位置加入图示中
#     if TrigEN > 0 and TrigEN < n1:
#         bgraghic.plot(ax[TrigEN], ay[TrigEN], 'k.', linewidth=4)  # end位置加入图示中
#     bgraghic.set_title('传感器2_未去基线')
#     plt.pause(0.001)  # 设置暂停时间，太快图表无法正常显示
#     # plt.show()