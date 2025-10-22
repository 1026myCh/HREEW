# -*-coding:UTF-8 -*-
import datetime
import warnings
import EEW_Triger2
import SimpleLogger as logger
from Azimuth2 import Azimuth2
from EEW_Triger3 import EEW_Triger2_Nature
from EEW_Triger3 import MypickerS
from Judges import Judges
from StaticVar import StaticVar as persistent  # 静态类
from class_StaV import Sta_V
from class_obj import AlarmS
from function1 import *
from Nature_Distance import cal_distance
from Nature_Magnitude import cal_magnitude
from Nature_function import *
import torch
from srs_integration import srs_integration, srs_integrationV
from Azimuth3 import Azimuth3
import os
warnings.filterwarnings("ignore")

# def EEW_Single(Data_now, StartT, Lon, Lat, Stv_arg1, Stv_arg2):
# 天然模式入口
# Sta_vars1 = Sta_V()
# Sta_vars2 = Sta_V()
# Sta_vars_ret1 = Sta_V()
# Sta_vars_ret2 = Sta_V()
# Alarm1 = AlarmS()
# Alarm2 = AlarmS()
# 添加触发模式
model_Pwave = None
model_Swave = None


def Nature_Single(Data_now, StartT,MaxEEW_times, StationInfo, NewInfo, Debug, Sprate, ThreshGals, S_time2,
                  Buffer_len, EEW_Time_After_S, Pspeed, Sspeed, Alarm, Sta_vars1, Sta_vars2,
                Alarm1, Alarm2,flagarea,in_model,Gain, continue_on_non_eqk=False):
    global model_Pwave
    global model_Swave

    current_path = os.path.dirname(os.path.abspath(__file__))
    if model_Pwave is None or model_Swave is None:
        model_Pwave = torch.jit.load(os.path.join(current_path, 'china.rnn.jit'),map_location=torch.device('cpu'))
        model_Swave = torch.jit.load(os.path.join(current_path, 'rnn.pnsn.01.jit'),map_location=torch.device('cpu'))  # axis=1

    sum_Value = np.shape(Data_now)[0]
    [Sta_vars1, Sta_vars2, disAlarm1, Alarm] = EEW_Triger2_Nature(Sta_vars1, Sta_vars2, Alarm, model_Pwave, model_Swave)

    disAlarm2 = disAlarm1
    if Sta_vars1.End_time > 1 or persistent.FirstStart == 1:
        Sta_vars1.ini2(MaxEEW_times,persistent.FirstStart, Alarm)  # 初始化

    if Sta_vars2.End_time > 1 or persistent.FirstStart == 1 or Sta_vars1.End_time > 1:
        endtime_max = np.max([Sta_vars1.End_time, Sta_vars2.End_time])
        Sta_vars1.ini2(MaxEEW_times,persistent.FirstStart, Alarm)  # 初始化
        Sta_vars2.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
        Sta_vars1.End_time = endtime_max
        Sta_vars2.End_time = endtime_max
    if sum_Value < 6:
        if disAlarm1 == 1:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            NewInfo = 4
            Sta_vars1.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            Sta_vars2.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
        elif disAlarm2 == 2:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            Sta_vars1.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            Sta_vars2.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            NewInfo = 100
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
    else:
        if disAlarm1 == 1 or disAlarm2 == 1:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            NewInfo = 4
            Sta_vars1.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            Sta_vars2.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
        elif disAlarm1 == 2 or disAlarm2 == 2:
            Alarm1 = AlarmS()
            Alarm2 = AlarmS()
            NewInfo = 100
            Sta_vars1.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            Sta_vars2.ini2(MaxEEW_times,persistent.FirstStart, Alarm)
            return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    LastAlarmTime = [Sta_vars1.LastAlarmTime, Sta_vars2.LastAlarmTime]  # 后续还需改进,前后不一致！！！！！
    Distance = Sta_vars1.Distance_Pred
    AZ1 = Sta_vars1.Azimuth
    S_time1 = Sta_vars1.S_time
    AZ2 = Sta_vars2.Azimuth
    S_time2 = Sta_vars2.S_time

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
    traces_org = np.copy(Traces_evt)
    DurCurr = np.shape(Traces_evt1)[1] / Sprate

    if DurCurr < 2 and persistent.FirstStart != 1:
        NewInfo = 0
        return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    if Sta_vars1.Is_EQK < 1 or Sta_vars2.Is_EQK < 1:
        [result,corrout] = Judges(Traces_evt, 'multi_parameters')
        temp = np.shape(Traces_evt)[1]
        # if temp>1100:
        #     result=0
        [PGA1, PGA2, PGApos1, PGApos2] = cmp_vector(Traces_evt)
        PGA1 = float(PGA1)
        PGA2 = float(PGA2)

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
                # 根据2025年cy
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
                        Alarm.PGA) + ',delT:' + str(Alarm.delT) + ',recordtime:' + str(Alarm.recordtime) +
                             ',StartT:' + str(StartT) + ',Alarmtime:' + str(Alarm.Alarmtime) + ',NewInfo:' + str(
                                NewInfo) + '，报次：' + str(Sta_vars1.PGAEEW_times))
                    logger.product(str11, 1, True)
            elif PGA1 > 0 and PGA2 > 0 and np.abs(PGA1 - PGA2) / maxPGA < 0.3:
                if (PGA1 + PGA2) / 2 < Sta_vars1.PGAcurrold:
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
            if persistent.FirstStart == 0 and not continue_on_non_eqk:
                # if Debug == 1:
                #     str1 = '非地震、非地震、非地震'
                #     logger.product(str1, 1, True)
                return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm
            # return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

    # 条件结合新修改的地震状态标志修改！20251011
    # if Sta_vars1.Is_EQK >= 0:
    SNout = snr(Sta_vars1.Package,np.vstack((Sta_vars1.Buffer, Sta_vars2.Buffer)),Sta_vars1.StartT,Sta_vars1.P_time,2)
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
    P = np.array([P11, P12, P13, P21, P22, P23])
    # [PGA1, PGA2] = cmp_vector(P)
    [PGA1, PGA2, PGApos1, PGApos2] = cmp_vector(P)
    PGA1 = np.round(PGA1, 1)
    PGA2 = np.round(PGA2, 1)
    MM = max(PGA1, PGA2)
    pos12 = (PGApos1 + PGApos2) / 2
    lenbase = len(Sta_vars2.BaseLine[2])
    PGAt = StartT + (Data_now.shape[0] + pos12 - lenbase) / Sprate  # 秒，浮点数
    temp_CurrPGA = (Sta_vars1.PGA_Curr + Sta_vars2.PGA_Curr) / 2  # 上一包PGA
    if PGA1 > 0 and PGA2 > 0 and np.abs(PGA1 - PGA2) / MM > 0.3:
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
                Alarm.PGA) + ',traceLen:' + str(len(Traces_evt[0]))+',delT:' + str(Alarm.delT) + ', recordtime:' + str(Alarm.recordtime) +
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
            if Alarm1.AlarmLevel > 0 and Alarm2.AlarmLevel > 0 and Sta_vars1.Is_EQK > 0 and Sta_vars2.Is_EQK > 0:
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
            AZ1 = np.round(Azimuth2(Traces_evt[0:3]), 2)
            AZ2 = np.round(Azimuth2(Traces_evt[3:6]), 2)
            if AZ1 < 0:
                AZ1 = np.round(Azimuth2(Traces_evt[0:3]), 2)
            if AZ2 < 0:
                AZ2 = np.round(Azimuth2(Traces_evt[3:6]), 2)
        tempAzi = (AZ1 + AZ2) / 2
        AZ1 = tempAzi
        AZ2 = tempAzi

    if persistent.FirstStart == 0:
        if S_time1 < 0 and S_time2 < 0:
            S_time = MypickerS(Sta_vars1, Sta_vars2, Traces_evt, Sprate, model_Swave)
            if S_time > 0:
                Sta_vars1.S_time = S_time
                Sta_vars2.S_time = S_time
                print("识别到S波")
                S_time2 = S_time
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

    if abs(MaxEEW_times - int(Sta_vars1.EEW_times)) < 2:
        # if (S_time2 < 0 or persistent.FirstStart == 1):
        Distance1 = cal_distance(Traces_evt1[2], Sprate)
        Distance2 = cal_distance(Traces_evt2[2], Sprate)
        Distance = (Distance2 + Distance1) / 2

    # 震级计算
    if persistent.flag_nature == 1:
        flagarea = 3
        flag_Sfirst = 0
        PGAnew = Sta_vars1.PGA_Curr
        if Sta_vars1.Magnitude <= 0:
            if Sta_vars1.S_time - Sta_vars1.S_timeold > 0:  ## &&Sta_vars1.S_timeold <= 0
                flag_Sfirst = 1
            Mag,P_ud_m,TC,sAvghm = cal_magnitude(Traces_evt, Distance, 0, Sprate,Debug, Sta_vars1.S_time, flagarea
                  ,Sta_vars1.PGAcurrold, PGAnew, flag_Sfirst, in_model,Gain, flag=None)
            Mag = np.round(Mag, 1)
            if Mag>7.5:
                Mag=7.5
            MagBe = Mag
            if Sta_vars1.Is_EQK < 0:
                MagBe = -1
            Sta_vars1.Magnitude = Mag
            MagnitudeFinal = Sta_vars1.Magnitude
        else:
            if Sta_vars1.S_time - Sta_vars1.S_timeold > 0:  ## & & Sta_vars1.S_timeold <= 0
                flag_Sfirst = 1
            Mag,P_ud_m,TC,sAvghm = cal_magnitude(Traces_evt, Distance, Sta_vars1.Magnitude, Sprate,Debug, Sta_vars1.S_time, flagarea
                                , Sta_vars1.PGAcurrold, PGAnew, flag_Sfirst, in_model, Gain, flag=None)
            Mag = np.round(Mag, 1)
            if Mag>7.5:
                Mag=7.5
            MagBe = Sta_vars1.Magnitude
            if Sta_vars1.Is_EQK < 0:
                MagBe = -1
            Sta_vars1.Magnitude = Mag
            MagnitudeFinal = Sta_vars1.Magnitude

        Sta_vars1.S_timeold = Sta_vars1.S_time
        # 20251015new,位置及内容待完善
        # 增加新的输出变量FlagArea','P_ud_m'（值填入PGD_Curr位置）,'LenCh6','TCmean','sAvghm
        Sta_vars1.flagarea = flagarea
        Sta_vars2.flagarea = flagarea
        # 与matlab一致，PGD_Curr更新为sAvghm。后期可精简。20251015
        Sta_vars1.PGD_Curr = P_ud_m
        Sta_vars2.PGD_Curr = P_ud_m
        Sta_vars1.LenCh6 = len(Traces_evt[0,:])
        Sta_vars2.LenCh6 = len(Traces_evt[0,:])
        Sta_vars1.TC = TC
        Sta_vars2.TC = TC
        Sta_vars1.sAvghm = sAvghm
        Sta_vars2.sAvghm = sAvghm

    # Stime识别
    if persistent.FirstStart == 1:
        dis_time = 0
    Sta_vars1.DurCurr = DurCurr
    Sta_vars2.DurCurr = DurCurr
    trigS1 = 0
    trigS2 = 0

    if persistent.FirstStart == 0:
        if S_time1 < 0 and S_time2 < 0:
            S_time = MypickerS(Sta_vars1, Sta_vars2, Traces_evt, Sprate, model_Swave)
            if S_time > 0:
                Sta_vars1.S_time = S_time
                Sta_vars2.S_time = S_time
                print("识别到S波")
                # Sta_vars1.EEW_times = EEW_Time_After_S

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
    # if persistent.flag_nature == 0:
    #     dis11 = Distance
    #     if dis11 + 25 < 139:
    #         dis11 = Distance + 20
    #     [Epi_Long, Epi_Lat] = LonLat(Sta_Long, Sta_Lat, AZ1, dis11)
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
        # Sta_vars1.PGD_Curr = tracesD1_max
        Sta_vars1.DurCurr = 3
        persistent.First3s = 0
    else:
        tempv = np.sqrt(tempV1 ** 2 + tempV2 ** 2 + tempV3 ** 2)
        tempD = np.sqrt(tempD1 ** 2 + tempD2 ** 2 + tempD3 ** 2)
        tracesV1_max = max(tempv)
        tracesD1_max = max(tempD)
        Sta_vars1.PGV_Curr = tracesV1_max
        # Sta_vars1.PGD_Curr = tracesD1_max
        Sta_vars1.DurCurr = DurCurr

    if np.shape(Data_now)[0] >= 6:
        # if DurCurr >= 3:
        tempV4 = iomega(tmepdata[3], Sprate, 1)
        tempV5 = iomega(tmepdata[4], Sprate, 1)
        tempV6 = iomega(tmepdata[5], Sprate, 1)
        tempD4 = iomega(tmepdata[3], Sprate, 2)
        tempD5 = iomega(tmepdata[4], Sprate, 2)
        tempD6 = iomega(tmepdata[5], Sprate, 2)
        # tempV4 = srs_integrationV(tmepdata[3], Sprate)
        # tempV5 = srs_integrationV(tmepdata[4], Sprate)
        # tempV6 = srs_integrationV(tmepdata[5], Sprate)
        # tempD4 = srs_integration(tmepdata[3], Sprate, 0)
        # tempD5 = srs_integration(tmepdata[4], Sprate, 0)
        # tempD6 = srs_integration(tmepdata[5], Sprate, 0)
        if persistent.First3s2 == 1:
            tempv22 = np.sqrt(
                tempV4[0:3 * Sprate] ** 2 + tempV5[0:3 * Sprate] ** 2 + tempV6[0:3 * Sprate] ** 2)
            tempD22 = np.sqrt(
                tempD4[0:3 * Sprate] ** 2 + tempD5[0:3 * Sprate] ** 2 + tempD6[0:3 * Sprate] ** 2)
            tracesV2_max = max(tempv22)
            tracesD2_max = max(tempD22)
            Sta_vars1.PGV_Curr = tracesV2_max
            # Sta_vars1.PGD_Curr = tracesD2_max
            Sta_vars1.DurCurr = 3
            persistent.First3s2 = 0
        else:
            tempv22 = np.sqrt(tempV4 ** 2 + tempV5 ** 2 + tempV6 ** 2)
            tempD22 = np.sqrt(tempD4 ** 2 + tempD5 ** 2 + tempD6 ** 2)
            tracesV2_max = max(tempv22)
            tracesD2_max = max(tempD22)
            Sta_vars2.DurCurr = DurCurr
            Sta_vars2.PGV_Curr = tracesV2_max
            # Sta_vars2.PGD_Curr = tracesD2_max

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
                         2) != Sta_vars1.AZIold or Sta_vars1.Distance_Pred != Sta_vars1.distold or trigS1 > 0 or trigS2 > 0 or \
                np.round(Alarm.PGA, 1) > np.round(Sta_vars1.PGAcurrold, 1):
            if (Alarm.PGA >= 40 and Sta_vars1.PGAold < Alarm.PGA) and (MagnitudeFinal != MagBe or np.round(AZ1,
                                                                                                            2) != Sta_vars1.AZIold or StationInfo.Distance != Sta_vars1.distold):
                NewInfo = 3
                Sta_vars1.NewInfo = NewInfo
                Sta_vars2.NewInfo = NewInfo
                Sta_vars1.EEW_times = Sta_vars1.EEW_times - 1
                Sta_vars2.EEW_times = Sta_vars2.EEW_times - 1
            elif (Alarm.PGA < 40) and (MagnitudeFinal != MagBe or np.round(AZ1,
                                                                           2) != Sta_vars1.AZIold or StationInfo.Distance != Sta_vars1.distold):
                NewInfo = 1
                Sta_vars1.NewInfo = NewInfo
                Sta_vars2.NewInfo = NewInfo
                Sta_vars1.EEW_times = Sta_vars1.EEW_times - 1
                Sta_vars2.EEW_times = Sta_vars2.EEW_times - 1
            elif (Alarm.PGA >= 40 and Sta_vars1.PGAold < Alarm.PGA) and (MagnitudeFinal == MagBe and np.round(AZ1,
                                                                                                               2) == Sta_vars1.AZIold and StationInfo.Distance == Sta_vars1.distold and Sta_vars1.PGAold < Alarm.PGA):
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

        if NewInfo == 1 or NewInfo == 3 or NewInfo == 2 and (PGA1 + PGA2) / 2 >= Sta_vars1.PGAcurrold + 0.5:
            if Sta_vars1.PGAold != -1:
                if Sta_vars1.PGAold < Alarm.PGA:
                    Sta_vars1.PGAold = np.copy(Alarm.PGA)
                    StationInfo.PGA_Curr = np.round(np.copy(Sta_vars1.PGAold), 1)
                else:
                    StationInfo.PGA_Curr = np.round(np.copy(Sta_vars1.PGAold), 1)
            else:
                Sta_vars1.PGAold = StationInfo.PGA_Curr
            currtime = datetime.now()
            currtime3 = np.round(currtime.timestamp(), 3)
            if math.isnan(StationInfo.Magnitude)==1:
                print('mag error')

            txt1 = ('地震预警！震级:' + str(StationInfo.Magnitude) + ',方位角:' + str(StationInfo.Azimuth)
                    + ',震中距:' + str(StationInfo.Distance) + ',PGA:' + str(StationInfo.PGA_Curr)  + ',traceLen:' + str(len(Traces_evt[0]))+
                    ',Newinfo:' + str(NewInfo) + ',Ptime:' + str(StationInfo.P_time) + ',StartT:' + str(
                        StartT) + ',报次：' + str(
                        int(MaxEEW_times - Sta_vars1.EEW_times))
                    + ',Alarmtime:' + str(currtime3))
            logger.product(txt1, 1, True)
            # PGA、AZI、Dist存到Sta_vars1，作为下一报次的参考
            Sta_vars1.PGAcurrold = np.round((PGA1 + PGA2) / 2, 2)
            persistent.beMagnitude = np.copy(MagnitudeFinal)
            Sta_vars1.AZIold = np.copy(Sta_vars1.Azimuth)
            Sta_vars1.distold = np.copy(Sta_vars1.Distance_Pred)

    persistent.FirstStart = 0
    Sta_vars1 = Sta_vars1
    Sta_vars2 = Sta_vars2

    return NewInfo, StationInfo, Sta_vars1, Sta_vars2, StartT, Alarm

