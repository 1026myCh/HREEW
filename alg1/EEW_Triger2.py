# -*-coding:UTF-8 -*-
import math
from collections import Counter

import numpy as np
import scipy.integrate as integrate
import scipy.signal as filter
from scipy.signal import find_peaks

import SimpleLogger as logger
from StaticVar import StaticVar as persistent  # 静态类
from StaticVar import Static_EEW_Params as EEW_Params  # 静态类
from function1 import MyFilter


def EEW_Triger2(Stv_arg1,Stv_arg2, Alarm):
    # siez of input data is[6，:],input class Stv_arg output classs Sta_vars,
    # 用于数据触发判断 STA/LTA触发法,此脚本为触发模块
    Debug = int(EEW_Params.Debug)
    MinDuration = float(EEW_Params.MinDuration)
    MaxEEW_times = int(EEW_Params.MaxEEW_times)
    LongestNonEqk = float(EEW_Params.LongestNonEqk)
    LTW = int(EEW_Params.LTW)
    Sprate = int(EEW_Params.Sprate)
    disAlarm = 0
    L_fs = float(EEW_Params.L_fs)
    H_fs = float(EEW_Params.H_fs)
    Sta_vars1 = Stv_arg1
    Sta_vars2 = Stv_arg2
    Triged = Sta_vars1.Triged
    Buffer1 = np.copy(Sta_vars1.Buffer)
    Buffer2 = np.copy(Sta_vars2.Buffer)
    Buffer=np.vstack((Buffer1,Buffer2))
    filter_Buffer1 = np.copy(Sta_vars1.filter_Buffer)  # hampel滤波后面需要添加
    filter_Buffer2 = np.copy(Sta_vars2.filter_Buffer)  # 中值滤波后面需要添加
    filter_Buffer=np.vstack((filter_Buffer1,filter_Buffer2))
    StartT = float(Sta_vars1.StartT)
    PackLen = int(np.shape(Sta_vars1.Package)[1])
    N_pnsr1 = persistent.N_pnsr1
    N_pnsr2 = persistent.N_pnsr2

    Buffer_len = int(np.shape(Buffer)[1])
    if Buffer_len>3000:
        Buffer_len=3000
    # if PackLen>2980:
    #     PackLen=2980


    # 事件捡拾
    if Triged < 1:  # 未触发
        # try:
        endtimepre = Stv_arg1.End_time  # Stv_arg1.End_time
        startT = Stv_arg1.StartT
        Trig_fine = MypickAIC(Sprate, PackLen, filter_Buffer,Buffer,endtimepre,startT)  # 使用AIC法

        if Trig_fine <=0:
            return Sta_vars1,Sta_vars2,disAlarm, Alarm

        # mdenoise module

        Triged = 1
        Sta_vars1.Triged = Triged
        Sta_vars2.Triged = Triged
        Triged_time = round(StartT + (Trig_fine - Buffer_len)/Sprate, 3)
        Sta_vars1.End_time = -1
        Sta_vars2.End_time = -1
        Sta_vars1.triger_fine = Trig_fine
        Sta_vars2.triger_fine = Trig_fine

        if (Triged_time <= persistent.end_time_Pre and Triged_time - persistent.P_time_Pre <= 0.1 or
                Triged_time <= persistent.S_time_Pre):  # Trig_time <= end_time_Pre):
            Triged = -1
            Sta_vars1.Triged = Triged
            Sta_vars2.Triged = Triged
            return Sta_vars1,Sta_vars2, disAlarm,Alarm

        Sta_vars1.Traces_evt = Sta_vars1.BaseLine[:, Trig_fine:]
        Sta_vars2.Traces_evt = Sta_vars2.BaseLine[:, Trig_fine:]
        Sta_vars1.P_time = Triged_time
        Sta_vars2.P_time = Triged_time

        if Debug == 1:
            temp = np.shape(Sta_vars1.Traces_evt)[1]
            str1 = '触发、触发、触发，Traces_evt length:' + str(temp)
            logger.product(str1, 1, True)

        Buffer_f0 = MyFilter(Sprate, Buffer[0], L_fs, H_fs)  # 注意核查
        Buffer_f1 = MyFilter(Sprate, Buffer[1], L_fs, H_fs)  # 注意核查
        Buffer_f2 = MyFilter(Sprate, Buffer[2], L_fs, H_fs)  # 注意核查
        Buffer_f3 = MyFilter(Sprate, Buffer[3], L_fs, H_fs)  # 注意核查
        Buffer_f4 = MyFilter(Sprate, Buffer[4], L_fs, H_fs)  # 注意核查
        Buffer_f5 = MyFilter(Sprate, Buffer[5], L_fs, H_fs)  # 注意核查

        if Trig_fine < Sprate:
            noisele1 = Buffer_f0[0:Trig_fine]
            noiseln1 = Buffer_f1[0:Trig_fine]
            noiselz1 = Buffer_f2[0:Trig_fine]
            noisele2 = Buffer_f3[0:Trig_fine]
            noiseln2 = Buffer_f4[0:Trig_fine]
            noiselz2 = Buffer_f5[0:Trig_fine]
        else:
            noisele1 = Buffer_f0[Trig_fine - Sprate:Trig_fine-1]
            noiseln1 = Buffer_f1[Trig_fine - Sprate:Trig_fine-1]
            noiselz1 = Buffer_f2[Trig_fine - Sprate:Trig_fine-1]
            noisele2 = Buffer_f3[Trig_fine - Sprate:Trig_fine-1]
            noiseln2 = Buffer_f4[Trig_fine - Sprate:Trig_fine-1]
            noiselz2 = Buffer_f5[Trig_fine - Sprate:Trig_fine-1]
        N_pnsr1 = np.zeros(shape=(3, 1))
        N_pnsr2= np.zeros(shape=(3, 1))

        N_pnsr1[0] = np.max(noisele1) - np.min(noisele1)
        N_pnsr1[1] = np.max(noiseln1) - np.min(noiseln1)
        N_pnsr1[2] = np.max(noiselz1) - np.min(noiselz1)

        N_pnsr2[0] = np.max(noisele2) - np.min(noisele2)
        N_pnsr2[1] = np.max(noiseln2) - np.min(noiseln2)
        N_pnsr2[2] = np.max(noiselz2) - np.min(noiselz2)

        persistent.N_pnsr1 = N_pnsr1
        persistent.N_pnsr2 = N_pnsr2

        return Sta_vars1, Sta_vars2, disAlarm, Alarm
    elif Triged == 1:  # 已判定为触发
        tempPakage1 = np.copy(Sta_vars1.BaseLine[:, -PackLen:])
        Sta_vars1.Traces_evt = np.hstack((Sta_vars1.Traces_evt, tempPakage1))  # 有问题需要改
        tempPakage2 = np.copy(Sta_vars2.BaseLine[:, -PackLen:])
        Sta_vars2.Traces_evt = np.hstack((Sta_vars2.Traces_evt, tempPakage2))  # 有问题需要改
        # print(np.shape(Sta_vars.Traces_evt)[1])
        [Pend1, Sta_vars1, afterPGA] = MyEnder(Sta_vars1, N_pnsr1)
        [Pend2, Sta_vars2, afterPGA] = MyEnder(Sta_vars2, N_pnsr2)
        if Pend1 >= 0 and Pend2>=0:
            # disAlarm = 1
            End_out = StartT  # 处理有问题
            Sta_vars1.Duration = End_out - Sta_vars1.P_time
            Sta_vars2.Duration = End_out - Sta_vars2.P_time
            Sta_vars1.End_time = End_out
            Sta_vars2.End_time = End_out
            strend='结束时间：'+str(End_out)
            logger.product(strend, 1, True)
            persistent.end_time_Pre=End_out
            persistent.P_time_Pre=Sta_vars1.P_time
            Sta_vars1.Traces_evt = np.empty(shape=(3, 0))
            Sta_vars2.Traces_evt = np.empty(shape=(3, 0))
            # if (Sta_vars1.Duration < MinDuration and Sta_vars1.EEW_times < MaxEEW_times) and (Sta_vars2.Duration < MinDuration and Sta_vars2.EEW_times < MaxEEW_times):
            #     if Debug == 1:
            #         print( "------------- 误报解除, 持续时间小于15秒，--------Duration=" + str(
            #             Sta_vars1.Duration))
            #     disAlarm = 2
            AlarmFlag = 0
            if Sta_vars1.AlarmFlag==1 and Sta_vars2.AlarmFlag==1:
                AlarmFlag=1

            if AlarmFlag==1 and Sta_vars1.Is_EQK==1 and Sta_vars2.Is_EQK==1:
                disAlarm=1
            else:
                disAlarm=2
            persistent.PGAold = -1
            persistent.AZIold = -1
            persistent.distold = -1
        return Sta_vars1,Sta_vars2,disAlarm, Alarm


##############################################################################下面为子函数
def MypickAIC(Sprate, PackLen, filter_Buffer,Buffer,endtimepre,startT):
    # 触发数据外围处理及判断
    Back = int(EEW_Params.Back)
    Fowrd = int(EEW_Params.Fowrd)
    thresh = float(EEW_Params.thresh)
    STW = float(EEW_Params.STW)
    LTW = float(EEW_Params.LTW)
    ifliter = int(EEW_Params.iflilter)
    MinThresh = float(EEW_Params.MinThresh)
    Bufffer_second=float(EEW_Params.Buffer_seconds)
    Trig_fine = -1
    Trigraw = -1
    L_fs = float(EEW_Params.L_fs)
    H_fs = float(EEW_Params.H_fs)    #，10,与MATLAB一致，避免finepick结果不一致
    len = int(np.size(filter_Buffer[0]))

    Buffer_len=int(Sprate*Bufffer_second)
    Buffer_len_now=int(np.shape(filter_Buffer[0])[0])
    Data1 = Buffer[-Buffer_len:][2]
    Data2 = Buffer[-Buffer_len:][5]
    DataF1=Buffer[2]-np.mean(Buffer[2])
    DataF2=Buffer[5]-np.mean(Buffer[5])
    LTW_ind = int(LTW * Sprate)
    try:
        max1=np.max(np.abs(Data1[LTW_ind:]))
    except:
        max1=0

    try:
        max3 = np.max(np.abs(Data1[20:LTW_ind]))
    except:
        max3=0

    try:
        max12 = np.max(np.abs(Data2[LTW_ind:]))
    except:
        max12=0

    try:
        max32 = np.max(np.abs(Data2[20:LTW_ind]))
    except:
        max32=0

    st=Buffer_len_now-Buffer_len
    if st<0:
        print("Buffer 没有积累满")
        return Trig_fine
    Buffer1=filter_Buffer[0][st:]-np.mean(filter_Buffer[0])
    Buffer2=filter_Buffer[1][st:]-np.mean(filter_Buffer[1])
    # Buffer1 = DataF1
    # Buffer2 = DataF2

    # MinThresh=0#调试用
    # if max1>MinThresh and max1>max3 and max12>MinThresh and max12>max32:
    #     Trigraw = MyPicker(Buffer1,Buffer2, Sprate, 1, thresh, STW, LTW, ifliter, PackLen)
    Trigraw = MyPicker(Buffer1, Buffer2, Sprate, 1, thresh, STW, LTW, ifliter, PackLen, endtimepre, startT)
    if Trigraw <= 0 and max1 > MinThresh and max12 > MinThresh:  # and max1 > max3 and max12 > max32
        ind1 = np.where(np.abs(Data1) == max1)
        ind2 = np.where(np.abs(Data2) == max12)
        # try:
        all_indices = np.concatenate([ind1[0], ind2[0]])
        if all_indices.size > 0:
            Trigraw = np.min(all_indices)
        else:
            Trigraw =-1
        # except:
        #     print('here')
    len_Buf = Buffer1.size
    if Trigraw > 0:
        Trigraw_T = startT + (PackLen + Trigraw - len_Buf) / Sprate

        if Trigraw_T <= endtimepre:
            Trigraw = -1
        else:
            Buffer1_t0 = startT + (PackLen - len_Buf) / Sprate  # buffer时间起点
            # 触发点以后的数据
            dataaf1 = Data1[Trigraw:]  # Buffer1[Trigraw:]
            dataaf2 = Data2[Trigraw:]  # Buffer2[Trigraw:]
            if endtimepre < Buffer1_t0:
                datapre1 = Data1[0:Trigraw]  # Buffer1[0:Trigraw - 1]
                datapre2 = Data2[0:Trigraw]  # Buffer2[0:Trigraw - 1]
            else:
                endtimepre_pos = int(len_Buf - (startT - endtimepre + PackLen / Sprate) * Sprate)
                datapre1 = Data1[endtimepre_pos:Trigraw]  # Buffer1[endtimepre_pos:Trigraw - 1]
                datapre2 = Data2[endtimepre_pos:Trigraw]  # Buffer2[endtimepre_pos:Trigraw - 1]
            maxaf = np.max([np.abs(dataaf1), np.abs(dataaf2)])
            maxpre = np.max([np.abs(datapre1), np.abs(datapre2)])
            if maxpre >= maxaf:
                Trigraw = -1
    if Trigraw == -1:
        return Trig_fine
    if ifliter > 0:
        DataF1 = MyFilter(Sprate, Data1, L_fs, H_fs)
        DataF2 = MyFilter(Sprate, Data2, L_fs, H_fs)
    PUpper=int(Trigraw-Back*Sprate)
    if PUpper<=1:
        PUpper = 1
    Plower = Trigraw + Fowrd * Sprate
    if len-Plower<=1*Sprate:
        Plower=len
    Trig_fine = FinePick(DataF1,DataF2, 1, PUpper, Plower, LTW * Sprate)
    m1 = np.max(abs(DataF1[0:Trig_fine-1]))
    mm1 = np.max(abs(DataF1[Trig_fine:]))
    m2 = np.max(abs(DataF2[0:Trig_fine - 1]))
    mm2 = np.max(abs(DataF2[Trig_fine:]))
    if m1>mm1 or m2>mm2 or  mm1 < MinThresh and mm2 < MinThresh:
        Trig_fine=-1
    # if Trig_fine>0:
    #     print('here')
    return Trig_fine


def MyPicker(acc1,acc2, sprate, phase, thresh, STW, LTW, iflilter, PackLen, endtimepre, startT):
    # phase:1(P波) 2(S波)
    acc111=np.copy(acc1)
    acc222 = np.copy(acc2)
    acc111=acc111[20:]
    acc222 =acc222[20:]
    len = np.size(acc111)
    difacc1 = np.diff(acc111)
    difacc2 = np.diff(acc222)
    # difacc=np.append(0,np.diff(acc))
    # difacc=np.hstack(([[0],difacc]))
    spanDeoddSecond = float(EEW_Params.spanDeOddSecond)
    L_fs = float(EEW_Params.L_fs)
    H_fs = float(EEW_Params.H_fs)
    trig = -1
    # difacc12 = DeOdd(difacc1, spanDeoddSecond * sprate)
    # difacc22 = DeOdd(difacc2, spanDeoddSecond * sprate)
    difacc12 =difacc1
    difacc22 =difacc2
    Facc1 = integrate.cumulative_trapezoid(difacc12, initial=0) # integrate.cumtrapz(difacc12, initial=0)
    Facc2 = integrate.cumulative_trapezoid(difacc22, initial=0)

    if iflilter > 0:
        Facc1 = MyFilter(sprate, Facc1, L_fs, H_fs)#H_fs
        Facc2 = MyFilter(sprate, Facc2, L_fs, H_fs)
    CF1 = (Facc1 + 0.01) ** 2
    CF2 = (Facc2 + 0.01) ** 2
    len = CF1.size
    # Lta1 = np.mean(CF1[:int(LTW * sprate+1)])
    # Lta2 = np.mean(CF2[:int(LTW * sprate+1)])

    # if phase == 1:
    #     Pbegin = int(LTW*sprate)
    #     tempnum =int(len-Pbegin-STW*sprate)
    #     for i in range(0,tempnum,20):
    #         Sta1 = np.mean(CF1[int(Pbegin+ i):int(Pbegin + i+STW*sprate)])
    #         Sta2 = np.mean(CF2[int(Pbegin+ i):int(Pbegin + i+STW*sprate)])
    #         # print(Sta1/Lta1 )
    #         if Sta1 / Lta1 > thresh and Sta2/Lta2>thresh:
    #             trig = i + Pbegin+20
    #             break
    #
    #     if trig == -1:
    #         # temp1 = Facc1[Pbegin:]
    #         # temp2 = Facc2[Pbegin:]
    #         # ind1 = np.where(temp1 > 10)  # 需要更改
    #         # ind2 = np.where(temp2 > 10)
    #         # if (np.size(ind1) == 0 and np.size(ind2)==0) or np.abs(temp1[ind1[0][0]]-temp2[ind2[0][0]])>1000:
    #         #     trig = -1
    #         #     return trig
    #         # else:
    #         #     if ind1[0][0]< ind2[0][0]:
    #         #         ind=ind1[0][0]
    #         #     else:
    #         #         ind = ind2[0][0]
    #         #
    #         #     trig = Pbegin + ind
    #             return trig
    #     else:
    #         return trig
    # # elif phase==2:
    # #      trig=MyPickerS(Facc,sprate)
    # #      return trig

    # 原则上每包进来只算一次
    step = int(0.2 * sprate)
    if PackLen > int(0.2 * sprate):
        n = math.ceil(PackLen / (0.2 * sprate))
        step = int(PackLen / n)
    # 结束点在当前buffer中的位置,点
    Trigt_end = int(len - (startT - endtimepre + PackLen / sprate) * sprate)
    if int(len - 5 * sprate) <= 0:
        return trig
    nbegin = int(len - 5 * sprate - (PackLen))  # 循环开始位置
    if nbegin <= 0:
        nbegin = 1
    for i in range(nbegin, int(len - 5 * sprate), step):
        LTA1 = np.mean(CF1[i + 1:int(i + 5 * sprate)])
        LTA2 = np.mean(CF2[i + 1:int(i + 5 * sprate)])
        STA1 = np.mean(CF1[int(i + 5 * sprate - 0.2 * sprate + 1):int(i + 5 * sprate)])  # STW：0.4，配置参数获取
        STA2 = np.mean(CF2[int(i + 5 * sprate - 0.2 * sprate + 1):int(i + 5 * sprate)])  # STW：0.4，配置参数获取
        minSTALTA = min([STA1 / LTA1, STA2 / LTA2])
        if (round(STA1 / LTA1) >= thresh) and (round(STA2 / LTA2) >= thresh):  # 20
            # #If there is interference + earthquake in a data window at the same time,
            # #please  eliminate the interference(there is interference before earthquake)
            if Trigt_end > 0:
                Trigt_p = int(i + 5 * sprate - 0.2 * sprate + 1)
                # #触发点与Endtime相差不过3秒，跳过此段
                if Trigt_p < Trigt_end:  # | | Trigt_p - Trigt_end <= sprate * 1
                    continue

            trig = int(i + 5 * sprate - 0.2 * sprate + 1)
            # probty_sta = (STA1 / LTA1 + STA2 / LTA2) / 2 / thresh
            break

    if trig == -1:  # 没有触发
        Pbegin = int(len - 0.2 * sprate)  # 短窗起点，从长窗之后开始
        ind = np.where(Facc1[Pbegin:] > 10)[0]  # find(Facc1(Pbegin:end) >= 10)
        ind2 = np.where(Facc2[Pbegin:] > 10)[0]  # find(Facc2(Pbegin:end) >= 10 )  # 20240321
        if  ind.size > 0 and ind2.size > 0:  # 查看加速度滤波后是否已经超过10gal了,max(abs(Facc1[Pbegin:]))>max(abs(Facc1[0:Pbegin])) and max(abs(Facc2[Pbegin:]))>max(abs(Facc2[0:Pbegin])) and
            indmin = min(ind[0], ind2[0])
            trig = Pbegin + indmin
    # if trig>0:
    #     print('here')
    return trig
def DeOdd(data, span):
    M = 5
    NonOdd = data
    num = int(math.floor(int(np.size(data)) / span))
    for k in range(num):
        WaveSpan = NonOdd[int(k * span):int((k + 1) * span)]
        V1 = np.max(np.abs(WaveSpan))
        T1 = np.argmax(np.abs(WaveSpan))
        WaveSpan[T1] = 0
        V2 = np.max(np.abs(WaveSpan))
        T2 = np.argmax(np.abs(WaveSpan))
        WaveSpan[T2] = 0
        if V1 > M * (np.max(np.abs(WaveSpan))):
            NonOdd[int(k * span + T1)] = -np.sum(NonOdd[int(k * span + T1 - 1)])
        if V2 > M * (np.max(np.abs(WaveSpan))):
            NonOdd[int(k * span + T2)] = -np.sum(NonOdd[int(k * span + T2 - 1)])
    return NonOdd


def MyPickerS(trace_input, sprate, nature):
    #  N    = 40       # Order
    # Fc   = 10       # Cutoff Frequency
    # flag = 'scale'  # Sampling Flag
    # blackmanharris(N+1)
    # S波拾取 fir 带窗fir 滤波
    # S波拾取模块
    length = np.shape(trace_input)[1]
    if length>=20*sprate:
        n = int(length / 2)
        stepf = (length - 1) // (n - 1)
        indices = np.arange(0, length, stepf)
        trace = trace_input[:, indices]
    else:
        trace = np.copy(trace_input)
    len=np.shape(trace)[1]
    row1 = np.shape(trace)[0]  # 6
    trigS = -1  # output
    delt_S = 0
    f1 = sprate
    if len <= 4 * sprate:
        return trigS
    if row1 < 6:
        print("MyPickerS数据有" + str(row1) + "列")
        return trigS
    if len > 5 * f1:  # 方差与均值的比值，可更细化一些
        Vmoldtresh = 20
    else:
        Vmoldtresh = 10

    stepn = f1 / 2

    Pdegree1 = [-1]
    Pdegree2 = [-1]
    # np.empty((2, 2), dtype=float)
    Pdegree_outall = np.empty((math.ceil(len / stepn) - 1, 4), dtype=float)  # zeros(math.ceil(len/200) - 1, 4)
    SN1 = np.empty((math.ceil(len / 200) - 1, 6), dtype=float)
    SN2 = np.empty((math.ceil(len / stepn) - 1, 6), dtype=float)
    out_energy_HV = np.empty((math.ceil(len / stepn) - 1, 8), dtype=float)
    e_norm = np.empty((math.ceil(len / 200) - 1, 8), dtype=float)

    L_fs = 0.1
    H_fs = 20
    tr0 = MyFilter(f1, trace[0], L_fs, H_fs)
    tr1 = MyFilter(f1, trace[1], L_fs, H_fs)
    tr2 = MyFilter(f1, trace[2], L_fs, H_fs)
    tr3 = MyFilter(f1, trace[3], L_fs, H_fs)
    tr4 = MyFilter(f1, trace[4], L_fs, H_fs)
    tr5 = MyFilter(f1, trace[5], L_fs, H_fs)

    tracefil = np.vstack([tr0, tr1, tr2, tr3, tr4, tr5])  # 6*n

    # 计算噪声段的PNSR
    S_pnsr = [[], [], [], [], [], []]
    S_pnsr[0] = max(tr0[0: f1]) - min(tr0[0: f1])
    S_pnsr[1] = max(tr1[0: f1]) - min(tr1[0: f1])
    S_pnsr[2] = max(tr2[0: f1]) - min(tr2[0: f1])
    S_pnsr[3] = max(tr3[0: f1]) - min(tr3[0: f1])
    S_pnsr[4] = max(tr4[0: f1]) - min(tr4[0: f1])
    S_pnsr[5] = max(tr5[0: f1]) - min(tr5[0: f1])

    n1 = math.ceil(len/stepn) - 1
    for j in range(1, n1 + 1):
        d1 = tracefil[:, 0:int(j * stepn)]
        d2 = tracefil[:, int(j * stepn):]
        # 偏振法子
        # Pdegree介于0 - 1之间, 地震波趋于1，噪声趋于0.
        [Pdegree1, Pdegree2] = polarization(d1)  # 前
        [Pdegree3, Pdegree4] = polarization(d2)  # 后
        Pdegree_outall[j - 1, :] = [Pdegree1, Pdegree2, Pdegree3, Pdegree4]

        # 信噪比
        stepn1 = int(f1)
        if j < math.ceil(len / stepn1) - 1:  # SN1 length error!!!
            sd1 = tracefil[:, int((j - 1) * stepn1):int(j * stepn1)]
            if j + 1 > math.ceil(len / stepn1) - 1:
                sd2 = tracefil[int(j * stepn1):-1, :]
            else:
                sd2 = tracefil[:, int(j * stepn1):int((j + 1) * stepn1)]
            # p1 = statistics.mean(sd1 * sd1)
            a1_6n = sd1 * sd1
            p1 = a1_6n.mean(axis=1)  # line
            # s1 = statistics.mean(sd2 * sd2)
            a2_6n = sd2 * sd2
            s1 = a2_6n.mean(axis=1)  # line
            SN1[j - 1, :] = 20 * np.log10(s1 / (p1 + 1e-6))  # half SN2?
        p2 = (d1 * d1).mean(axis=1)
        s2 = (d2 * d2).mean(axis=1)
        SN2[j - 1, :] = 20 * np.log10(s2 / (p2 + 1e-6))

        # H/V比, 事件数据的水平纵向比, 分段
        # Sta_vars.Buffer( - lennew: ,:)
        energy_HV = [0, 0, 0, 0]
        energy_e1 = sum(abs(d1[0, :]) ** 2)
        energy_n1 = sum(abs(d1[1, :]) ** 2)
        energy_z1 = sum(abs(d1[2, :]) ** 2)
        energy_e2 = sum(abs(d1[3, :]) ** 2)
        energy_n2 = sum(abs(d1[4, :]) ** 2)
        energy_z2 = sum(abs(d1[5, :]) ** 2)
        energy_HV1 = [energy_e1 / energy_z1, energy_n1 / energy_z1, energy_e2 / energy_z2, energy_n2 / energy_z2]
        energy_e1_s = sum(abs(d2[0, :]) ** 2)
        energy_n1_s = sum(abs(d2[1, :]) ** 2)
        energy_z1_s = sum(abs(d2[2, :]) ** 2)
        energy_e2_s = sum(abs(d2[3, :]) ** 2)
        energy_n2_s = sum(abs(d2[4, :]) ** 2)
        energy_z2_s = sum(abs(d2[5, :]) ** 2)
        # 取out_energy_HV 1 - 4 特征
        energy_HV1_s = [energy_e1_s / energy_z1_s, energy_n1_s / energy_z1_s, energy_e2_s / energy_z2_s,
                        energy_n2_s / energy_z2_s]
        out_energy_HV[j - 1, :] = energy_HV1 + energy_HV1_s  # n*8
        # e_norm[j,:] = [norm(d1[:, 1])/norm(d1[:, 3]), norm(d1[:, 2])/norm(d1[:, 3]),
        # norm(d1[:, 4])/norm(d1[:, 6]), norm(d1[:, 5])/norm(d1[:, 6]), ...
        # norm(d2[:, 1])/norm(d2[:, 3]), norm(d2[:, 2])/norm(d2[:, 3]), ...
        # norm(d2[:, 4])/norm(d2[:, 6]), norm(d2[:, 5[)/norm(d2[:, 6])]

    # 分段离散度np.empty((math.ceil(len/stepn) - 1, 6))
    varall = np.empty((math.ceil(len / stepn), 6), dtype=float)
    for j in range(1, 1 + math.ceil(len / stepn)):
        if j == math.ceil(len / stepn):
            vardata = tracefil[:, int((j - 1) * stepn):]
        else:
            vardata = tracefil[:, int((j - 1) * stepn):int(j * stepn)]
            # varall[j-1,:] = statistics.variance(vardata) #vardata:list
            varall[j - 1, :] = vardata.var(axis=1)
    ## 2, findpeaks这块待优化
    locp12 = []
    locp1 = []
    locp2 = []
    locpall = []
    y1 = -1 * Pdegree_outall[:, 0]
    # [pksp1, locp1] = findpeaks(y1)  # 列
    locp1, _ = find_peaks(y1)
    pksp1 = y1[locp1]
    if locp1.any():  # ~isempty(locp1):
        # pksp1 = max(pksp1)
        p1 = np.where(pksp1 == max(pksp1))
        locpall = locp1[int(p1[0])]
    y2 = -1 * Pdegree_outall[:, 1]
    locp2, _ = find_peaks(y2)
    pksp2 = y2[locp2]
    if locp2.any():
        # pksp2 = max(pksp2)
        p2 = np.where(pksp2 == max(pksp2))
        locpall = np.append(locpall, locp2[int(p2[0])])  # locpall+locp2[p2]
    y3 = -1 * Pdegree_outall[:, 2]
    locp3, _ = find_peaks(y3)
    pksp3 = y3[locp3]
    if locp3.any():
        # pksp3 = max(pksp3)
        p3 = np.where(pksp3 == max(pksp3))
        locpall = np.append(locpall, locp3[int(p3[0])])  # locpall+locp3[p3]
    y4 = -1 * Pdegree_outall[:, 3]
    locp4, _ = find_peaks(y4)
    pksp4 = y4[locp4]
    if locp4.any():
        # pksp4 = max(pksp4)
        p4 = np.where(pksp4 == max(pksp4))
        locpall = np.append(locpall, locp4[p4[0]])  # (locpall,locp4[p4])

    ##Pdegree_outall：1-2列呈下降趋势，波谷为可能S信号；3-4呈缓慢上升，前平稳后波动大，适合长短时orAIC。
    pos11 = []
    pos22 = []
    pos33 = []
    pos44 = []
    pdgree_diff = Pdegree_outall[1:, :] - Pdegree_outall[0:-1, :]  # diff(Pdegree_outall)
    # [va1, pos1] = findpeaks(pdgree_diff[:, 1])
    y1 = pdgree_diff[:, 0]
    pos1, _ = find_peaks(y1)
    va1 = y1[pos1]
    if va1.any():
        # maxv1 = max(va1)
        maxpos1 = np.where(va1 == max(va1))
        pos11 = pos1[maxpos1] + 1
    y2 = pdgree_diff[:, 1]
    pos2, _ = find_peaks(y2)
    va2 = y2[pos2]
    if va2.any():
        # maxv2 = max(va2)
        maxpos2 = np.where(va2 == max(va2))
        pos22 = pos2[maxpos2] + 1
    y3 = pdgree_diff[:, 2]
    pos3, _ = find_peaks(y3)
    va3 = y3[pos3]
    if va3.any():
        # maxv3 = max(va3)
        maxpos3 = np.where(va3 == max(va3))
        pos33 = pos3[maxpos3] + 1
    y4 = pdgree_diff[:, 3]
    pos4, _ = find_peaks(y4)
    va4 = y4[pos4]
    if va4.any():
        # maxv4 = max(va4)
        maxpos4 = np.where(va4 == max(va4))
        pos44 = pos4[maxpos4] + 1

    # pdpeaksall =[pos11,pos22,pos33,pos44] #未使用

    # 20230901new add
    pospd2 = []
    x = np.arange(0, np.size(Pdegree_outall, 0), 1)
    # x1 = 1:0.05:len(Pdegree_outall, 1)  # 翻倍
    x1 = np.arange(0, np.size(Pdegree_outall, 0), 0.05)
    # fun1 = interp1d(x, Pdegree_outall[:, 1], kind = 'linear')  # interp1(x, Pdegree_outall[:, 1), x1, 'linear')
    y1 = np.interp(x1, x, Pdegree_outall[:, 0])
    # fun2 = interp1d(x, Pdegree_outall[:, 2], kind = 'linear')  # interp1(x, Pdegree_outall[:, 1], x1, 'linear')
    y2 = np.interp(x1, x, Pdegree_outall[:, 1])
    # fun3 = interp1d(x, Pdegree_outall[:, 3], kind='linear')  # interp1(x, Pdegree_outall[:, 2], x1, 'linear')
    y3 = np.interp(x1, x, Pdegree_outall[:, 2])
    # fun4 = interp1d(x, Pdegree_outall[:, 4], kind='linear')  # interp1(x, Pdegree_outall[:, 3], x1, 'linear')
    y4 = np.interp(x1, x, Pdegree_outall[:, 3])
    yy = [y1, y2, y3, y4]
    delt13 = abs(y1 - y3)
    delt24 = abs(y2 - y4)
    # pos13 = find(delt13 < 0.02)
    pos13 = [index for index, value in enumerate(delt13) if value < 0.02]
    # pos24 = find(delt24 < 0.02)
    pos24 = [index for index, value in enumerate(delt24) if value < 0.02]
    if pos13:
        pos13 = pos13[-1]
        pospd2 = pospd2 + [round(x1[pos13])]
    # min(delt13)
    if pos24:
        pos24 = pos24[-1]
        pospd2 = pospd2 + [round(x1[pos24])]
    # min(delt24)
    # 找次小
    delt13[pos13] = 1  # [m11, pos13_2] = min(delt13)
    delt24[pos24] = 1  # [m22, pos24_2] = min(delt24)
    # pos13_2 = find(delt13 < 0.02)
    pos13_2 = [index for index, value in enumerate(delt13) if value < 0.02]
    # pos24_2 = find(delt24 < 0.02)
    pos24_2 = [index for index, value in enumerate(delt24) if value < 0.02]
    if pos13_2:
        pos13_2 = pos13_2[-1]
        pospd2 = pospd2 + [round(x1[pos13_2])]
    # min(delt13)
    if pos24_2:
        pos24_2 = pos24_2[-1]
        pospd2 = pospd2 + [round(x1[pos24_2])]
    # min(delt24)

    pd1234 = []
    if np.size(Pdegree_outall, 1) >= 20:
        [pd1234, pddelt_AICout1234] = TOC_AIC2(Pdegree_outall, 10)  # 短了不适用，改

    pd1234 = [number for number in pd1234 if number > 0]  # pd1234(pd1234 > 0)
    pdall = list(locpall) + pd1234 + pospd2 + pospd2
    pdall = [number for number in pdall if number > 0]  # pdall(pdall > 0)
    pdall = sorted(pdall)  # 最多8个

    # 4个的方差最小则认为是
    # if size(traces, 1) <= 3 * sprate # 20230907
    [vmold1, trigS1] = minvar(pdall, 3, stepn)
    # else
    # [vmold1, trigS1]=minvar(pdall, 4,stepn)

    if trigS1 <= 5 * sprate and vmold1 > 1 or trigS1 > 5 * sprate and vmold1 > 5:
        trigS1 = -1

    ###横纵幅值比
    trigS3 = -1
    peaks_HV = []
    out_energy_HV1234 = out_energy_HV[:, 1: 4]
    ##
    pos3 = []
    for n in range(8):
        peaks_HV = allpicks(out_energy_HV[:, n], 3)  # 正负峰位置，限制半峰宽度
        if np.size(peaks_HV) > 0:
            peak3 = np.zeros((np.size(peaks_HV), 1))
            for i in range(np.size(peaks_HV, 0)):
                data3 = out_energy_HV[int(peaks_HV[i] - 1):int(peaks_HV[i] + 1), n]
                peak3[i, :] = np.var(data3) / np.mean(data3)
            # delt1 = abs(out_energy_HV(peaks_HV(i) - 1, 1) - out_energy_HV(peaks_HV(i), 1))
            # delt2 = abs(out_energy_HV(peaks_HV(i) + 1, 1) - out_energy_HV(peaks_HV(i), 1))
            # demtMax(i) = max([delt1, delt2]) #同var/mean效果
            pos3ls = [index for index, value in enumerate(peak3) if value > 0.1 * max(peak3)]
            pos3 = pos3 + pos3ls  # hang

        ##AIC
    [pos1234, delt_AICout1234] = TOC_AIC2(out_energy_HV1234, 5)
    [pos5678, delt_AICout5678] = TOC_AIC2(out_energy_HV[:, 5: 8], 5)
    pos1234 = pos1234.flatten()
    pos1234 = [x - 1 for x in pos1234]
    pos1234 = [value for value in pos1234 if value > 0]
    # pos5678 = pos5678(pos5678 > 0)
    pos5678 = pos5678.flatten()
    pos5678 = [x - 1 for x in pos5678]
    pos5678 = [value for value in pos5678 if value > 0]

    pos12345678 = pos1234 + pos5678
    if pos12345678:
        # row:value,times,fre
        tABLE = count_and_sort_numbers(np.array(pos12345678))
        tABLE = np.array(tABLE).reshape(-1, 2)
        t2 = tABLE[:, 1]
        # [m3, pm3] = max()
        m3 = max(t2)  # 次数相同时取第一个最大？
        pm3 = np.where(t2 == max(t2))
        if ~(m3 > 3 and tABLE[pm3, 1] / sum(tABLE[:, 1]) > 0.30):
            pos12345678new = pos12345678
        else:
            # pos12345678new = repmat(tABLE(pm3, 1), m3, 1)  # 如果有两个值次数相同，当前方法取得较小值
            pos12345678new = tABLE[pm3, 0] * np.ones((int(m3), 1))
            [vmold3, trigS3] = minvar(pos12345678new, 3, stepn)
            if vmold3 > Vmoldtresh:
                trigS3 = -1

    ###分段离散度
    trigS4 = -1
    # step=0.5 * sprate
    ##需要重新调整, 不能取最大值，要取起跳点
    pos4all = []
    # [a1, pos1] = findpeaks(varall[:, 1))  # max(varall[:, 1)) #=
    # [a2, pos2] = findpeaks(varall[:, 2))  # =max(varall[:, 2)) #= findpeaks(-1 * varall[:, 2))
    # [a4, pos4] = findpeaks(varall[:, 4))  # =max(varall[:, 4)) #= findpeaks(-1 * varall[:, 4))
    # [a5, pos5] = findpeaks(varall[:, 5))  # =max(varall[:, 5)) #= findpeaks(-1 * varall[:, 5))
    y1 = varall[:, 0]
    pos1, _ = find_peaks(y1)
    a1 = y1[pos1]
    y2 = varall[:, 1]
    pos2, _ = find_peaks(y2)
    a2 = y2[pos2]
    y4 = varall[:, 3]
    pos4, _ = find_peaks(y4)
    a4 = y4[pos4]
    y5 = varall[:, 4]
    pos5, _ = find_peaks(y5)
    a5 = y5[pos5]
    # [a6, pos6] = findpeaks(-1 * varall[:, 6))； # =max(varall[:, 6)) #= findpeaks(-1 * varall[:, 6))
    pos4all = np.concatenate((pos1, pos2, pos4, pos5))  # [pos1,pos2,pos4,pos5]  # pos4all = pos4all - 1
    aictresh = 10  # 10
    [posVar, delt_AICout] = TOC_AIC2(varall[:, 0:2] + varall[:, 3:5], 10)
    posVar = posVar.flatten()  # 1dir
    if np.size(posVar, 0) >= 4 and np.var(posVar) <= 0.5:
        p4 = posVar
    else:
        p4 = np.concatenate((pos4all, posVar))
    p4 = [element for element in p4 if element > 0]  # p4 = p4(p4>0)
    [vmold4, trigS4] = minvar(p4, 3, stepn)
    trigS4 = trigS4 - stepn
    # [vmold4, trigS4]=minvar(p4, 2,stepnn)
    # trigS4=trigS4 # -stepn
    if vmold4 > Vmoldtresh:
        trigS4 = -1

    # 重算起点
    pstartnew = 1  # new start　对CYRLATST_20221215071823duiqi_txt6short不友好
    if nature == 0:  # size(traces, 1) >= 5 * sprate # 20230908
        pstartnew = round(0.3 * np.size(trace, 1))
    else:
        pstartnew = round(0.2 * np.size(trace, 1))
    tracesnew = trace[:, pstartnew - 1:-1]  # 双刃剑，远震可去除前面低噪声段，近震直接去掉了P波段。

    # 保留6，去掉7，提高效率？
    ############方法6：加权CF + AIC方法
    ############ 方法7：CF + AIC方法                                   ## traces滤波，输出为traces1
    trigS6 = -1
    trigS7 = -1
    tracesnew = tracesnew.T
    trace = trace.T  # 该开头转置的
    tracesS = tracesnew
    tracesS[:, 0] = FilterS(f1, tracesnew[:, 0])
    tracesS[:, 1] = FilterS(f1, tracesnew[:, 1])
    tracesS[:, 2] = FilterS(f1, tracesnew[:, 2])
    tracesS[:, 3] = FilterS(f1, tracesnew[:, 3])
    tracesS[:, 4] = FilterS(f1, tracesnew[:, 4])
    tracesS[:, 5] = FilterS(f1, tracesnew[:, 5])

    EW1 = tracesS[:, 0]
    NS1 = tracesS[:, 1]
    UD1 = tracesS[:, 2]
    EW2 = tracesS[:, 3]
    NS2 = tracesS[:, 4]
    UD2 = tracesS[:, 5]

    #### CF(i) = Y(i) ^ 2 + w * (Y(i) - Y(i - 1)) ^ 2, 一般w = 3, w = 1/(len - 1) * sum((Yi - meanY) ^ 4)/(
    # 1 / (len - 1) * sum((Yi - meanY) ^ 2)) ^ 2
    steps = 0.5 * sprate  # 0.1太短
    stepL = 3 * sprate  # new problem, 长窗成为盲区，近震漏掉S
    CF1e = EW1[2:-1] ** 2 + 3 * (EW1[2:-1] - EW1[1:-2]) ** 2
    CF1n = NS1[2:-1] ** 2 + 3 * (NS1[2:-1] - NS1[1:-2]) ** 2
    CF2e = EW2[2:-1] ** 2 + 3 * (EW2[2:-1] - EW2[1:-2]) ** 2
    CF2n = NS2[2:-1] ** 2 + 3 * (NS2[2:-1] - NS2[1:-2]) ** 2
    # xx = np.arange(stepL,np.size(CF1e, 0)-steps+0.05*sprate,int(0.05*sprate))
    # lenr = np.size(xx,0)
    CF1234 = np.vstack((CF1e, CF1n, CF2e, CF2n))  # CF1234 = [CF1e, CF1n, CF2e, CF2n]
    CF1234 = CF1234.T
    if np.size(tracesnew, 0) > 1 * sprate:
        # 方法7：CF + AIC
        ## 对于近震来说，delt_AIC低于固定AIC阈值600，可考虑按特征长度分类处理
        rigS7 = -1
    ## CF(i) = Y(i) ^ 2 - Y(i - 1) * Y(i + 1)
    CF_EW1 = EW1[2: -1] ** 2 - EW1[1: -2] * EW1[3:]  # CF(i) = Y(i) ^ 2 - Y(i - 1) * Y(i + 1)
    CF_NS1 = NS1[2: -1] ** 2 - NS1[1: -2] * NS1[3:]
    CF_EW2 = EW2[2: -1] ** 2 - EW2[1: -2] * EW2[3:]
    CF_NS2 = NS2[2: -1] ** 2 - NS2[1: -2] * NS2[3:]
    CFold1234 = [CF_EW1, CF_NS1, CF_EW2, CF_NS2]
    CFold1234 = np.vstack((CF_EW1, CF_NS1, CF_EW2, CF_NS2))  # CF1234 = [CF1e, CF1n, CF2e, CF2n]
    CFold1234 = CFold1234.T
    if np.size(CF_EW1, 0) <= 5 * sprate:  # 应线性
        if nature == 0:
            trs = 500
        else:
            trs = 300
    else:
        trs = 600
    datain = np.hstack((CFold1234, CF1234))
    [TrigSout, delt_AICout, rallout] = TOC_AIC0(datain, trs, nature)
    TrigSout = TrigSout.T
    delt_AICout = delt_AICout.T
    rallout = rallout.T
    # trig7
    trigENall70 = TrigSout[0:4, :]
    delt_AIC70 = delt_AICout[0:4, :]
    r70 = rallout[0:4, :]

    # pos7 = find(trigENall70 > 0)
    pos7 = [index for index, value in enumerate(trigENall70) if value > 0]  # 找出>0的值的位置
    if not pos7:
        trigENall7 = []
        delt_AIC = []
    else:
        trigENall7 = trigENall70[pos7]
        delt_AIC = delt_AIC70[pos7]  # 同步Trigenall, delt_AIC(delt_AIC >= trs)不可

    if nature == 0:
        if trigENall7.__len__() > 2:
            deltminr7 = (max(trigENall7) - min(trigENall7)) / min(trigENall7)
            if deltminr7 < 0.4:
                trigS7 = pstartnew + np.mean(trigENall7)
    else:  # nature == 1
        if trigENall7.__len__() > 2:
            arr = trigENall7 + delt_AIC
            rows = len(arr)  # 获取数组的行数和列数
            cols = len(arr[0])
            # 对每列独立排序
            sorted_arr = [[row[col] for row in arr] for col in range(cols)]
            # 将排序后的数据转换为二维数组
            trigENall_delt_AIC7 = [[row[i] for i in range(cols)] for row in sorted_arr]
            deltminr7 = abs(trigENall_delt_AIC7[-2, 1] - trigENall_delt_AIC7[-1, 1]) / min(
                trigENall_delt_AIC7[-2:, 1])
            if deltminr7 < 0.5:
                trigS7 = pstartnew + np.mean(trigENall_delt_AIC7[-1:, 1])

    # 方法8： 方法6的CF + 方法7的AIC：newCF + AIC,
    ##对于近震来说，delt_AIC低于固定AIC阈值600，可考虑按特征长度分类处理
    trigS8 = -1  # CF1234 = [CF1e, CF1n, CF2e, CF2n]
    # trs = 700
    if np.size(CF1234[:, 1], 0) <= 5 * sprate:
        if nature == 0:
            trs = 500
        else:
            trs = 300
    else:
        trs = 600

    TrigS8all = TrigSout[4:8, :]
    delt_AIC8 = delt_AICout[4:8, :]
    raic = rallout[4:8, :]

    # result = [(value, index) for index, value in enumerate(TrigS8all) if value >0]
    pos8 = [index for index, value in enumerate(TrigS8all) if value > 0]  # 找出>0的值的位置
    if not pos8:
        TrigS8all = []
        delt_AIC8 = []
    else:
        TrigS8all = TrigS8all[pos8]
        delt_AIC8 = delt_AIC8[pos8]  # 同步Trigenall, delt_AIC(delt_AIC >= trs)不可

    ##模拟数据双传感器一样，需要从严
    if nature == 0:  # nature
        if TrigS8all.__len__()>2:
            deltminr = (max(TrigS8all) - min(TrigS8all)) / min(TrigS8all)
            if deltminr < 0.4:
                trigS8 = pstartnew + np.mean(TrigS8all)
    else:
        if TrigS8all.__len__()>2:
            arr = TrigS8all + delt_AIC8
            rows = len(arr)  # 获取数组的行数和列数
            cols = len(arr[0])
            # 对每列独立排序
            sorted_arr = [[row[col] for row in arr] for col in range(cols)]
            # 将排序后的数据转换为二维数组
            TrigS8all_delt_AIC8 = [[row[i] for i in range(cols)] for row in sorted_arr]
            deltminr = abs(TrigS8all_delt_AIC8[-1, 1] - TrigS8all_delt_AIC8[-1, 1]) / min(
                TrigS8all_delt_AIC8[-1:, 1])
            if deltminr < 0.5:
                trigS8 = pstartnew + np.mean(TrigS8all_delt_AIC8[-1:, 1])

    ### 汇总，trigS6与triger8类似，建议删掉一个
    S6 = [trigS1, trigS8, trigS3, trigS4, trigS6, trigS7]
    S5 = [trigS1, trigS8, trigS3, trigS4, trigS7]
    # S5 = [trigS1, trigS8, trigS3, trigS4, trigS8] #S8加权
    S3 = [trigS4, trigS6, trigS7]

    S5 = [element for element in S5 if element > 0]
    S5 = sorted(S5)  # up
    lenS5 = np.size(S5, 0)  #
    if lenS5 - 3 >= 0:
        vv1 = 1000 * np.ones((lenS5 - 2))
        vv2 = np.ones((lenS5 - 2))
    nout = 3  # 有3个值就判断输出，4？
    minvv1 = 1000

    if trigS <= 0 and lenS5 >= nout and (trigS8 > 0 or trigS7 > 0):
        for n in range(lenS5 - 3 + 1):
            vv1[n] = np.var(S5[n:n + nout]) / np.mean(S5[n: n + nout])
            vv2[n] = (max(S5[n:n + nout]) - min(S5[n: n + nout])) / min(S5[n: n + nout])

        minvv1 = min(vv1)
        posvv=np.where(vv1==minvv1)
        # [minvv1, vv2(posvv, 1)]
        if vv2[posvv] < 0.25:  # 0.5
            if np.mean(S5[int(posvv[0]): int(posvv[0]) + nout]) <= 5 * sprate and minvv1 < 10 or np.mean(\
                    S5[int(posvv[0]): int(posvv[0]) + nout]) > 5 * sprate and minvv1 < 20:  # 5秒内的5太小改10
                trigS = round(np.mean(S5[int(posvv[0]):int(posvv[0])+nout]))
            if nature == 0:
                rs = 1.3
            else:
                rs = 1
            if trigS > 0:
                # after
                tracenew0 = trace[1:trigS - 1, :]
                tracenew1 = trace[trigS:, :]
                ab = [max(abs(tracenew1[:, 1])) / max(abs(tracenew1[:, 1])), max(abs(tracenew1[:, 2])) / max(
                    abs(tracenew0[:, 2])), max(abs(tracenew1[:, 4])) / max(abs(tracenew0[:, 4])), max(
                    abs(tracenew1[:, 5])) / max(abs(tracenew0[:, 5]))]
                if max(ab) < 1:
                    trigS = -1

                energy_e1 = sum(abs(tracenew1[:, 0]))
                energy_n1 = sum(abs(tracenew1[:, 1]))
                energy_z1 = sum(abs(tracenew1[:, 2]))
                energy_e2 = sum(abs(tracenew1[:, 3]))
                energy_n2 = sum(abs(tracenew1[:, 4]))
                energy_z2 = sum(abs(tracenew1[:, 5]))
                energy_HV = [energy_e1 / energy_z1, energy_n1 / energy_z1, energy_e2 / energy_z2,
                             energy_n2 / energy_z2]  # > 1

                a2 = abs(tracenew1)  # mean(a2)
                a1 = abs(trace[1:trigS, :])  # mean(a1)
                a = np.mean(a2, axis=0) / np.mean(a1, axis=0)  # 1, 2, 4, 5比3/6大才对, 反例：eg: 93
                ah = [a[0], a[1], a[3], a[4]]
                av = [a[2], a[5]]

    if trigS > 0 and round((trigS8 + trigS7) / 2) > trigS:  # trigS <= 0 and trigS8 > 0 and trigS7 > 0 or
        if abs(trigS8 - trigS7) / min([trigS8, trigS7]) < 0.2:  # 0.25dale
            trigS = round((trigS8 + trigS7) / 2)
    elif trigS > 0 and max([trigS8, trigS7]) < trigS / 2:  # 差异太大不要
        trigS = -1

    arr1 = trace[:, 0] ** 2 + trace[:, 1] ** 2 + trace[:, 2] ** 2
    squares1 = [math.sqrt(num) for num in arr1]
    pga1 = max(squares1)
    pos1pga = np.where(arr1 == max(arr1))
    arr2 = trace[:, 3] ** 2 + trace[:, 4] ** 2 + trace[:, 5] ** 2
    squares2 = [math.sqrt(num) for num in arr2]
    pga2 = max(squares2)
    pos2pga = np.where(arr2 == max(arr2))
    maxpgapos = max([pos1pga, pos2pga])

    if trigS > 0 and trigS < sprate * 1.5 and maxpgapos - trigS > sprate * 2:  # trigS < len/3 and maxpgapos > len - sprate
        # dispall(['S波识别太近，请检查！！！' num2str(trigS)], Debug)
        trigS = -1

    sall = [trigS1, trigS8, trigS3, trigS4, trigS6, trigS7, len, minvv1, trigS]  # 临时使用, 9
    if trigS > 0:
        if len!=length:
            trigS=stepf*trigS
        str11 = 'S波到了：' + str(trigS)
        logger.product(str11, 1, True)
    return trigS


def count_and_sort_numbers(arr):
    # 使用Counter统计每个数字出现的次数
    count = Counter(arr)
    # 排序,频次从高到低
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_count


def minvar(posSN, NN, stp):
    #  posSN:单行or单列待处理数据
    #  N：最小计算尺度
    #  stp：当前步长

    vmold = -1
    vmnew = -1
    trigSv = -1
    len = np.size(posSN, 0)
    posSN = sorted(posSN)  # 勿倒序
    if len < NN:
        return vmold, trigSv
    if len >= NN:
        for n in range(NN - 1, len):  # NN:len
            array = posSN[n - NN + 1:n + 1]
            arr1 = [element * stp for element in array]
            vmnew = np.var(arr1) / (stp * np.mean(posSN[n - NN + 1:n + 1]))
            if n == NN:
                vmold = vmnew
                trigSv = np.mean(posSN[n - NN + 1:n + 1]) * stp
            else:
                if vmnew <= vmold:
                    vmold = vmnew
                    trigSv = np.mean(posSN[n - NN + 1:n + 1]) * stp
    return vmold, trigSv


def TOC_AIC2(Data, mintresh):
    # Data: 单方向数据，多列，每列单独出结果哦。20230614.
    # TrigS ；S波在当前数据中的点位

    rows = len(Data[0])  # size(Data, 2) 列
    N = len(Data) - 1
    # mintresh = 600
    TrigSout = -1 * np.ones((1, rows))
    delt_AICout = -1 * np.ones((1, rows))
    delt_AIC = -1 * np.ones((1, rows))

    AICs1 = np.zeros(N)
    ##row1
    Data_m2 = Data[:, 0] * Data[:, 0]
    Data_m1 = Data[:, 0]
    total_m1 = sum(Data_m1[0:N + 1])  # AIC段全部数据和
    total_m2 = sum(Data_m2[0:N + 1])  # AIC段全部数据平方和
    temp_m1 = Data_m1[0]
    temp_m2 = Data_m2[0]

    TrigS1 = -1
    TrigS2 = -1
    TrigS3 = -1
    TrigS4 = -1
    delt_AIC1 = -1
    delt_AIC2 = -1
    delt_AIC3 = -1
    delt_AIC4 = -1

    if rows >= 2:
        AICs2 = np.zeros(N)
        ##row2
        Data_m2_2 = Data[:, 1] * Data[:, 1]
        Data_m1_2 = Data[:, 1]
        total_m1_2 = sum(Data_m1_2[0:N + 1])  # AIC段全部数据和
        total_m2_2 = sum(Data_m2_2[0:N + 1])  # AIC段全部数据平方和
        temp_m1_2 = Data_m1_2[0]
        temp_m2_2 = Data_m2_2[0]
    if rows >= 3:
        AICs3 = np.zeros(N)
        ##row3
        Data_m2_3 = Data[:, 2] * Data[:, 2]
        Data_m1_3 = Data[:, 2]
        total_m1_3 = sum(Data_m1_3[0:N + 1])  # AIC段全部数据和
        total_m2_3 = sum(Data_m2_3[0:N + 1])  # AIC段全部数据平方和
        temp_m1_3 = Data_m1_3[1]
        temp_m2_3 = Data_m2_3[1]
    if rows >= 4:
        AICs4 = np.zeros(N)
        ##row4
        Data_m2_4 = Data[:, 3] * Data[:, 3]
        Data_m1_4 = Data[:, 3]
        total_m1_4 = sum(Data_m1_4[0:N + 1])  # AIC段全部数据和
        total_m2_4 = sum(Data_m2_4[0:N + 1])  # AIC段全部数据平方和
        temp_m1_4 = Data_m1_4[0]
        temp_m2_4 = Data_m2_4[0]

    for k in range(N - 2):  # 1:N - 2:
        j = k + 1
        # row1
        temp_m1 = temp_m1 + Data_m1[j]  # 和
        temp_m2 = temp_m2 + Data_m2[j]  # 平方和
        s1 = temp_m2 / j - (temp_m1 / j) ** 2
        s2 = (total_m2 - temp_m2) / (N - k) - (total_m1 - temp_m1) ** 2 / (N - k) ** 2
        if s1 <= 0:
            s1 = 0.0001
        if s2 <= 0:
            s2 = 0.0001
        AICs1[k] = j * math.log10(s1) + (N - k) * math.log10(s2)
        # row2
        if rows >= 2:
            temp_m1_2 = temp_m1_2 + Data_m1_2[j]  # 和
            temp_m2_2 = temp_m2_2 + Data_m2_2[j]  # 平方和
            s1 = temp_m2_2 / j - (temp_m1_2 / j) ** 2
            s2 = (total_m2_2 - temp_m2_2) / (N - k) - (total_m1_2 - temp_m1_2) ** 2 / (N - k) ** 2
            if s1 <= 0:
                s1 = 0.0001
            if s2 <= 0:
                s2 = 0.0001
            AICs1[k] = j * math.log10(s1) + (N - k) * math.log10(s2)
            AICs2[k] = j * math.log10(s1) + (N - k) * math.log10(s2)
        # row3
        if rows >= 3:
            temp_m1_3 = temp_m1_3 + Data_m1_3[j]  # 和
            temp_m2_3 = temp_m2_3 + Data_m2_3[j]  # 平方和
            s1 = temp_m2_3 / j - (temp_m1_3 / j) ** 2
            s2 = (total_m2_3 - temp_m2_3) / (N - k) - (total_m1_3 - temp_m1_3) ** 2 / (N - k) ** 2
            if s1 <= 0:
                s1 = 0.0001
            if s2 <= 0:
                s2 = 0.0001
            AICs3[k] = j * math.log10(s1) + (N - k) * math.log10(s2)
        # row4
        if rows >= 4:
            temp_m1_4 = temp_m1_4 + Data_m1_4[j]  # 和
            temp_m2_4 = temp_m2_4 + Data_m2_4[j]  # 平方和
            s1 = temp_m2_4 / j - (temp_m1_4 / j) ** 2
            s2 = (total_m2_4 - temp_m2_4) / (N - k) - (total_m1_4 - temp_m1_4) ** 2 / (N - k) ** 2
            if s1 <= 0:
                s1 = 0.0001
            if s2 <= 0:
                s2 = 0.0001
            AICs4[k] = j * math.log10(s1) + (N - k) * math.log10(s2)

    #######################################
    if N > 100:
        point3 = 3
    else:
        point3 = N - 1

    if has_no_complex(AICs1):
        # [AIC_min1, TrigS1] = findpeaks(-1*AICs1)
        a1 = -1 * AICs1
        TrigS1, _ = find_peaks(a1)  # pos
        if not TrigS1.any():# AIC_min1 = a1[TrigS1]  # value
            TrigS1=-1
        else:
            AIC_min1= a1[TrigS1]
        # [AIC_min1, b1] = max(AIC_min1)
            b1 = np.where(AIC_min1 == max(AIC_min1))  # pos
            b1 = int(np.array(b1))
            TrigS1 = TrigS1[b1]
        delt_AIC1 = abs(AICs1[point3] - AICs1[TrigS1])
        if delt_AIC1 is None:
            TrigS1 = -1
        else:
            if (delt_AIC1 < mintresh or TrigS1 == len(AICs1) or TrigS1 == 1):  # AICs1(3) - AIC_min1 < mintresh
                TrigS1 = -1
        TrigSout[:, 0] = TrigS1
        delt_AICout[:, 0] = delt_AIC1
    # row2
    if rows >= 2 and has_no_complex(AICs2):
        # isinstance(AICs2, float) or isinstance(AICs2, int)
        # [AIC_min2, TrigS2] = findpeaks(-1*AICs2)
        a2 = -1 * AICs2
        TrigS2, _ = find_peaks(a2)
        if not TrigS2.any():# AIC_min1 = a1[TrigS1]  # value
            TrigS2=-1
        else:
            AIC_min2 = a2[TrigS2]
            # [AIC_min2, b2] = max(AIC_min2)
            b2 = np.where(AIC_min2 == max(AIC_min2))  # pos
            b2 = int(np.array(b2))
            TrigS2 = TrigS2[b2]
        delt_AIC2 = abs(AICs2[point3] - AICs2[TrigS2])
        if delt_AIC2 is None:
            TrigS2 = -1
        else:
            if (delt_AIC2 < mintresh or TrigS2 == len(AICs2) or TrigS2 == 1):  # AICs2(3) - AIC_min2 < mintresh
                TrigS2 = -1
        TrigSout[:, 1] = TrigS2
        delt_AICout[:, 1] = delt_AIC2
    # row3
    if rows >= 3 and has_no_complex(AICs3):
        # [AIC_min3, TrigS3] = findpeaks(-1*AICs3)
        a3 = -1 * AICs3
        TrigS3, _ = find_peaks(a3)
        if not TrigS3.any():# AIC_min1 = a1[TrigS1]  # value
            TrigS3=-1
        else:
            AIC_min3 = a2[TrigS3]
            # [AIC_min3, b3] = max(AIC_min3)
            b3 = np.where(AIC_min3 == max(AIC_min3))  # pos
            b3 = int(np.array(b3))
            TrigS3 = TrigS3[b3]
        delt_AIC3 = abs(AICs3[point3] - AICs3[TrigS3])
        if delt_AIC3 is None:
            TrigS3 = -1
        else:
            if (delt_AIC3 < mintresh or TrigS3 == len(AICs3) or TrigS3 == 1):  # AICs3(3) - AIC_min3 < mintresh
                TrigS3 = -1
        TrigSout[:, 2] = TrigS3
        delt_AICout[:, 2] = delt_AIC3
    # row4
    if rows >= 4 and has_no_complex(AICs4):
        # [AIC_min4, TrigS4] = findpeaks(-1*AICs4)
        a4 = -1 * AICs4
        TrigS4, _ = find_peaks(a4)
        if not TrigS4.any():# AIC_min1 = a1[TrigS1]  # value
            TrigS4=-1
        else:
            AIC_min4 = a4[TrigS4]
            # [AIC_min4, b4] = max(AIC_min4)
            b4 = np.where(AIC_min4 == max(AIC_min4))  # pos
            b4 = int(np.array(b4))
            TrigS4 = TrigS4[b4]
        delt_AIC4 = abs(AICs4[point3] - AICs4[TrigS4])
        if delt_AIC4 is None:
            TrigS4 = -1
        else:
            if (delt_AIC4 < mintresh or TrigS4 == len(AICs4) or TrigS4 == 1):  # AICs4(3) - AIC_min4 < mintresh
                TrigS4 = -1
        TrigSout[:, 3] = TrigS4
        delt_AICout[:, 3] = delt_AIC4
    ##########################20230815new, 测试可否平替以上部分
    return TrigSout, delt_AICout


def has_no_complex(arr):
    return all(not isinstance(x, complex) for x in arr)


def allpicks(datapeak, n):
    # 找出输入数据中的峰值，按列
    # in：
    # datapeak：输入数据，每列为一个独立的搜索
    # n:1是找峰值，2代表找负峰值，3代表同时找俩
    # out:
    # result:输出峰值的位置；
    result = np.array([])
    # result = result[np.newaxis, :]
    len = np.size(datapeak, 0)
    datapeak = datapeak.reshape(len, -1)
    # if np.size(datapeak)==len:
    #     row = 1  # np.size(datapeak,1)
    # else:
    row = np.size(datapeak, 1)

    if n == 1:
        for i in range(row):
            # [pks1,loc1] = findpeaks(datapeak[:,i])
            y1 = datapeak[:, i]
            loc1, _ = find_peaks(y1)
            pks1 = y1[loc1]
            if np.size(loc1) > 0:  # is not None:
                # result.append(loc1)
                result = np.concatenate((result, loc1))
    elif n == 2:
        for i in range(row):
            # [pks1,loc1] = findpeaks(-1*datapeak[:,i])
            y1 = -1 * datapeak[:, i]
            loc1, _ = find_peaks(y1)
            pks1 = y1[loc1]
            if np.size(loc1) > 0:  # is not None:
                # result.append(loc1)
                result = np.concatenate((result, loc1))
    elif n == 3:
        for i in range(row):
            # [pks1,loc1] = findpeaks(datapeak[:,i])
            y1 = datapeak[:, i]
            loc1, _ = find_peaks(y1)
            pks1 = y1[loc1]
            if np.size(loc1) > 0:  # is not None:
                # result.append(loc1)
                result = np.concatenate((result, loc1))
            # [pks1,loc1] = findpeaks(-1*datapeak[:,i])
            y1 = -1 * datapeak[:, i]
            loc1, _ = find_peaks(y1)
            pks1 = y1[loc1]
            if np.size(loc1) > 0:  # is not None:
                # result.append(loc1)
                result = np.concatenate((result, loc1))
    return result


def polarization(traces):
    # d1: evt data,6*n
    # Pdegree1: sensor1 pdegree
    # Pdegree2: sensor2 pdegree

    result = 0
    Pdegree1 = 0
    Pdegree2 = 0

    if np.size(traces, 1) < 10:  # 10个点以内不算啦
        return Pdegree1, Pdegree2
    ##########################################################传感器1ud
    aa = traces[0:3, :]
    m = np.cov(aa)
    [D, V1] = np.linalg.eigh(m)  # eig_vals, eig_vecs
    if np.size(D, 0) < 3:
        return Pdegree1, Pdegree2
    Pdegree1 = 1 - ((D[1] + D[0]) / (2 * D[2]))
    u1 = math.cos(V1[0, 2])  # 特征向量ｕ１的垂直方向余弦, P: 1, S: 0, v2(3, 3)? 输入为弧度值
    #######################################################传感器2ud
    aa2 = traces[3:, :]
    m2 = np.cov(aa2)
    [D2, V2] = np.linalg.eigh(m2)  # eig_vals, eig_vecs
    if np.size(D2, 0) < 3:
        return Pdegree1, Pdegree2
    Pdegree2 = 1 - ((D2[1] + D2[0]) / (2 * D2[2]))
    u2 = math.cos(V2[0, 2])  # 特征向量ｕ１的垂直方向余弦, P: 1, S: 0, v2(3, 3)? 输入为弧度值

    if (Pdegree1 > 0.7 and Pdegree2 > 0.7):
        result = 1  # 地震
    return Pdegree1, Pdegree2


def TOC_AIC0(Data, thresh, nature):
    # Data: 单方向数据，一列or多列，每列独立运算
    # flag: 是否空运行，1：空运行.0: 正常运行
    # out:
    # TrigS ；S波在当前数据中的点位,line向量
    # TrigSout, line向量
    # delt_AICout, line向量
    # rall列向量

    if thresh <= 0:
        thresh = 700
    # 模拟数据双传感器是一样的，故阈值应该高一些。
    if nature == 0:
        vm = 4.1  # 3太小
    else:
        vm = 10

    rows = np.size(Data, 1)
    N = np.size(Data, 0) - 1

    TrigSout = -1 * np.ones((1, rows))
    delt_AICout = -1 * np.zeros((1, rows))
    rall = -1 * np.zeros((1, rows))
    trigAICall = -1 * np.zeros((1, rows))
    delt_AICout0 = -1 * np.zeros((1, rows))

    # AIC1 = zeros(1, N)
    # Data_m2_1 = Data(:, 1)*Data(:, 1)
    # Data_m1_1 = Data(:, 1)
    # total_m1_1 = sum(Data_m1_1(1:N + 1))  # AIC段全部数据和
    # total_m2_1 = sum(Data_m2_1(1:N + 1))  # AIC段全部数据平方和
    # temp_m1_1 = Data_m1_1(1)
    # temp_m2_1 = Data_m2_1(1)

    # 每列独立样本
    AIC_all = np.zeros((N, rows))  # AIC_all[:,i]
    Data_m2_1_all = np.zeros((N + 1, rows))
    Data_m1_1_all = np.zeros((N + 1, rows))
    total_m1_1_all = np.zeros((1, rows))
    total_m2_1_all = np.zeros((1, rows))
    temp_m1_1_all = np.zeros((1, rows))
    temp_m2_1_all = np.zeros((1, rows))
    for i in range(rows):  # every row
        Data_m2_1_all[:, i] = Data[:, i] * Data[:, i]
        Data_m1_1_all[:, i] = Data[:, i]
        total_m1_1_all[:, i] = sum(Data_m1_1_all[1:N + 1, i])  # AIC段全部数据和
        total_m2_1_all[:, i] = sum(Data_m2_1_all[1:N + 1, i])  # AIC段全部数据平方和
        temp_m1_1_all[:, i] = Data_m1_1_all[1, i]
        temp_m2_1_all[:, i] = Data_m2_1_all[1, i]

        for k in range(N - 2):  # ?
            j = k + 1
            temp_m1_1_all[:, i] = temp_m1_1_all[:, i] + Data_m1_1_all[j, i]  # 和
            temp_m2_1_all[:, i] = temp_m2_1_all[:, i] + Data_m2_1_all[j, i]  # 平方和
            s1 = temp_m2_1_all[:, i] / j - (temp_m1_1_all[:, i] / j) ** 2
            s2 = (total_m2_1_all[:, i] - temp_m2_1_all[:, i]) / (N - k) - (
                        (total_m1_1_all[:, i] - temp_m1_1_all[:, i]) ** 2) / ((N - k) ** 2)
            if s1 <= 0:
                s1 = 0.000001
            if s2 <= 0:
                s2 = 0.000001
            AIC_all[k, i] = j * math.log10(s1) + (N - k) * math.log10(s2)

    trigAICall = -1 * np.ones((1, rows))
    delt_AICout0 = -1 * np.ones((1, rows))
    for i in range(rows):  # every row
        if has_no_complex(AIC_all[:, i]):
            # [AIC_min1, TrigAICS1] = min(AIC_all[2: - 3,i])
            arr1 = AIC_all[2: - 3, i]
            AIC_min1 = min(arr1)
            TrigAICS1 = np.where(arr1 == max(arr1))  # arr1.index(max(arr1))
            if AIC_min1 is None:  # ~exist('AIC_min1', 'var')  # 可能存在非正常AIC结果
                AIC_min1 = AIC_all[2, i]
            delt_AIC10 = abs(AIC_all[2, i] - AIC_min1)
            if delt_AIC10 > thresh:
                trigAICall[:, i] = np.array(TrigAICS1)
            else:
                trigAICall[:, i] = -1

            delt_AICout0[:, i] = delt_AIC10
            ###new, 20230815
            difAIC1 = abs(AIC_all[2:-2, i] - AIC_all[1:-3, i])  # abs(diff(AIC_all[2: - 3,i]))
            # [va1, pos1] = findpeaks(difAIC1)
            pos1, _ = find_peaks(difAIC1)  # pos
            va1 = difAIC1[pos1]  # value
            maxv1 = max(va1)
            maxpos = np.where(va1 == maxv1)
            TrigS1 = pos1[maxpos] + 1
            # 次大
            va1 = np.delete(va1, maxpos)  # = []
            pos1 = np.delete(pos1, maxpos)
            maxv11 = max(va1)
            maxpos11 = np.where(va1 == maxv11)
            if maxv11 is None:
                maxv11 = maxv1
                TrigS11 = TrigS1
            else:
                TrigS11 = pos1[maxpos11] + 1
            ##大于0.75倍的最大值的点
            # pos1last = find(va1 > 0.75 * maxv1)
            pos1last = [index for index, value in enumerate(va1) if value > 0.75 * maxv1]
            if len(pos1last) > 0:
                trigpoint1 = pos1[pos1last[0]] + 1
                trigpoint2 = pos1[pos1last[-1]] + 1
                dmax1 = max([abs(trigpoint1 - TrigS1), abs(trigpoint2 - TrigS1)])
            else:
                dmax1 = 0
            ####
            delt_AIC1 = AIC_all[2, i] - AIC_all[TrigS1, i]
            r1 = abs(delt_AIC1 / AIC_all[2, i])
            s1 = delt_AIC1 >= thresh  # bool
            s10 = dmax1 / N < 0.2
            s2 = (abs(TrigS11 - TrigS1) / N < 0.1 or maxv11 / maxv1 < 0.4)
            s3 = delt_AIC1 >= thresh / 3 and (maxv11 / maxv1 < 0.2)
            a1 = (s1 and s10 and s2 or s3)
            a2 = max(Data[int(TrigS1 - 1):, i]) < max(Data[0: int(TrigS1 - 1), i])
            if (not a1) or a2:
                # if ~(delt_AIC1>=thresh and dmax1/N<0.2 and (abs(TrigS11-TrigS1)/N<0.1 or maxv11/maxv1<0.4)\
                #      or delt_AIC1 >= thresh/3 and (maxv11/maxv1 < 0.2)) or max(Data[TrigS1-1: ,i]) < max(Data[0: TrigS1 - 1, i]): # 700
                TrigS1 = -1
            TrigSout[:, i] = TrigS1
            delt_AICout[:, i] = delt_AIC1
            rall[:, i] = r1
        else:
            TrigSout[:, i] = -1
            delt_AICout[:, i] = -1
            rall[:, i] = -1
            trigAICall[:, i] = -1
            delt_AICout0[:, i] = -1

        ##对5 - 8列结果检查
        # if length(TrigSout) < 8 or length(trigAICall) < 8
        # keyboard
        #
        if len(TrigSout[0]) >= 8:
            TrigSout2 = TrigSout[:, 4:8]
            TrigSout2 = [item for sublist in TrigSout2 for item in sublist]  # one_dimensional_list
            TrigSout2 = [element for element in TrigSout2 if element > 0]  # TrigSout2 = TrigSout2(TrigSout2 > 0)
            if np.var(trigAICall[:, 4: 8]) / np.mean(trigAICall[:, 4: 8]) < vm and (len(TrigSout2) == 0):
                TrigSout[:, 4: 8] = trigAICall[:, 4: 8]  # [TrigSout(1: 4,:)trigAICall(5: 8,:)] #AICold的结果
                delt_AICout[:, 4: 8] = delt_AICout0[:, 4: 8]  # delt_AICout = [delt_AICout(1:4,:)delt_AICout0(5: 8,:)]
            # rall # 未更新，输出不可用

    return TrigSout, delt_AICout, rall







def MyEnder(Stv_arg, N_pnsr):
    Sta_vars = Stv_arg
    MaxDur = float(EEW_Params.MaxDur)
    debug = int(EEW_Params.Debug)
    ForceEndThresh = float(EEW_Params.ForceEndThresh)
    Pend = -1
    afterPGA = -1
    L_fs = float(EEW_Params.L_fs)
    H_fs = float(EEW_Params.H_fs)
    pins = float(EEW_Params.pins)
    ratio = float(EEW_Params.ratio)
    DUR = Sta_vars.StartT - Sta_vars.P_time
    Buffer = np.copy(Sta_vars.Buffer)
    M = np.max(np.abs(Buffer))
    Sprate = int(EEW_Params.Sprate)
    Traces_evt = np.copy(Sta_vars.Traces_evt)
    len = np.size(Traces_evt[0])

    if len < (2.5 * Sprate):
        return Pend, Sta_vars, afterPGA
    if DUR > MaxDur:
        Pend = 1
        if debug >=1:
            print("----------持时大于300秒了and MAX BUFFER<2GAL，强制结束 " + str(DUR) + str(M))
        return Pend, Sta_vars, afterPGA
    if len > Sprate:
        Span1s1 = np.copy(Sta_vars.Buffer[0][-2 * Sprate:])
        Span1s2 = np.copy(Sta_vars.Buffer[1][-2 * Sprate:])
        Span1s1 = Span1s1 - np.mean(Span1s1)
        Span1s2 = Span1s2 - np.mean(Span1s2)
        if np.max(Span1s1) < ForceEndThresh and np.max(Span1s2) < ForceEndThresh:
            Pend =1
            if debug >= 1:
                print("---------去除基线的水平2通道的最近2秒幅值已经小于" + str(ForceEndThresh) + "gal， 结束 ")
            return Pend, Sta_vars, afterPGA
        if np.isnan(Sta_vars.PGA_Curr):
            return Pend, Sta_vars, afterPGA
        if CkSteady(Buffer[0][-10 * Sprate:], ratio, pins) and CkSteady(Buffer[1][-10 * Sprate:], ratio,
                                                                        pins) and CkSteady(Buffer[2][-10 * Sprate:],
                                                                                           ratio, pins) \
                and M < Sta_vars.PGA_Curr / 10:
            Pend = 1
            if debug >= 1:
                print("--------Buffer后10秒 Steady了, Buffer max小于PGA/10 或者小于3.5gal then end -------------")
            return Pend, Sta_vars, afterPGA
        # if len / Sprate < 10:
        #     return Pend, Sta_vars, afterPGA
        Traces_evt[0] = MyFilter(Sprate, Traces_evt[0], L_fs, H_fs)
        Traces_evt[1] = MyFilter(Sprate, Traces_evt[1], L_fs, H_fs)
        Traces_evt[2] = MyFilter(Sprate, Traces_evt[2], L_fs, H_fs)
        ple = Traces_evt[0][-Sprate:]
        pln = Traces_evt[1][-Sprate:]
        plz = Traces_evt[2][-Sprate:]
        EW_pnsr = np.max(ple) - np.min(ple)
        NS_pnsr = np.max(pln) - np.min(pln)
        UD_pnsr = np.max(plz) - np.min(plz)
        if EW_pnsr / N_pnsr[0] < 3 and NS_pnsr / N_pnsr[1] < 3 and UD_pnsr / N_pnsr[2] < 3:
            Pend = 1
            if debug >= 1:
                print('--------信噪比低，结束结束 ---------')
            return Pend, Sta_vars, afterPGA
        Traces_value = np.sqrt(Traces_evt[0] ** 2 + Traces_evt[1] ** 2 + Traces_evt[2] ** 2)
        PGA = np.max(Traces_value)
        PGA_end = np.max(Traces_value[-Sprate * 3 + 1:])
        if (PGA_end / PGA) < 0.05 and PGA_end<5 :
            Pend = 1
            if debug >= 1:
                print(PGA_end / PGA)
                print("--------事件大幅衰减，结束结束 ---------")
            return Pend, Sta_vars, afterPGA
    return Pend, Sta_vars, afterPGA


def FinePick(Data1,Data2,method, N1, N2, LTWlen):
    # AIC 池则信息法自回归突变点判断
    # method=1 AIC
    # metho=2 BIC
    method = 1
    if method == 1:
        if N2>3000:
            N2=3000
        if N1<0:
            N1=0
        N1=int(N1)
        N2=int(N2)

        N = N2 - N1
        AIC1 = np.zeros(N)
        AIC2 = np.zeros(N)
        if N1 <= 0:
            N1 = 1

        Data_m21 = Data1[N1 - 1:N2 + 1] **2
        Data_m11 = Data1[N1 - 1:N2 + 1]
        Data_m22 = Data2[N1 - 1:N2 + 1] **2
        Data_m12 = Data2[N1 - 1:N2 + 1]
        total_m11 = np.sum(Data_m11[:N + 1])  # AIC段全部数据和
        total_m21 = np.sum(Data_m21[:N + 1])  # AIC段全部数据平方和
        total_m12 = np.sum(Data_m12[:N + 1])  # AIC段全部数据和
        total_m22 = np.sum(Data_m22[:N + 1])  # AIC段全部数据平方和

        temp_m11 = Data_m11[0]
        temp_m21 = Data_m21[0]
        temp_m12 = Data_m12[0]
        temp_m22 = Data_m22[0]
        Trig = -1
        if temp_m11 == 0:
            temp_m11 = 0.0000000005

        if temp_m21 == 0:
            temp_m21 = 0.0000000005

        if temp_m12 == 0:
            temp_m12 = 0.0000000005

        if temp_m22 == 0:
            temp_m22 = 0.0000000005


        for k in range(N - 2):
            kk = k + 1
            temp_m11 = temp_m11 + Data_m11[kk]
            temp_m21 = temp_m21 + Data_m21[kk]
            temp_m12 = temp_m12 + Data_m12[kk]
            temp_m22 = temp_m22 + Data_m22[kk]

            temp11 = temp_m21 / (kk + 1) - (temp_m11 / (kk + 1)) ** 2
            temp21 = (total_m21 - temp_m21) / (N - k - 1) - ((total_m11 - temp_m11) / (N - k - 1)) ** 2

            temp12 = temp_m22 / (kk + 1) - (temp_m12 / (kk + 1)) ** 2
            temp22 = (total_m22 - temp_m22) / (N - k - 1) - ((total_m12 - temp_m12) / (N - k - 1)) ** 2
            # plt.figure()
            # plt.plot(Data_m1)
            try:
                ls11 = math.log10(temp11)
                ls21 = math.log10(temp21)
                AIC1[k] = (kk + 1) * ls11.real + (N - k - 1) * ls21.real

                ls12 = math.log10(temp12)
                ls22 = math.log10(temp22)
                AIC2[k] = (kk + 1) * ls12.real + (N - k - 1) * ls22.real
            except:
                print("Aic 异常 EEW_Triger，FinePick")
        AIC_min1 = np.min(AIC1[2:-3])  # confirm
        Trig1 = np.argmin(AIC1[2:-3])
        Trig1 = Trig1 + N1 + 1

        AIC_min2 = np.min(AIC2[2:-3])  # confirm
        Trig2 = np.argmin(AIC2[2:-3])
        Trig2 = Trig2 + N1 + 1

        if Trig1<Trig2:
            Trig=Trig1
        elif Trig2>Trig1:
            Trig=Trig2
        else:
            Trig=Trig1

        # a0 = Data1[Trig]
        # temp_triger1=Trig
        # temp_triger2=Trig
        # for i in range(10):
        #     if np.abs(Data1[temp_triger1 - 1]) < np.abs(a0):
        #         if Data1[temp_triger1 - 1] * a0 >= 0:
        #             a0 = Data1[temp_triger1 - 1]
        #             temp_triger1 = temp_triger1 - 1
        #         else:
        #             temp_triger1 = temp_triger1 - 1
        #             break
        #     else:
        #         break
        #
        # for i in range(10):
        #     if np.abs(Data1[temp_triger2 - 1]) < np.abs(a0):
        #         if Data1[temp_triger2 - 1] * a0 >= 0:
        #             a0 = Data1[temp_triger2 - 1]
        #             temp_triger2 = temp_triger2 - 1
        #         else:
        #             temp_triger2 = temp_triger2 - 1
        #             break
        #     else:
        #         break
        #
        # if temp_triger1<temp_triger1:
        #     Trig=temp_triger1
        # else:
        #     Trig = temp_triger2
        #
        if AIC1[2] - AIC_min1 <700 or AIC2[2] - AIC_min2 <700:
            # print(AIC1[2] - AIC_min1 )
            Trig = -1
        #     print(AIC1[2] - AIC_min1)
        #
        # else:
        #     print(AIC1[2] - AIC_min1 )
        # plt.plot(AIC1)
        # print(AIC1[2] - AIC_min1)
        #Trig = Trig-20
        # if Trig>0:
        #     print('here')
        return Trig

def CkSteady(data, ratiomax, pins):
    result = False
    len = np.size(data)
    len4 = int(np.floor(len / pins))
    pins = int(pins)
    MM = np.zeros((pins, 1))
    for i in range(pins):
        temp = np.abs(data[1 + (i) * len4:(i + 1) * len4 + 1])
        MM[i] = np.max(temp)
    np.sort(MM)
    ratio1 = MM[0] / MM[-1]
    ratio2 = MM[-2] / MM[-1]
    if ratio1 > ratiomax and ratio2 > 0.99:
        result = True
        return result
    return result


def FilterS(Fs, Z):
    Z1 = np.copy(Z)
    len = int(np.size(Z1))
    if len < 256:
        y = 0
        return y
    b = [-2.49263027857326e-22, -2.37225438680822e-06, -1.98420583459990e-05, -7.90328729629732e-05,
         -0.000221450490630881, -0.000491564910530530,
         -0.000903145496537876, -0.00138414413073868, -0.00171166539125376, -0.00146094258661226, 9.03453844468861e-19,
         0.00344148081494769,
         0.00961813769495998, 0.0190483797807789, 0.0317785055639675, 0.0472049797160798, 0.0640282344401041,
         0.0803835826942116, 0.0941448519812235, 0.103339698229090,
         0.106572618553273, 0.103339698229090, 0.0941448519812235, 0.0803835826942116, 0.0640282344401041,
         0.0472049797160798, 0.0317785055639675, 0.0190483797807789,
         0.00961813769495999, 0.00344148081494770, 9.03453844468861e-19, -0.00146094258661226, -0.00171166539125376,
         -0.00138414413073868, -0.000903145496537877,
         -0.000491564910530531, -0.000221450490630881, -7.90328729629737e-05, -1.98420583459984e-05,
         -2.37225438680844e-06, -2.49263027857326e-22]
    a = 1
    y = filter.lfilter(b, a, Z1)  # 41 为 a、b数组最大值
    # y=filter.filtfilt(b,1,Z1)
    return y