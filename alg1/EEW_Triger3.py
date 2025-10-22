# -*-coding:UTF-8 -*-
import math
import signal
import numpy as np
import scipy.integrate as integrate
from scipy.signal import lfiltic,lfilter_zi
import torch
import SimpleLogger as logger
from StaticVar import StaticVar as persistent  # 静态类
from StaticVar import Static_EEW_Params as EEW_Params  # 静态类
from function1 import MyFilter
from toc_aic import TOC_AIC
import scipy.signal as filter
from scipy.signal import find_peaks, decimate
from collections import Counter
import SimpleLogger as logger
import scipy.integrate as integrate
print(torch.__version__)
device = torch.device('cpu')
from Nature_Single import *
model_Pwave=None
model_Swave=None
def EEW_Triger2_Nature(Stv_arg1,Stv_arg2, Alarm,model_Pwave1,model_Swave1):
    global model_Pwave
    global model_Swave
    model_Pwave=model_Pwave1
    model_Swave=model_Swave1
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
    # if Buffer_len>3000:
    #     Buffer_len=3000
    # if PackLen>2980:
    #     PackLen=2980


    # 事件捡拾
    if Triged < 1:  # 未触发
        # try:
        endtimepre = Stv_arg1.End_time # Stv_arg1.End_time
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

        if (Triged_time <= persistent.end_time_Pre or Triged_time - persistent.P_time_Pre <= 0.1 or
                Triged_time <= persistent.S_time_Pre):  # Trig_time <= end_time_Pre):
            Triged = -1
            Sta_vars1.Triged = Triged
            Sta_vars2.Triged = Triged
            return Sta_vars1,Sta_vars2, disAlarm,Alarm

        Sta_vars1.Traces_evt = Sta_vars1.BaseLine[:, Trig_fine:]
        Sta_vars2.Traces_evt = Sta_vars2.BaseLine[:, Trig_fine:]
        Sta_vars1.P_time = Triged_time
        Sta_vars2.P_time = Triged_time

        if Debug >= 1 and persistent.FirstStart==0:
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
        if Sta_vars1.End_time<0 and Sta_vars2.End_time<0:
            [Pend1, Sta_vars1, afterPGA] = MyEnder(Sta_vars1, N_pnsr1)
            [Pend2, Sta_vars2, afterPGA] = MyEnder(Sta_vars2, N_pnsr2)
            if Pend1 >= 0 and Pend2>=0:
                # disAlarm = 1
                print("--------------事件结束------------------")
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
            if Sta_vars1.End_time > 0 and Sta_vars2.End_time > 0:
                if Sta_vars1.Is_EQK == 1 and Sta_vars2.Is_EQK == 1:
                    disAlarm = 1
                else:
                    disAlarm = 2
            persistent.PGAold = -1
            persistent.AZIold = -1
            persistent.distold = -1
        return Sta_vars1,Sta_vars2,disAlarm, Alarm


##############################################################################下面为子函数
def MypickAIC(Sprate, PackLen, filter_Buffer,Buffer,endtimepre,startT):
    # 触发数据外围处理及判断
    global model_Pwave

    Back = int(EEW_Params.Back)
    Fowrd = int(EEW_Params.Fowrd)
    thresh = float(EEW_Params.thresh)*1.1
    STW = float(EEW_Params.STW)
    LTW = float(EEW_Params.LTW)
    ifliter = int(EEW_Params.iflilter)
    MinThresh = float(EEW_Params.MinThresh)
    Bufffer_second=float(EEW_Params.Buffer_seconds)*3
    Trig_fine = -1
    Trigraw = -1
    L_fs = float(EEW_Params.L_fs)
    H_fs = float(EEW_Params.H_fs)
    len = int(np.size(filter_Buffer[0]))

    Buffer_len=int(Sprate*Bufffer_second)
    Buffer_len_now=int(np.shape(filter_Buffer[0])[0])
    Data1 = Buffer[-Buffer_len:][2]
    Data2 = Buffer[-Buffer_len:][5]
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
    # from matplotlib import pyplot as plt
    # MinThresh=0#调试用
    # if max1>MinThresh and max1>max3 and max12>MinThresh and max12>max32:
    Trigraw = MyPicker(Buffer1,Buffer2, Sprate, 1, thresh, STW, LTW, ifliter, PackLen,endtimepre,startT)
    if Trigraw<=0 and max1 > MinThresh  and max12 > MinThresh :   #and max1 > max3 and max12 > max32
        ind1 = np.where(np.abs(Data1) == max1)
        ind2 = np.where(np.abs(Data2) == max12)
        try:
            all_indices = np.concatenate([ind1[0], ind2[0]])
            if all_indices.size > 0:
                Trigraw = np.min(all_indices)
        except:
            Trigraw =-1
    len_Buf = Buffer1.size
    if Trigraw>0:
        Trigraw_T = startT+(PackLen+Trigraw-len_Buf)/Sprate

        if Trigraw_T<=endtimepre:
            Trigraw = -1
        else:
            Buffer1_t0 = startT + (PackLen - len_Buf) / Sprate  # buffer时间起点
            # 触发点以后的数据
            dataaf1 = Data1[Trigraw:]   # Buffer1[Trigraw:]
            dataaf2 = Data2[Trigraw:]   #Buffer2[Trigraw:]
            if endtimepre < Buffer1_t0:
                datapre1 = Data1[0:Trigraw]  # Buffer1[0:Trigraw - 1]
                datapre2 = Data2[0:Trigraw]  # Buffer2[0:Trigraw - 1]
            else:
                endtimepre_pos = int(len_Buf-(startT-endtimepre+PackLen/Sprate)*Sprate)
                datapre1 = Data1[endtimepre_pos:Trigraw]  # Buffer1[endtimepre_pos:Trigraw - 1]
                datapre2 = Data2[endtimepre_pos:Trigraw]  # Buffer2[endtimepre_pos:Trigraw - 1]
            maxaf = np.max([np.abs(dataaf1),np.abs(dataaf2)])
            maxpre = np.max([np.abs(datapre1), np.abs(datapre2)])
            if maxpre >= maxaf:
                Trigraw = -1
    if Trigraw == -1:
        return Trig_fine
    NS1=Buffer[0]
    EW1=Buffer[1]
    UD1=Buffer[2]
    NS2=Buffer[3]
    EW2=Buffer[4]
    UD2=Buffer[5]
    downsample_factor = 2
    NS1_dec = decimate(NS1, downsample_factor, ftype='iir')
    EW1_dec = decimate(EW1, downsample_factor, ftype='iir')
    UD1_dec = decimate(UD1, downsample_factor, ftype='iir')
    NS2_dec = decimate(NS2, downsample_factor, ftype='iir')
    EW2_dec = decimate(EW2, downsample_factor, ftype='iir')
    UD2_dec = decimate(UD2, downsample_factor, ftype='iir')
    b, a = filter.butter(2, 15*2/100, 'low')
    # b, a = filter.butter(2, 0.075 * 2 / 200, 'high')
    # import matplotlib.pyplot as plt
    NS1_dec = filter.filtfilt(b,a,NS1_dec)
    EW1_dec=filter.filtfilt(b,a,EW1_dec)
    UD1_dec=filter.filtfilt(b,a,UD1_dec)
    NS2_dec=filter.filtfilt(b,a,NS2_dec)
    EW2_dec=filter.filtfilt(b,a,EW2_dec)
    UD2_dec=filter.filtfilt(b,a,UD2_dec)
    Ph_Buffer1=np.stack((EW1_dec,NS1_dec,UD1_dec),axis=1).astype(np.float32)
    Ph_Buffer2 = np.stack((EW2_dec, NS2_dec, UD2_dec),axis=1).astype(np.float32)
    # Ph_Buffer11 = np.stack((EW1_dec, NS1_dec, UD1_dec),axis=0).astype(np.float32)
    # Ph_Buffer22 = np.stack((EW2_dec, NS2_dec, UD2_dec), axis=0).astype(np.float32)
    # Ph_Buffer11=Ph_Buffer11[np.newaxis,:]
    # phaseNet_model=torch.load("D:\\Python_EEW\\cres-cce1\\jit\\9_.si.chuan.sheng..jit.jit",map_location=torch.device('cpu'))#axis=1
    # model = AutoEncoder()
    # model.load_state_dict(torch.load("D:\\Python_EEW\\cres-cce1\\denoise.pt",map_location=torch.device('cpu')))
    # phaseNet_model = torch.load("D:\\Python_EEW\\cres-cce1\\china.rnn.jit",map_location=torch.device('cpu'))#axis=1数据不要太短
    # pnsn
    # .01.


    # device=torch.device("cpu")
    # phaseNet_model.eval()
    # model.eval()
    # phaseNet_model.to(device)diting.eqt.jit
    # phaseNet_model = torch.jit.load("D:\\Python_EEW\\cres-cce1\\rnn.pnsn.01.jit")#axis=1
    # phaseNet_model = torch.jit.load("D:\\Python_EEW\\cres-cce1\\eqt.jit")  # axis=0
    try:
        with torch.no_grad():
            # import time
            # 记录开始时间
            # start_time = time.perf_counter()
            x1 = torch.tensor(Ph_Buffer1, dtype=torch.float32, device='cpu')
            x2 = torch.tensor(Ph_Buffer2, dtype=torch.float32, device='cpu')
            y = model_Pwave(x1)
            y2 = model_Pwave(x2)
            # from matplotlib import pyplot as plt
            phase1 = y.cpu().numpy()
            phase2 = y2.cpu().numpy()
            count1 = np.where(phase1[:, 0] == 0)
            count2 = np.where(phase2[:, 0] == 0)
            if phase2[0,2]<0.45 and phase1[0,2]<0.45:
                count1=None
                count2=None
            if count1 is None and  count2 is None:  # RNN识别失败
                if ifliter > 0:
                    DataF1 = MyFilter(Sprate, Data1, L_fs, H_fs)
                    DataF2 = MyFilter(Sprate, Data2, L_fs, H_fs)
                PUpper = int(Trigraw - Back * Sprate)
                if PUpper <= 1:
                    PUpper = 1
                Plower = Trigraw + Fowrd * Sprate
                if len - Plower <= 1 * Sprate:
                    Plower = len
                Trig_fine = FinePick(DataF1, DataF2, 1, PUpper, Plower, LTW * Sprate)
                return Trig_fine
            if count1[0].size==0 and count2[0].size==0: # RNN识别失败
                if ifliter > 0:
                    DataF1 = MyFilter(Sprate, Data1, L_fs, H_fs)
                    DataF2 = MyFilter(Sprate, Data2, L_fs, H_fs)
                PUpper = int(Trigraw - Back * Sprate)
                if PUpper <= 1:
                    PUpper = 1
                Plower = Trigraw + Fowrd * Sprate
                if len - Plower <= 1 * Sprate:
                    Plower = len
                Trig_fine = FinePick(DataF1, DataF2, 1, PUpper, Plower, LTW * Sprate)
                return Trig_fine
            if  count1[0].size==1 and count2[0].size==0:
                Trig_fine=int(phase1[count1[0],1])*2
                Befwind = Buffer[1][Trig_fine - 200:Trig_fine]
                aftwind = Buffer[1][Trig_fine:Trig_fine + 200]
                E_bef = np.linalg.norm(Befwind)
                E_aft = np.linalg.norm(aftwind)
                if E_aft / E_bef > 1.05:
                    return Trig_fine
                else:
                    if ifliter > 0:
                        DataF1 = MyFilter(Sprate, Data1, L_fs, H_fs)
                        DataF2 = MyFilter(Sprate, Data2, L_fs, H_fs)
                    PUpper = int(Trigraw - Back * Sprate)
                    if PUpper <= 1:
                        PUpper = 1
                    Plower = Trigraw + Fowrd * Sprate
                    if len - Plower <= 1 * Sprate:
                        Plower = len
                    Trig_fine = FinePick(DataF1, DataF2, 1, PUpper, Plower, LTW * Sprate)
                    return Trig_fine

            if count2[0].size==1 and  count1[0].size==0:
                Trig_fine =int(phase2[count2[0],1])*2
                Befwind = Buffer[1][Trig_fine - 200:Trig_fine]
                aftwind = Buffer[1][Trig_fine:Trig_fine + 200]
                E_bef = np.linalg.norm(Befwind)
                E_aft = np.linalg.norm(aftwind)
                if E_aft / E_bef > 1.05:
                    return Trig_fine
                else:
                    if ifliter > 0:
                        DataF1 = MyFilter(Sprate, Data1, L_fs, H_fs)
                        DataF2 = MyFilter(Sprate, Data2, L_fs, H_fs)
                    PUpper = int(Trigraw - Back * Sprate)
                    if PUpper <= 1:
                        PUpper = 1
                    Plower = Trigraw + Fowrd * Sprate
                    if len - Plower <= 1 * Sprate:
                        Plower = len
                    Trig_fine = FinePick(DataF1, DataF2, 1, PUpper, Plower, LTW * Sprate)
                    return Trig_fine

            if count2[0].size==1 and  count1[0].size==1:
                tmepind1=int(phase1[count1[0],1])*2
                tmepind2 = int(phase2[count2[0],1])*2
                Trig_fine=int(np.max([tmepind1,tmepind2]))



                if np.abs(tmepind1-tmepind2)>200:
                    if phase1[count1[0],2]>phase2[count1[0],2]:
                        Trig_fine=tmepind1
                    elif phase1[count1[0],2]<phase2[count1[0],2]:
                        Trig_fine = tmepind2
                temp_len=Buffer_len-Trig_fine
                if temp_len>600 and temp_len<2300 :
                    Trig_fine=Trig_fine #+5
                elif temp_len>=2300 and abs(tmepind1-tmepind2)>20:
                    if ifliter > 0:
                        DataF1 = MyFilter(Sprate, Data1, L_fs, H_fs)
                        DataF2 = MyFilter(Sprate, Data2, L_fs, H_fs)
                    PUpper = int(Trig_fine - Back * Sprate)
                    if PUpper <= 1:
                        PUpper = 1
                    Plower = Trig_fine + Fowrd * Sprate
                    if len - Plower <= 1 * Sprate:
                        Plower = len
                    Trig_fine = FinePick(DataF1, DataF2, 1, PUpper, Plower, LTW * Sprate)
                else:
                    Befwind = Buffer[1][Trig_fine - 40:Trig_fine]
                    aftwind = Buffer[1][Trig_fine:Trig_fine + 40]
                    E_bef = np.linalg.norm(Befwind)
                    E_aft = np.linalg.norm(aftwind)
                    if E_aft/E_bef<=1.05:
                        Trig_fine=Trig_fine#+40

                return Trig_fine


            if count2[0].size>1 or count1[0].size>1:
                tmepind1 = phase1[count1[0], 1] * 2
                tmepind2 = phase2[count2[0], 1] * 2
                idx=np.abs(tmepind1-tmepind2).argmin()
                if np.size(tmepind1)>np.size(tmepind2):
                    Trig_fine=int(tmepind1[idx])
                else:
                    Trig_fine = int(tmepind2[idx])
                return Trig_fine

        # import matplotlib.pyplot as plt
        # plt.plot(UD1, alpha=0.5)
    except Exception as e1:
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
    return Trig_fine


def MyPicker(acc1,acc2, sprate, phase, thresh, STW, LTW, iflilter, PackLen,endtimepre,startT):
    # phase:1(P波) 2(S波)
    acc111=np.copy(acc1)
    acc222 = np.copy(acc2)
    acc111=acc111[20:]
    acc222 =acc222[20:]
    len = np.size(acc111)
    difacc1 = np.diff(acc111)
    difacc2 = np.diff(acc222)
    # difacc=np.append(0,np.diffacc))
    # difacc=np.hstack(([[0],difacc]))
    spanDeoddSecond = float(EEW_Params.spanDeOddSecond)
    L_fs = float(EEW_Params.L_fs)
    H_fs = float(EEW_Params.H_fs)
    # debug = float(EEW_Params.Debug)
    trig = -1
    difacc12 = DeOdd(difacc1, spanDeoddSecond * sprate)
    difacc22 = DeOdd(difacc2, spanDeoddSecond * sprate)
    difacc12 =difacc12
    difacc22 =difacc22
    Facc1 = integrate.cumtrapz(difacc12, initial=0)
    Facc2 = integrate.cumtrapz(difacc22, initial=0)

    if iflilter > 0:
        Facc1 = MyFilter(sprate, Facc1, L_fs, H_fs)#H_fs
        Facc2 = MyFilter(sprate, Facc2, L_fs, H_fs)
    CF1 = (Facc1 + 0.01) ** 2
    CF2 = (Facc2 + 0.01) ** 2
    len = CF1.size
    # Lta1 = np.mean(CF1[:int(LTW * sprate+1)])
    # Lta2 = np.mean(CF2[:int(LTW * sprate+1)])

    # 原则上每包进来只算一次
    step = int(0.2*sprate)
    if PackLen > int(0.2*sprate):
        n = math.ceil(PackLen/(0.2*sprate))
        step = int(PackLen/n)
    # 结束点在当前buffer中的位置,点
    Trigt_end = int(len - (startT - endtimepre + PackLen / sprate) * sprate)
    if int(len - 5 * sprate) <= 0:
        return trig
    nbegin = int(len - 5 * sprate - (PackLen)) # 循环开始位置
    if nbegin <= 0:
        nbegin = 1
    for i in range(nbegin,int(len-5*sprate),step):
        LTA1 = np.mean(CF1[i + 1:int(i + 5 * sprate)])
        LTA2 = np.mean(CF2[i + 1:int(i + 5 * sprate)])
        STA1 = np.mean(CF1[int(i + 5 * sprate - 0.2 * sprate + 1):int(i + 5 * sprate)]) #STW：0.4，配置参数获取
        STA2 = np.mean(CF2[int(i + 5 * sprate - 0.2 * sprate + 1):int(i + 5 * sprate)]) #STW：0.4，配置参数获取
        minSTALTA = min([STA1 / LTA1, STA2 / LTA2])
        if (round(STA1/LTA1) >= thresh) and (round(STA2/LTA2) >= thresh): # 20
            # #If there is interference + earthquake in a data window at the same time,
            # #please  eliminate the interference(there is interference before earthquake)
            if Trigt_end > 0:
                Trigt_p = int(i + 5 * sprate - 0.2 * sprate + 1)
                # #触发点与Endtime相差不过3秒，跳过此段
                if Trigt_p < Trigt_end: #  | | Trigt_p - Trigt_end <= sprate * 1
                    continue

            trig = int(i + 5 * sprate - 0.2 * sprate + 1)
            # probty_sta = (STA1 / LTA1 + STA2 / LTA2) / 2 / thresh
            break


    if trig == -1:  # 没有触发
        Pbegin = int(len - 0.2 * sprate)  # 短窗起点，从长窗之后开始
        ind = np.where(Facc1[Pbegin:] > 10)[0]    # find(Facc1(Pbegin:end) >= 10)
        ind2 = np.where(Facc2[Pbegin:] > 10)[0]   #find(Facc2(Pbegin:end) >= 10 )  # 20240321
        if ind.size>0 and ind2.size>0:   # 查看加速度滤波后是否已经超过10gal了
            indmin = min(ind[0],ind2[0])
            trig = Pbegin + indmin
    return  trig



    # if phase == 1:
    #     Pbegin = int(LTW*sprate)
    #     tempnum =int(len-Pbegin-STW*sprate)
    #     for i in range(0,tempnum,20):
    #         Sta1 = np.mean(CF1[int(Pbegin+ i):int(Pbegin + i+STW*sprate)])
    #         Sta2 = np.mean(CF2[int(Pbegin+ i):int(Pbegin + i+STW*sprate)])
    #         # print(Sta1/Lta1 )
    #
    #         if Sta1 / Lta1 > thresh and Sta2/Lta2>thresh:
    #             trig = i + Pbegin+20#+20
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
        if len / Sprate < 10:
            return Pend, Sta_vars, afterPGA
        Traces_evt[0] = MyFilter(Sprate, Traces_evt[0], L_fs, H_fs)
        Traces_evt[1] = MyFilter(Sprate, Traces_evt[1], L_fs, H_fs)
        Traces_evt[2] = MyFilter(Sprate, Traces_evt[2], L_fs, H_fs)
        ple = Traces_evt[0][-Sprate:]
        pln = Traces_evt[1][-Sprate:]
        plz = Traces_evt[2][-Sprate:]
        EW_pnsr = np.max(ple) - np.min(ple)
        NS_pnsr = np.max(pln) - np.min(pln)
        UD_pnsr = np.max(plz) - np.min(plz)
        if EW_pnsr / N_pnsr[0] < 2.5 and NS_pnsr / N_pnsr[1] <2.5 and UD_pnsr / N_pnsr[2] <2.5:
            Pend = 1
            if debug >= 1:
                print('--------信噪比低，结束结束 ---------')
            return Pend, Sta_vars, afterPGA
        # from matplotlib import pyplot as plt
        Traces_value = np.sqrt(Traces_evt[0] ** 2 + Traces_evt[1] ** 2 + Traces_evt[2] ** 2)
        PGA = np.max(Traces_value)
        PGA_end = np.max(Traces_value[-Sprate * 3 + 1:])
        if (PGA_end / PGA) < 0.1 :
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
    Trig = -1

    # 如果最大值在第一秒，那么不计算了
    if np.max(np.abs(Data1[0:200]))>=np.max(np.abs(Data1[201:])) and np.max(np.abs(Data2[0:200]))>=np.max(np.abs(Data2[201:])):
        return Trig


    if method == 1:
        # if N2>3000:
        #     N2=3000
        # if N1<0:
        #     N1=0
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

        a0 = Data1[Trig]
        temp_triger1=Trig
        temp_triger2=Trig
        for i in range(10):
            if np.abs(Data1[temp_triger1 - 1]) < np.abs(a0):
                if Data1[temp_triger1 - 1] * a0 >= 0:
                    a0 = Data1[temp_triger1 - 1]
                    temp_triger1 = temp_triger1 - 1
                else:
                    temp_triger1 = temp_triger1 - 1
                    break
            else:
                break

        for i in range(10):
            if np.abs(Data1[temp_triger2 - 1]) < np.abs(a0):
                if Data1[temp_triger2 - 1] * a0 >= 0:
                    a0 = Data1[temp_triger2 - 1]
                    temp_triger2 = temp_triger2 - 1
                else:
                    temp_triger2 = temp_triger2 - 1
                    break
            else:
                break

        if temp_triger1 < temp_triger1:
            Trig = temp_triger1
        else:
            Trig = temp_triger2

        if np.abs(AIC1[-3] - AIC_min1) <500 or np.abs(AIC2[-3] - AIC_min2) <500:
            Trig = -1
        # else:
        #     print('np.max(AIC1) - AIC_min1 :'+str(np.max(AIC1) - AIC_min1)+'np.max(AIC2) - AIC_min2: '+str(np.max(AIC2) - AIC_min2))

        #     print(AIC1[2] - AIC_min1)
        #
        # else:
        #     print(AIC1[2] - AIC_min1 )
        # plt.plot(AIC1)
        # print(AIC1[2] - AIC_min1)
        #Trig = Trig-20

        if np.max(np.abs(Data1[0:Trig-1])) >= np.max(np.abs(Data1[Trig:])) and np.max(np.abs(Data2[0:Trig-1])) >= np.max(
                np.abs(Data2[Trig:])):
            Trig = -1
        if Trig>0:
            str1 = 'finepick out result '
            print(str1)
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

def MypickerS(Stv_arg1,Stv_arg2,Trace_evt,Sprate,model_Swave):
    # phaseNet_model = torch.load("D:\\Python_EEW\\cres-cce1\\rnn.pnsn.01.jit",
    #                             map_location=torch.device('cpu'))  # axis=1数据不要太短
    # rnn.pnsn.jit
    Sta_vars1 = Stv_arg1
    Sta_vars2 = Stv_arg2
    Buffer1 = np.copy(Sta_vars1.Buffer)
    Buffer2 = np.copy(Sta_vars2.Buffer)
    len_buffer=np.shape(Buffer1)[1]
    Buffer = np.vstack((Buffer1, Buffer2))
    b, a = filter.butter(4, 30 * 2 / 100, 'low')
    zi = lfilter_zi(b, a)



    NS1 = Buffer[0]
    EW1 = Buffer[1]
    UD1 = Buffer[2]
    NS2 = Buffer[3]
    EW2 = Buffer[4]
    UD2 = Buffer[5]
    temp_NS1=NS1[-600:]
    temp_EW1 = EW1[-600:]
    temp_UD1 = UD1[-600:]
    temp_NS2 = NS2[-600:]
    temp_EW2 = EW2[-600:]
    temp_UD2 = UD2[-600:]

    temp_NS1 = filter.lfilter(b, a, temp_NS1, zi=zi * temp_NS1[0])[0]
    temp_EW1 = filter.lfilter(b, a, temp_EW1, zi=zi * temp_EW1[0])[0]
    temp_UD1 = filter.lfilter(b, a, temp_UD1, zi=zi * temp_UD1[0])[0]
    temp_NS2 = filter.lfilter(b, a, temp_NS2, zi=zi * temp_NS2[0])[0]
    temp_EW2 = filter.lfilter(b, a, temp_EW2, zi=zi * temp_EW2[0])[0]
    temp_UD2 = filter.lfilter(b, a, temp_UD2, zi=zi * temp_UD2[0])[0]


    E_NS1=np.linalg.norm(temp_NS1)
    E_EW1=np.linalg.norm(temp_EW1)
    E_UD1=np.linalg.norm(temp_UD1)
    E_NS2 = np.linalg.norm(temp_NS2)
    E_EW2 = np.linalg.norm(temp_EW2)
    E_UD2 = np.linalg.norm(temp_UD2)
    if (E_UD1/E_NS1>0.95 and E_UD1/E_EW1>0.95) or (E_UD2/E_NS2>0.95 and E_UD2/E_EW2>0.95):
        Triged_time = -1
        return Triged_time
    downsample_factor = 2
    NS1_dec = decimate(NS1, downsample_factor, ftype='iir')
    EW1_dec = decimate(EW1, downsample_factor, ftype='iir')
    UD1_dec = decimate(UD1, downsample_factor, ftype='iir')
    NS2_dec = decimate(NS2, downsample_factor, ftype='iir')
    EW2_dec = decimate(EW2, downsample_factor, ftype='iir')
    UD2_dec = decimate(UD2, downsample_factor, ftype='iir')


    # # b, a = filter.butter(2, 0.075 * 2 / 200, 'high')
    # import matplotlib.pyplot as plt
    NS1_dec = filter.filtfilt(b, a, NS1_dec)
    EW1_dec = filter.filtfilt(b, a, EW1_dec)
    UD1_dec = filter.filtfilt(b, a, UD1_dec)
    NS2_dec = filter.filtfilt(b, a, NS2_dec)
    EW2_dec = filter.filtfilt(b, a, EW2_dec)
    UD2_dec = filter.filtfilt(b, a, UD2_dec)


    StartT=Sta_vars1.StartT
    Ph_Buffer1 = np.stack((EW1_dec, NS1_dec, UD1_dec), axis=1).astype(np.float32)
    Ph_Buffer2 = np.stack((EW2_dec, NS2_dec, UD2_dec), axis=1).astype(np.float32)
    try:
        with torch.no_grad():
            x1 = torch.tensor(Ph_Buffer1, dtype=torch.float32, device='cpu')
            x2 = torch.tensor(Ph_Buffer2, dtype=torch.float32, device='cpu')
            y = model_Swave(x1)
            y2 = model_Swave(x2)
            phase1 = y.cpu().numpy()
            phase2 = y2.cpu().numpy()
            # flag1=np.array([])
            # flag2=np.array([])
            # flag11=np.array([])
            # flag22=np.array([])
            flag1=np.where(phase1[:,0]==1)[0]
            flag2=np.where(phase2[:,0]==1)[0]
            # flag11 =np.where(phase1[:, 0] == 3)[0]#Sn
            # flag22 =np.where(phase2[:, 0] == 3)[0]#Sn

            # if flag1==[] and flag2[]
            # indS1=np.array([])
            # indS2 = np.array([])
            # Prob=np.array([])
            Trig_fine = -1
            if flag1.size>0 and flag2.size==0:
                Prob=phase1[flag1[0]][2]
                if Prob>0.4:#初始 0.3
                    Trig_fine=phase1[flag1[0]][1]*2
            elif flag2.size>0 and flag1.size==0:
                Prob = phase2[flag2[0]][2]
                if Prob > 0.4:#初始 0.3
                    Trig_fine = phase2[flag2[0]][1]*2
            elif flag2.size>0 and flag1.size>0:
                if phase1[flag1[0]][2]>0.15 and phase2[flag2[0]][2]>0.15:
                    indS1=phase1[flag1[0]][1]
                    indS2=phase2[flag2[0]][1]
                    Trig_fine=max([indS1,indS2])*2




            #     phase = phase2
            # phase[:, 1] = phase[:, 1] * 2

        # import matplotlib.pyplot as plt
        # Trig_fine = -1
        # plt.plot(UD1, alpha=0.5)
        # from matplotlib import pyplot as plt
        # for pha in phase:
        #     if pha[0] == 0:
        #         c = "r"
        #     elif pha[0]==1:
        #         if pha[2]<0.3:
        #             print(pha[2])
        #             Trig_fine=-1
        #             continue
        #         Trig_fine = int(pha[1])
        #         break
        #     elif pha[0] == 3:
        #         if pha[2] < 0.3:
        #             print(pha[2])
        #             Trig_fine = -1
        #             continue
        #         Trig_fine = int(pha[1])
        #         break

        Buffer_len=np.shape(UD1)[0]
        Triged_time = round(StartT + (Trig_fine - Buffer_len) / Sprate, 3)
        temp_Point=int(np.abs((Sta_vars1.P_time-Triged_time)*200))
        trace_len=np.shape(Sta_vars1.Traces_evt[0])[0]
        # delta_time=np.abs(Sta_vars1.P_time-Triged_time)*200
        if Triged_time<Sta_vars1.P_time  or Trig_fine==-1 or np.abs(Sta_vars1.P_time-Triged_time)<1 or (trace_len-temp_Point)>trace_len*2 :
            Triged_time=-1
    except Exception as e1:
        Triged_time=-1
    return Triged_time

# def get_min_from_merged(tuple1, tuple2):
#     """合并两个元组并返回最小值"""
#     merged = tuple1 + tuple2
#     return min(merged) if merged else None