import os

# -*-coding:UTF-8 -*-
import numpy as np
import scipy
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import lfilter_zi
from scipy.signal import correlate
from StaticVar import StaticVar as persistent  # 静态类
from StaticVar import Static_EEW_Params as EEW_Params  # 静态类
from function1 import Fourier
from function1 import nextpow2
from xcorr import xcorr
import pywt
def Judges(traces_in ,method ):
    # # function [ result,r,posf ] = Judges(traces, f1, duration,method )
    # #JUDGES Summary of this function goes here
    # # INPUT:
    # #   traces,[EW NS UD] 三分量触发后的数据；
    # #   method，字符串，方法选择，
    # #       decision：决策树
    # #       multi_parameters,频率范围，单一频率等综合方法。
    # #   Parameters={f1,duration,'','',''}
    #
    # # OUTPUT: 逻辑数，为1是则为真实的地震事件；\
    ## traces 对应的外部变量也会被更改，注意赋值
    traces=np.copy(traces_in)
    f1=float(EEW_Params.Sprate)
    MaxCorrInSens=float(EEW_Params.MaxCorrInSens)
    MaxHorientalFirstSecond=float(EEW_Params.MaxHorientalFirstSecond)
    IfCkCorr=float(EEW_Params.IfCkCorr)
    MinCorrEW=float(EEW_Params.MinCorrEW)  #对于真实地址，2个传感器间的最小相关系数。
    MinCorrNS=float(EEW_Params.MinCorrNS)
    MinCorrUD=float(EEW_Params.MinCorrUD)
    corrout = np.array([0.0, 0.0, 0.0], dtype=float)

    if  persistent.flag_nature== 1:  ##
        MinCorrEW=EEW_Params.MinCorrEW_Nature #对于真实地址，2个传感器间的最小相关系数。
        MinCorrNS=EEW_Params.MinCorrNS_Nature
        MinCorrUD=EEW_Params.MinCorrUD_Nature

    L_fs=float(EEW_Params.L_fs)
    H_fs=float(EEW_Params.H_fs)
    Debug=int(EEW_Params.Debug)

    row2 = np.size(traces,1)
    traces_org=np.copy(traces)
    cor1_12f=np.min(np.abs(np.corrcoef(traces[0],traces[1])))#传感器1
    cor1_13f=np.min(np.abs(np.corrcoef(traces[0],traces[2])))
    cor1_23f=np.min(np.abs(np.corrcoef(traces[1],traces[2])))
    cor2_12f=np.min(np.abs(np.corrcoef(traces[3],traces[4])))#传感器2
    cor2_13f=np.min(np.abs(np.corrcoef(traces[3],traces[5])))
    cor2_23f=np.min(np.abs(np.corrcoef(traces[4],traces[5])))
    sum1 =np.array([cor1_12f, cor1_13f, cor1_23f])
    sum2 = np.array([cor2_12f, cor2_13f, cor2_23f])


    Cnum1 = np.size(np.where(sum1>MaxCorrInSens))   # len(find(sum1>MaxCorrInSens))
    Cnum11 = np.size(np.where(sum1>=0.99)) #len(find(sum1>=0.99))
    Cnum2 =  np.size(np.where(sum2>MaxCorrInSens))  #len(find(sum2>MaxCorrInSens))
    Cnum22 = np.size(np.where(sum2>=0.99))  #len(find(sum2>=0.99))
    if Cnum1>=2 or Cnum2>=2 or Cnum11>=1 or Cnum22>=1: #两个以上大于阈值或者1个以上等于1，视为通道内相似性太高
        result=0
        if Debug==1:
            print('-------------同一传感器间的相关系数高--cor1_12f/cor1_13f/cor1_23f/cor2_12f/cor2_13f/cor2_23f=' +str(cor1_12f)+str(cor1_13f) +\
            str(cor1_23f) + str(cor2_12f) +str(cor2_13f) +str(cor2_23f))
        return result,corrout

    #######20210610增加,滤波前的.根据对比分析，使用滤波前的更好区分干扰和地震
    DataLen = np.size(traces[2,:])
    if DataLen>=450:
        DataLen=450
    mi=nextpow2(DataLen)
    NFFT =2**mi  # Next power of 2 from length of y
    #NFFT =pow(2,next_pow_2(DataLen))  # Next power of 2 from length of y
    Yud1 = fft(traces[0:DataLen-1,3],NFFT)/DataLen
    Yud2 = fft(traces[0:DataLen-1,6],NFFT)/DataLen
    n=int(NFFT/2+1)
    f = f1/2*np.linspace(0,1,n)   #linspace(0,1,NFFT/2+1)
    fftout1 = np.array(2*abs(Yud1[1:n]))
    fftout2 = np.array(2*abs(Yud2[1:n]))
    ## 求最大值能量占比
    maxv1= np.max(fftout1)
    maxpos1=np.argmax(fftout1)
    sumv1 =np.sum(fftout1)
    rr1 = np.true_divide(maxv1,sumv1)

    maxv2= np.max(fftout1)
    maxpos2=np.argmax(fftout1)
    sumv2 = np.sum(fftout2)
    rr2 = np.true_divide(maxv2,sumv2)
    rr=rr1+rr2
    # [rr f(maxpos1) f(maxpos2)]
    #专门针对李家巷的疑似过车干扰，频率分布：3.515625, 3.90625, 4.296875hz。

    if rr>0.33 and 3.5<f[maxpos1] and f[maxpos1]<=4.3 and 3.5<f[maxpos2] and f[maxpos2]<=4.3:
        result=0
        return result,corrout


    if persistent.flag_nature == 1:
        traces_len=np.size(traces[1])
        # if traces_len>600:
        #     ind1=600
        # else:
        #     ind1=traces_len
        ret1=triger_fd_identify(traces[1],traces[0],traces[2],400)
        ret11 = triger_td_identify(traces[1],traces[0],traces[2],400)
        ret2 = triger_fd_identify(traces[4],traces[3],traces[5],400)
        ret22 = triger_td_identify(traces[1],traces[0],traces[2],400)
        ret=ret1*ret11*ret22*ret2
        if ret==0:
            print("干扰")
            result=0
            return result,corrout
    # if len(traces_org[0])>=700:
    #     print('here!')
    trainflag1 = trainidentify(traces_org[0:3])
    trainflag2 = trainidentify(traces_org[3:6])
    if persistent.flag_nature==1:
        if trainflag1==0 or trainflag2==0:
            print("过车")
            result=0
            return result,corrout
    else:
        if trainflag1 == 0 and trainflag2 == 0:
            print("过车")
            result = 0
            return result,corrout

    nlie = np.size(traces,0)
    tracs_len=np.size(traces,1)
    for j in range(nlie):
        # difacc1 = [0diff(traces(:,j))]
        # [difacc11] = DeOdd(difacc1,spanDeOddSecond*f1 )
        # difacc22 = cumtrapz(difacc11)
        traces[j]= MyFilter22(f1,traces[j],L_fs,H_fs)

    if method == 'decision_tree':
        pass
        # traces
    elif method == 'multi_parameters':
        EW=traces[0]
        NS=traces[1]
        UD=traces[2]
        lenud=len(UD)

        ## 2个通道间的相关性
        if nlie== 6 and IfCkCorr:
#             if size(traces,2)<6warning('数据的通道不到6，如果只用了一个传感器，请关闭相关性检查IfCkCorr') result=0  return
            corrEW=np.min(np.corrcoef(traces[0],traces[3]))  #  corrEW = roundn(corrEW,-2)
            corrNS=np.min(np.corrcoef(traces[1],traces[4]))  #  corrNS = roundn(corrNS,-2)
            corrUD=np.min(np.corrcoef(traces[2],traces[5]))  #  corrUD = roundn(corrUD,-2)
            UDmax1=np.argmax(traces[2])
            UDmax2=np.argmax(traces[5])
            if np.abs(UDmax1-UDmax2)<11 :#and np.max(traces[2])>0.5 and np.max(traces[5])>0.5
                _,corrEW_arry=xcorr(traces[0],traces[3],normed=True,detrend=False,maxlags=10)
                _,corrNS_arry=xcorr(traces[1],traces[4],normed=True,detrend=False,maxlags=10)
                _,corrUD_arry=xcorr(traces[2],traces[5],normed=True,detrend=False,maxlags=10)

                # corrEW_arry=corrEW_arry/(tracs_len-np.arange(-4,5,1))/(np.std(traces[0])*np.std(traces[3]))
                # corrNS_arry=corrNS_arry/(np.max(corrNS_arry))
                # corrUD_arry=corrUD_arry/(np.max(corrUD_arry))
                sum_corr=corrEW_arry+corrNS_arry+corrUD_arry
                # ind1=DataLen-10
                # ind2=DataLen+10
                ind_max=np.argmax(sum_corr)
                corrEW=corrEW_arry[ind_max]
                corrNS = corrNS_arry[ind_max]
                corrUD = corrUD_arry[ind_max]
                corrout = [corrEW,corrNS,corrUD]

            corrNum = 0
            if corrEW >= MinCorrEW: #0.8
               corrNum = corrNum+2
            elif corrEW >= MinCorrEW*0.8: #0.64
               corrNum = corrNum+1
            if corrNS >= MinCorrNS: #0.8:
               corrNum = corrNum+2
            elif corrNS >= MinCorrNS*0.8: #0.64
               corrNum = corrNum+1
            if corrUD >= MinCorrUD:  #0.9
               corrNum = corrNum+3
            elif corrUD >= MinCorrUD*0.9: #0.81
               corrNum = corrNum+2
            elif corrUD >= MinCorrUD*0.8: #0.72
               corrNum = corrNum+1

            if corrNum < 4:    # or  corrUD<0.60  ###待修改，会漏掉部分地震，如大西1104YXZ
                result=0
                if Debug==1:
                    print('-------------2个传感器间的相关系数太低--corrEW= '+ str(corrEW)\
                        +' corrNS= '+ str(corrNS) +' corrUD= ' +str(corrUD))
                return result,corrout

        ## lingjiao  20170828再次修改
        winLen = 3*f1 #待修正 3s
        if lenud>winLen:
            sum_lingjiao3 = LingJiao(traces[2],winLen)
            if int(sum_lingjiao3)<=2:
                result=0
                if Debug:
                    print('-----------3s 没有零交，非地震 ')
                return result,corrout
            sum_lingjiao3 = LingJiao(traces[1],winLen)
            if int(sum_lingjiao3)<=2:
                result=0
                if Debug:
                    print('-----------3s 没有零交，非地震 ')
                return result,corrout
            sum_lingjiao1 = LingJiao(traces[0],winLen)
            if int(sum_lingjiao1)<=2:
                result=0
                if Debug:
                    print('-----------3s 没有零交，非地震 ')
                return result,corrout

        ## 通道间的相关系数
        f11=int(f1)
        ##  触发后1s内增长太快了
        traces1S = traces[:,0:f11]#EW NS
        maxs1= max(abs(traces1S[0]))
        maxs2= max(abs(traces1S[1]))
        if maxs1>MaxHorientalFirstSecond and maxs2>MaxHorientalFirstSecond: #
            result=0
            if Debug:  print('------------- 触发后超过100gal--------------')
            return result,corrout
    elif method == 'FsortRatio':
        pass# Frequency sort ratio in C visual studio

    if row2== 6:
          if Debug:
              print('-------------2个传感器相关系数--corrEW= ' +str(corrEW)+\
              ' corrNS= '+str(corrNS)+ ' corrUD= ' +str(corrUD))
    result=1
    return result,corrout

def LingJiao(traces,winLen):
    # sum_lingjiao = 0
    # len = np.shape(traces)[0]
    # winLen=int(winLen)
    # if len<=winLen:
    #     print('error: len<=winLen')
    #     return
    # for iii in range(int(len-winLen)):
    #     data_lingjiao = traces[iii:iii+winLen-1]
    #     data_lingjiao1 = data_lingjiao[1:-2]*data_lingjiao[2:-1]
    #     ind1 = np.where(data_lingjiao1<0) # find(data_lingjiao1<0)
    #     sum_lingjiao = np.shape(ind1)[0]
    #     if sum_lingjiao<=2:  #零交点太少，非地震。
    #             # return
    #         break
    zero_crossings = np.where(np.diff(np.sign(traces)))[0]
    zero_crossings_point=np.shape(zero_crossings)[0]
    return zero_crossings_point

def MyFilter22(fs,data,L_fs,H_fs):#judge 单独滤波0.05-10hz 不同台站需改变
    H_fs = 10
    data1=data
    if L_fs<=0 and H_fs<=0:
        if data is None:
            return data
    fa = [1, -12.7951889040079, 77.0101374352978, -289.364539661362, 759.648032221234, -1477.22716725441,
          2200.91279098431, -2562.49733304384, 2356.00482235939, -1716.11080764361, 986.935584280813, -443.385164563203,
          152.531192905557, -38.8407416346212, 6.90377453458187, -0.765238184979927, 0.0398461708972303]
    fb = [3.13062686973337e-14, 5.00900299157340e-13, 3.75675224368005e-12, 1.75315104705069e-11, 5.69774090291474e-11,
          1.36745781669954e-10, 2.50700599728249e-10, 3.58143713897498e-10, 4.02911678134685e-10, 3.58143713897498e-10,
          2.50700599728249e-10, 1.36745781669954e-10, 5.69774090291474e-11, 1.75315104705069e-11, 3.75675224368005e-12,
          5.00900299157340e-13, 3.13062686973337e-14]
    filtered_data = signal.lfilter(fb, fa, data1)
    return filtered_data

def trainidentify(data):
    result, VHmax, Rud = 0, 0, 0

    UDAmp = data[2]
    EWAmp = data[0]
    NSAmp = data[1]

    ewns=np.sqrt(EWAmp ** 2 + NSAmp ** 2)
    number = 0.00001
    result_array = []
    for num in ewns:
        result_array.append(num + number)
    VHmax = np.max(np.abs(UDAmp) / result_array)  # 1.07

        # Low-pass filter
    order_low = 8
    cutoff_low = 5.0
    b_low, a_low = signal.butter(order_low, cutoff_low / (200 / 2), btype='low')
    zi = lfilter_zi(b_low, a_low)
    lowfreqSignal = signal.lfilter(b_low, a_low, UDAmp,zi=zi*UDAmp[0])
    lowfreqSignal=np.array(lowfreqSignal[0])
        # High-pass filter
    order_high = 16
    cutoff_high = 30.0
    b_high, a_high =signal.butter(order_high, cutoff_high / (200 / 2), btype='high')
    zi = lfilter_zi(b_high, a_high)
    # from matplotlib import pyplot as plt

    highfreqSignal  =signal.lfilter(b_high, a_high, UDAmp,zi=zi*UDAmp[0])
    highfreqSignal=np.array(highfreqSignal[0])
    Rud = np.sum(np.abs(highfreqSignal)) / np.sum(np.abs(lowfreqSignal))  # 6

    if persistent.flag_nature==0:
        treshRud = 7  # 同样的滤波参数，求出的Rud比MATLAB小很多，因此treshRud=7不适合
    else:
        treshRud = 4
    if VHmax >= 1.07 and Rud <= treshRud:
        result = 1  # earthquake
    else:
        result = 0  # noise

    return result


    en_ratio_ud = np.sum(UD1 ** 2) / np.sum(UD3 ** 2)
    en_ratio_ew = np.sum(EW[0:41] ** 2) / np.sum(EW ** 2)
    en_ratio_ns = np.sum(NS[0:41] ** 2) / np.sum(NS ** 2)
    indx = np.argmax(r3)
    if r2[indx] < 40 and en_ratio_ud < 0.8 and en_ratio_ew < 0.8 and en_ratio_ns < 0.8:
        ret = 1
    else:
        ret = 0
    ind_peaks = scipy.signal.find_peaks(y1)  # 返回index值，递归
    temp_peaks = np.array(y1[ind_peaks[0]])
    p = np.argsort(temp_peaks)  # index
    p2_Value = temp_peaks[p[-2]]  # Value
    p1_Value = temp_peaks[p[-1]]  # Value
    # try:
    #     p2_Value = temp_peaks[p[-2]]  # Value
    #     p1_Value = temp_peaks[p[-1]]  # Value
    # except:
    #     os.system("pause")
    if np.isnan(p).any():
        ret = 0
        return ret

    if int(np.size(p)) == 1:
        ret = 0
        return ret
    else:
        ind2 = np.where(y1 == p2_Value)  # 所对应频率索引
        ind1 = np.where(y1 == p1_Value)  # 所对应频率索引

    if frq[ind2] / frq[ind1] == 2 and frq[ind2] > 10 and frq[ind1] > 10:
        ret = 0
        return ret
    return ret

def triger_td_identify(data_n,data_e,data_u,cnt):
    ret=1
    if np.isnan(data_n).any() or np.isnan(data_e).any() or np.isnan(data_u).any():
        ret=0
        return ret
    rms_ns=np.linalg.norm(data_n)
    rms_ew=np.linalg.norm(data_e)
    rms_ud=np.linalg.norm(data_u)
    len=np.size(data_u)


    if rms_ud/rms_ew>=0.61 or rms_ud/rms_ns>=0.61 or rms_ew>8 or rms_ns>8:
        ret=1
    elif (rms_ud/rms_ew>=0.5 or rms_ud/rms_ns>=0.5) and len>1000:
        ret1=1
    else:
        ret=0
    #前100个点 用于判断爆破
    # t1=np.size(data_n)-cnt-100
    # t2=np.size(data_n)-cnt+100
    t1=200
    t2=400
    rms_ns=np.linalg.norm(data_n[t1:t2])
    rms_ew=np.linalg.norm(data_e[t1:t2])
    rms_ud=np.linalg.norm(data_u[t1:t2])
    if (rms_ud/rms_ew<0.2 or rms_ud/rms_ns<0.2) and (rms_ew<8 and rms_ns<8):
        ret=0
    return ret

def  triger_fd_identify(NS1,EW1,data_u1,cnt):
    NS=np.copy(NS1)
    EW=np.copy(EW1)
    data_u=np.copy(data_u1)
    ret=1
    sprate=200
    or_UD=np.copy(data_u)
    if np.isnan(NS).any() or np.isnan(EW).any() or np.isnan(data_u).any() or np.isnan(cnt).any():
        ret=0
        return  ret
    b = [0.2066,0.4131,0.2066]
    a = [1.0000,-0.3695,0.1958]
    zi = lfilter_zi(b, a)
    data_u = signal.lfilter(b, a, data_u, zi=zi * data_u[0])[0]
    EW = signal.lfilter(b, a, EW, zi=zi * EW[0])[0]
    NS = signal.lfilter(b, a, NS, zi=zi * NS[0])[0]
    coeffs = pywt.wavedec(or_UD, 'db7', level=2)
    threshold =np.median(np.abs(coeffs[-1]))/0.6745
    thresholded_coeffs = [pywt.threshold(coeff,threshold,mode='soft') for coeff in coeffs]
    reconstructed_signal_UD = pywt.waverec(thresholded_coeffs, 'db7')

    signal_len=np.size(data_u)
    num=0.1*signal_len
    if signal_len<sprate*3:
        num=0.2*signal_len
    elif signal_len>sprate*5:
        num=150

    t1=0
    t2=151
    t3=41
    if t2>signal_len:
        t2=signal_len
    EW=EW[t1:t1+400]
    NS=NS[t1:t1+400]
    UD=reconstructed_signal_UD[t1:t2]
    UD1=data_u[t1:t1+int(num)]
    UD3=data_u[t1:t1+400]

    [frq,y1]=Fourier(UD,int(EEW_Params.Sprate))

    [r3,r2]=scipy.signal.welch(UD1,fs=200.0,detrend=False)#需确认结果!!!!!!!!!!!!!!!!!!!!!!!
    if np.isnan(r3).any() or np.isnan(r2).any():
        ret=0
        return ret
    en_ratio_ud=np.sum(UD1**2)/np.sum(UD3**2)
    en_ratio_ew=np.sum(EW[0:41]**2)/np.sum(EW**2)
    en_ratio_ns=np.sum(NS[0:41]**2)/np.sum(NS**2)
    indx=np.argmax(r2)


    if r3[indx]<50 and en_ratio_ud<0.85 and en_ratio_ew<0.85 and en_ratio_ns<0.85:
        ret=1
    elif (t1+cnt)>1000 and en_ratio_ud<0.85 and en_ratio_ew<0.85 and en_ratio_ns<0.85:
        ret=1
    else:
        ret=0
    ind_peaks=scipy.signal.find_peaks(y1)#返回index值，递归
    temp_peaks=np.array(y1[ind_peaks[0]])
    p=np.argsort(temp_peaks)#index
    try:
        p2_Value=temp_peaks[p[-2]]#Value
        p1_Value=temp_peaks[p[-1]]#Value
    except:
       ret=1
       return ret
    if np.isnan(p).any():
        ret=0
        return ret

    if int(np.size(p))==1:
        ret=0
        return ret
    else:
        ind2=np.where(y1==p2_Value)#所对应频率索引
        ind1=np.where(y1==p1_Value)#所对应频率索引
    # try:
    if frq[ind2[0][-1]]/frq[ind1[0][-1]]==2 and frq[ind2[0][-1]]>10 and frq[ind1[0][-1]]>10:
        ret=0
        return ret
    # except:
    #     print('here')
    return ret


