#-*-coding:UTF-8 -*-
import math
import os
from StaticVar import StaticVar as persistent#静态类
import numpy as np
from StaticVar import Static_EEW_Params as EEW_Params #静态类
from function1 import MyFilter
import scipy.integrate as integrate
import scipy.signal as filter
from function1 import Fourier
import scipy
from toc_aic import TOC_AIC


def EEW_Triger(Stv_arg,k,kk,Alarm):
    #siez of input data is[6，:],input class Stv_arg output classs Sta_vars,
    #用于数据触发判断 STA/LTA触发法,此脚本为触发模块
    Debug=int(EEW_Params.Debug)
    MinDuration=float(EEW_Params.MinDuration)
    MaxEEW_times=int(EEW_Params.MaxEEW_times)
    LongestNonEqk=float(EEW_Params.LongestNonEqk)
    LTW = int(EEW_Params.LTW)
    Sprate=int(EEW_Params.Sprate)
    disAlarm=0
    L_fs=float(EEW_Params.L_fs)
    H_fs=float(EEW_Params.H_fs)
    Sta_vars=Stv_arg
    Triged=Sta_vars.Triged
    Buffer=np.copy(Sta_vars.Buffer)
    filter_Buffer=np.copy(Sta_vars.Buffer)#中值滤波后面需要添加
    StartT=float(Sta_vars.StartT)
    PackLen=int(np.shape(Sta_vars.Package)[1])
    LTA_evt=Sta_vars.LTA_evt
    N_pnsr1=persistent.N_pnsr1
    N_pnsr2=persistent.N_pnsr2
    UD=Buffer[2]
    NS=Buffer[1]
    EW=Buffer[0]
    ret=None

    if (Sta_vars.Is_EQK==1) or (Triged==1):
        if kk==1:
            persistent.P_time_Pre[0]=Sta_vars.P_time
        if kk==2:
            persistent.P_time_Pre[1]=Sta_vars.P_time

    Buffer_len=int(np.shape(Buffer)[1])
    #事件捡拾
    if Triged<1:#未触发
        # try:
        Trig_fine=MypickAIC(UD,Sprate,PackLen,filter_Buffer)#使用AIC法
        # except:
        #     os.system("pause")
        if Trig_fine<0:
            return Sta_vars,disAlarm,ret,Alarm

        #mdenoise module
        Sta_vars.identifyCnt=Buffer_len-Trig_fine
        Sta_vars.identifyFinish=1
        if Sta_vars.identifyFinish==0:
            ret=3
        if Sta_vars.identifyFinish>0 and Sta_vars.identifyCnt>=400:
            # ret1=triger_td
            ret=triger_fd_identify(NS,EW,UD,Sta_vars.identifyCnt)
            Sta_vars.identifyFinish=0
            Sta_vars.identifyCnt=0

        if Sta_vars.identifyFinish>0 and Sta_vars.identifyCnt<400:
            ret=2
        Triged=1
        Sta_vars.Triged=Triged
        Trig_time=round(StartT+(Trig_fine-Buffer_len+PackLen)/200,3)

        Sta_vars.P_time=Trig_time
        Sta_vars.End_time=-1
        Sta_vars.triger_fine=Trig_fine
        if Debug==1:
            if kk==1:
                print("---------传感器1:触发、触发、触发------------")
            else:
                print("---------传感器2:触发、触发、触发------------")

        Sta_vars.Traces_evt=Sta_vars.BaseLine[:,Trig_fine:]
        Triged_time=StartT+(Trig_fine-Buffer_len+PackLen)/Sprate
        Triged_time=np.round(Triged_time,3)
        Sta_vars.P_time=Triged_time
        Len=int(np.shape(Buffer)[0])

        Buffer[0]=MyFilter(Sprate,Buffer[0],L_fs,H_fs)#注意核查
        Buffer[1]=MyFilter(Sprate,Buffer[1],L_fs,H_fs)#注意核查
        Buffer[2]=MyFilter(Sprate,Buffer[2],L_fs,H_fs)#注意核查

        if Trig_fine<Sprate:
            noisele = Buffer[0][0:Trig_fine]
            noiseln = Buffer[1][0:Trig_fine]
            noiselz = Buffer[2][0:Trig_fine]
        else:
            noisele=Buffer[0][Trig_fine-Sprate:Trig_fine]
            noiseln=Buffer[1][Trig_fine-Sprate:Trig_fine]
            noiselz=Buffer[2][Trig_fine-Sprate:Trig_fine]
        N_pnsr=np.zeros(shape=(3,1))


        N_pnsr[0]=np.max(noisele)-np.min(noisele)
        N_pnsr[1]=np.max(noiseln)-np.min(noiseln)
        N_pnsr[2]=np.max(noiselz)-np.min(noiselz)

        if kk==1:
            persistent.N_pnsr1=N_pnsr
        if kk==2:
            persistent.N_pnsr2=N_pnsr
        return Sta_vars,disAlarm,ret,Alarm
        # Trig_Lower=Trig_fine+100
        # if Trig_Lower>Buffer_len:
        #     Trig_Lower=Buffer_len
        # theta=np.abs(UD[Trig_fine:Trig_Lower])
        # Sta_vars.theta=np.max(theta)/5
    elif Triged==1:#已判定为触发
        tempPakage=np.copy(Sta_vars.BaseLine[:,-PackLen:])
        Sta_vars.Traces_evt=np.hstack((Sta_vars.Traces_evt,tempPakage))#有问题需要改
        # print(np.shape(Sta_vars.Traces_evt)[1])
        Sta_vars.identifyCnt=Sta_vars.identifyCnt+PackLen
        if Sta_vars.identifyFinish==0:
            ret=3
        if Sta_vars.identifyFinish>0 and Sta_vars.identifyCnt>=350:
            ret=triger_td_identify(NS,EW,UD,Sta_vars.identifyCnt)
            Sta_vars.identifyFinish=0
            Sta_vars.identifyCnt=0

        if Sta_vars.identifyFinish>0 and Sta_vars.identifyCnt<350:
            ret=2
        if kk==1:
            [Pend, Sta_vars,afterPGA] = MyEnder(Sta_vars, k,N_pnsr1)
        if kk==2:
            [Pend, Sta_vars,afterPGA] = MyEnder(Sta_vars, k,N_pnsr2)
        if Sta_vars.Is_EQK<1 and int(np.shape(Sta_vars.Traces_evt)[1])>LongestNonEqk*Sprate:
            Pend=1
            if Debug==1:
                print("------传感器"+str(kk)+": traces_evt大于了15秒，还没有判断为地震， 结束——")
        if Pend>=0:
            disAlarm=1
            End_out = StartT + PackLen/Sprate #处理有问题
            Sta_vars.Duration = End_out-Sta_vars.P_time
            Sta_vars.End_time = End_out
            Sta_vars.Traces_evt=np.empty(shape=(3,0))
            if Sta_vars.Duration<MinDuration and Sta_vars.EEW_times<MaxEEW_times:
                if Debug==1:
                    print(str(Sta_vars.STA_Name)+"------------- 误报解除, 持续时间小于15秒，--------Duration="+str(Sta_vars.Duration))
                disAlarm=2
            # Sta_vars.ini2(MaxEEW_times,Alarm)
            persistent.PGAold=-1
            persistent.AZIold=-1
            persistent.distold=-1
        return Sta_vars,disAlarm,ret,Alarm
##############################################################################下面为子函数
def MypickAIC(Data,Sprate,PackLen,filter_Buffer):
    #触发数据外围处理及判断
    Back=int(EEW_Params.Back)
    Fowrd=int(EEW_Params.Fowrd)
    thresh=float(EEW_Params.thresh)
    STW=float(EEW_Params.STW)
    LTW=float(EEW_Params.LTW)
    ifliter=int(EEW_Params.iflilter)
    MinThresh=float(EEW_Params.MinThresh)
    Trig_fine=-1
    Trigraw=-1
    L_fs=float(EEW_Params.L_fs)
    H_fs=float(EEW_Params.H_fs)
    len=int(np.shape(Data)[0])
    # MinThresh=0#调试用
    if np.max(np.abs(Data[int(LTW*Sprate):]))>MinThresh and np.max(np.abs(Data[int(LTW*Sprate):]))>np.max(np.abs(Data[0:int(LTW*Sprate)])):
        Trigraw=MyPicker(Data,Sprate,1,thresh,STW,LTW,ifliter,PackLen)
    if Trigraw==-1:
        return Trig_fine
    if ifliter>0:
        DataF=MyFilter(Sprate,Data,L_fs,H_fs)
    PUpper=1
    Plower=Trigraw+Fowrd*Sprate
    if Plower>len:
        Plower=len
    Trig_fine=FinePick(DataF,1,PUpper,Plower,LTW*Sprate)
    return Trig_fine

def MyPicker(acc,sprate,phase,thresh,STW,LTW,iflilter,PackLen):
    #phase:1(P波) 2(S波)
    len=np.size(acc)
    difacc=acc
    #difacc=np.append(0,np.diff(acc))
    # difacc=np.hstack(([[0],difacc]))
    spanDeoddSecond=float(EEW_Params.spanDeOddSecond)
    L_fs=float(EEW_Params.L_fs)
    H_fs=float(EEW_Params.H_fs)
    trig=-1
    difacc2=DeOdd(difacc,spanDeoddSecond*sprate)
    num=np.size(difacc2)
    #x = np.linspace(1,num,num)
    Facc=integrate.cumtrapz(difacc2,initial=0)
    #Facc=Facc-np.mean(Facc)
    if iflilter>0:
        Facc=MyFilter(sprate,Facc,L_fs,H_fs)
    CF=(Facc+0.01)**2
    Lta=np.mean(CF[:int(LTW*sprate)])
    if phase==1:
        Pbegin=int(len-PackLen)
        tempnum=int(PackLen+1)
        for i in range(tempnum):
            Sta=np.mean(CF[int(Pbegin-STW*sprate+i):int(Pbegin+i)])
            # print(Sta/Lta)
            if Sta/Lta>thresh:
                trig=i+Pbegin
                break

        if trig==-1:
            temp=Facc[Pbegin:]
            ind=np.where(temp>10)#需要更改
            if np.size(ind)==0:
                trig=-1
                return trig
            else:
                trig=Pbegin+ind[0][0]
                return trig
        else:
            return trig
    # elif phase==2:
    #      trig=MyPickerS(Facc,sprate)
    #      return trig
def DeOdd(data,span):
    M=5
    NonOdd=data
    num=int(math.floor(int(np.size(data))/span))
    for k in range(num):
        WaveSpan=NonOdd[int(k*span):int((k+1)*span)]
        V1=np.max(np.abs(WaveSpan))
        T1=np.argmax(np.abs(WaveSpan))
        WaveSpan[T1]=0
        V2=np.max(np.abs(WaveSpan))
        T2=np.argmax(np.abs(WaveSpan))
        WaveSpan[T2]=0
        if V1>M*(np.max(np.abs(WaveSpan))):
            NonOdd[int(k*span+T1)]=-np.sum(NonOdd[int(k*span+T1-1)])
        if V2>M*(np.max(np.abs(WaveSpan))):
            NonOdd[int(k*span+T2)]=-np.sum(NonOdd[int(k*span+T2-1)])
    return NonOdd




def MyPickerS(trace_input,sprate):
    #  N    = 40;       % Order
    # Fc   = 10;       % Cutoff Frequency
    # flag = 'scale';  % Sampling Flag
    # blackmanharris(N+1)
    # S波拾取 fir 带窗fir 滤波
    # S波拾取模块
    trace=np.copy(trace_input)
    len=np.shape(trace)[1]
    row1=np.shape(trace)[0]
    trigS=-1#output
    delt_S=0
    f1=sprate
    if len<=6*sprate:
        return trigS
    if row1!=3:
        print("MyPickerS数据有"+str(row1)+"列")
        return trigS
    trace11=FilterS(f1,trace[0])
    trace22=FilterS(f1,trace[1])
    trace33=FilterS(f1,trace[2])
    trace1=np.vstack([trace11,trace22,trace33])
    EW1=trace11
    NS1=trace22
    if np.shape(trace1)[1]/f1>30:
        EW1=np.copy(EW1[10*f1+1:])
        NS1=np.copy(NS1[10*f1+1:])
        delt_S=10*f1
    elif np.shape(trace1)[1]/f1>15:
        EW1=np.copy(EW1[6*f1+1:])
        NS1=np.copy(NS1[6*f1+1:])
        delt_S=6*f1
    elif np.shape(trace1)[1]/f1>5:
        EW1=np.copy(EW1[3*f1+1:])
        NS1=np.copy(NS1[3*f1+1:])
        delt_S=3*f1

    CF_EW1=EW1[1:-1]**2-EW1[0:-2]*EW1[2:]
    CF_NS1=NS1[1:-1]**2-NS1[0:-2]*NS1[2:]
    #np.linalg.
    if np.sum(CF_EW1)==0 or np.sum(CF_NS1)==0:
        print("水平方向值为为零")
        return trigS
    [TrigS_EW,delt_AIC_EW]=TOC_AIC(CF_EW1)
    [TrigS_NS,delt_AIC_NS]=TOC_AIC(CF_NS1)
    if TrigS_EW==-1 or TrigS_NS==-1 or abs(TrigS_EW-TrigS_NS)>np.shape(EW1)[0]*0.1 :
        return trigS

    if max(abs(EW1[0:TrigS_EW]))>max(abs(EW1[TrigS_EW+1:])) or max(abs(NS1[0:TrigS_NS]))>max(abs(NS1[TrigS_NS+1:])):
        return trigS
   
    if (delt_AIC_EW<2000 or delt_AIC_NS<2000) and TrigS_EW<np.shape(EW1)[0]/2 and TrigS_NS<np.shape(NS1)[0]/2:
        trigS=-1
        return  trigS

    EW1_max=max(abs(EW1))
    NS1_max=max(abs(NS1))
    if EW1_max>NS1_max:
        PGA_EN=EW1_max
    else:
        PGA_EN=NS1_max
    PGA_UD=max(abs(trace33))
    if PGA_EN<PGA_UD:
        trigS=-1
        return trigS
    trigS=np.fix((TrigS_NS+TrigS_EW)/2)+delt_S
    if trigS/f1<2:
        trigS=-1
        return trigS
    print("S波到了")
    return trigS


def triger_td_identify(data_n,data_e,data_u,cnt):
    ret=1
    if np.isnan(data_n).any() or np.isnan(data_e).any() or np.isnan(data_u).any():
        ret=0
        return ret
    rms_ns=np.linalg.norm(data_n)
    rms_ew=np.linalg.norm(data_e)
    rms_ud=np.linalg.norm(data_u)
    if rms_ud/rms_ew>=0.61 or rms_ud/rms_ns>=0.61 or rms_ew>8 or rms_ns>8:
        ret=1
    else:
        ret=0
    #前100个点 用于判断爆破
    t1=np.size(data_n)-cnt-100
    t2=np.size(data_n)-cnt+100
    rms_ns=np.linalg.norm(data_n[t1:t2])
    rms_ew=np.linalg.norm(data_e[t1:t2])
    rms_ud=np.linalg.norm(data_u[t1:t2])
    if (rms_ud/rms_ew<0.3 or rms_ud/rms_ns<0.2) and (rms_ew<8 and rms_ns<8):
        ret=0
    return ret

def triger_fd_identify(NS,EW,data_u,cnt):
    ret=1
    if np.isnan(NS).any() or np.isnan(EW).any() or np.isnan(data_u).any() or np.isnan(cnt).any():
        ret=0
        return  ret
    t1=int(np.size(data_u)-cnt)
    if t1<0:
        t1=0
    t2=int(np.size(data_u)-cnt+150)
    t3=int(np.size(data_u)-cnt+40)
    if t2>int(np.size(data_u)):
        t2=int(np.size(data_u))

    EW=EW[t1:t1+400]
    NS=NS[t1:t1+400]
    UD=data_u[t1:t1+400]
    UD1=data_u[t1:t2]
    UD3=data_u[t1:t1+400]
    [frq,y1]=Fourier(UD,int(EEW_Params.Sprate))

    [r3,r2]=scipy.signal.welch(UD1,fs=200.0,detrend=False)#需确认结果!!!!!!!!!!!!!!!!!!!!!!!
    if np.isnan(r3).any() or np.isnan(r2).any():
        ret=0
        return ret
    en_ratio_ud=np.sum(UD1**2)/np.sum(UD3**2)
    en_ratio_ew=np.sum(EW[0:41]**2)/np.sum(EW**2)
    en_ratio_ns=np.sum(NS[0:41]**2)/np.sum(NS**2)
    indx=np.argmax(r3)
    if r2[indx]<40 and en_ratio_ud<0.8 and en_ratio_ew<0.8 and en_ratio_ns<0.8:
        ret=1
    else:
        ret=0
    ind_peaks=scipy.signal.find_peaks(y1)#返回index值，递归
    temp_peaks=np.array(y1[ind_peaks[0]])
    p=np.argsort(temp_peaks)#index
    # try:
    p2_Value=temp_peaks[p[-2]]#Value
    p1_Value=temp_peaks[p[-1]]#Value
    # except:
    #     os.system("pause")
    if np.isnan(p).any():
        ret=0
        return ret

    if int(np.size(p))==1:
        ret=0
        return ret
    else:
        ind2=np.where(y1==p2_Value)#所对应频率索引
        ind1=np.where(y1==p1_Value)#所对应频率索引

    if frq[ind2]/frq[ind1]==2 and frq[ind2]>10 and frq[ind1]>10:
        ret=0
        return ret
    return ret

def MyEnder(Stv_arg,k,N_pnsr):
    Sta_vars=Stv_arg
    MaxDur=float(EEW_Params.MaxDur)
    debug=int(EEW_Params.Debug)
    ForceEndThresh=float(EEW_Params.ForceEndThresh)
    Pend=-1
    afterPGA=-1
    L_fs=float(EEW_Params.L_fs)
    H_fs=float(EEW_Params.H_fs)
    pins=float(EEW_Params.pins)
    ratio=float(EEW_Params.ratio)
    DUR=Sta_vars.StartT-Sta_vars.P_time
    Buffer=np.copy(Sta_vars.Buffer)
    M=np.max(Buffer)
    Sprate= int(EEW_Params.Sprate)
    Traces_evt=np.copy(Sta_vars.Traces_evt)
    len=np.size(Traces_evt[0])

    if len<(2.5*Sprate):
        return Pend,Sta_vars,afterPGA
    if DUR>MaxDur:
        Pend=1
        if debug==1:
            print("----------持时大于300秒了and MAX BUFFER<2GAL，强制结束 "+str(DUR)+str(M))
        return Pend,Sta_vars,afterPGA
    if len>Sprate:
        Span1s1=np.copy(Sta_vars.Buffer[0][-2*Sprate:])
        Span1s2=np.copy(Sta_vars.Buffer[1][-2*Sprate:])
        Span1s1=Span1s1-np.mean(Span1s1)
        Span1s2=Span1s2-np.mean(Span1s2)
        if np.max(Span1s1)<ForceEndThresh and np.max(Span1s2)<ForceEndThresh:
            Pend=1
            if debug==1:
                print("---------去除基线的水平2通道的最近2秒幅值已经小于"+str(ForceEndThresh)+"gal， 结束 ")
            return Pend,Sta_vars,afterPGA
        if np.isnan(Sta_vars.PGA_Curr):
            return Pend,Sta_vars,afterPGA
        if CkSteady(Buffer[0][-10*Sprate:],ratio,pins) and CkSteady(Buffer[1][-10*Sprate:],ratio,pins) and CkSteady(Buffer[2][-10*Sprate:],ratio,pins)\
        and M<Sta_vars.PGA_Curr/10:
            Pend=1
            if debug==1:
                print("--------Buffer后10秒 Steady了, Buffer max小于PGA/10 或者小于3.5gal then end -------------")
            return Pend,Sta_vars,afterPGA
        if len/Sprate<10:
            return Pend,Sta_vars,afterPGA
        Traces_evt[0]=MyFilter(Sprate,Traces_evt[0],L_fs,H_fs)
        Traces_evt[1]=MyFilter(Sprate,Traces_evt[1],L_fs,H_fs)
        Traces_evt[2]=MyFilter(Sprate,Traces_evt[2],L_fs,H_fs)
        ple=Traces_evt[0][-Sprate:]
        pln=Traces_evt[1][-Sprate:]
        plz=Traces_evt[2][-Sprate:]
        EW_pnsr=np.max(ple)-np.min(ple)
        NS_pnsr=np.max(pln)-np.min(pln)
        UD_pnsr=np.max(plz)-np.min(plz)
        if EW_pnsr/N_pnsr[0]<3 and NS_pnsr/N_pnsr[1]<3 and UD_pnsr/N_pnsr[2]<3:
            Pend=1
            if debug==1:
                print('--------信噪比低，结束结束 ---------')
            return Pend,Sta_vars,afterPGA
        Traces_value=np.sqrt(Traces_evt[0]**2+Traces_evt[1]**2+Traces_evt[2]**2)
        PGA=np.max(Traces_value)
        PGA_end=np.max(Traces_value[-Sprate*3+1:])
        if (PGA_end/PGA)<0.1:
            Pend=1
            if debug==1:
                print("--------事件大幅衰减，结束结束 ---------")
            return Pend,Sta_vars,afterPGA
    return Pend,Sta_vars,afterPGA

def FinePick(Data,method,N1,N2,LTWlen):
    #AIC 池则信息法自回归突变点判断
    #method=1 AIC
    #metho=2 BIC
    method=1
    if method==1:
        N=N2-N1
        AIC2=np.zeros(N)
        if N1<=0:
            N1=1
        Data_m2=Data[N1-1:N2+1]*Data[N1-1:N2+1]
        Data_m1=Data[N1-1:N2+1]
        total_m1=np.sum(Data_m1[0:N+1])#AIC段全部数据和
        total_m2=np.sum(Data_m2[0:N+1]) #AIC段全部数据平方和
        temp_m1=Data_m1[0]
        temp_m2=Data_m2[0]
        trig=-1
        if temp_m1==0:
            temp_m1=0.0000000005

        if temp_m2==0:
            temp_m2=0.0000000005
        for k in range(N-2):
            kk=k+1
            temp_m1=temp_m1+Data_m1[kk]
            temp_m2=temp_m2+Data_m2[kk]
            temp1=temp_m2/(kk+1)-(temp_m1/(kk+1))**2
            temp2=(total_m2-temp_m2)/(N-k-1)-((total_m1-temp_m1)/(N-k-1))**2
            test1=np.var(Data[0:kk])
            # plt.figure()
            # plt.plot(Data_m1)
            try:
                ls1=math.log10(temp1)
                ls2=math.log10(temp2)
                AIC2[k]=(kk+1)*ls1.real+(N-k-1)*ls2.real
            except:
                print("Aic 异常 EEW_Triger，FinePick")
        AIC_min=np.min(AIC2[2:-1])# confirm
        Trig=np.argmin(AIC2[2:-1])
        Trig=Trig+N1+1

        # plt.figure()
        # plt.plot(AIC2)

        if Trig<LTWlen:
            Trig==1
            return Trig
        a0=Data[Trig]
        for i in range(10):
            if np.abs(Data[Trig-1])<np.abs(a0):
                if Data[Trig-1]*a0>=0:
                    a0=Data[Trig-1]
                    Trig=Trig-1
                else:
                    Trig=Trig-1
                    break
            else:
                break
        if AIC2[2]-AIC_min<800:
            Trig=-1
        Trig=Trig-20
        return Trig
def CkSteady(data,ratiomax,pins):
    result=False
    len=np.size(data)
    len4=int(np.floor(len/pins))
    pins=int(pins)
    MM=np.zeros((pins,1))
    for i in range(pins):
        temp=np.abs(data[1+(i)*len4:(i+1)*len4+1])
        MM[i]=np.max(temp)
    np.sort(MM)
    ratio1=MM[0]/MM[-1]
    ratio2=MM[-2]/MM[-1]
    if ratio1>ratiomax and ratio2>0.99:
        result=True
        return result
    return result

def FilterS(Fs,Z):
    Z1=np.copy(Z)
    len=int(np.size(Z1))
    if len<256:
        y=0
        return y
    b=[-2.49263027857326e-22,-2.37225438680822e-06,-1.98420583459990e-05,-7.90328729629732e-05,-0.000221450490630881,-0.000491564910530530,
       -0.000903145496537876,-0.00138414413073868,-0.00171166539125376,-0.00146094258661226,9.03453844468861e-19,0.00344148081494769,
       0.00961813769495998,0.0190483797807789,0.0317785055639675,0.0472049797160798,0.0640282344401041,0.0803835826942116,0.0941448519812235,0.103339698229090,
       0.106572618553273,0.103339698229090,0.0941448519812235,0.0803835826942116,0.0640282344401041,0.0472049797160798,0.0317785055639675,0.0190483797807789,
       0.00961813769495999,0.00344148081494770,9.03453844468861e-19,-0.00146094258661226,-0.00171166539125376,-0.00138414413073868,-0.000903145496537877,
       -0.000491564910530531,-0.000221450490630881,-7.90328729629737e-05,-1.98420583459984e-05,-2.37225438680844e-06,-2.49263027857326e-22]
    a=1
    y =filter.lfilter(b,a,Z1) #41 为 a、b数组最大值
    #y=filter.filtfilt(b,1,Z1)
    return y