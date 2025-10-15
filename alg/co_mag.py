#-*-coding:UTF-8 -*-
import os
import numpy as np
import math
from function1 import Rdelta,iomega  # 73*2,ndarray

def cmp_mag(Traces_evt_in,Distance,beM,Sprate):
    # in
    # Traces_evt: 6*n,ndarray,float
    # Distance: 1*1,ndarray,float
    # beM: 1*1,ndarray,float
    # Sprate:1*1,ndarray,int
    Traces_evt=np.copy(Traces_evt_in)
    result = -1 # Mag初值
    if Distance < 0:
        return result
    num = np.size(Traces_evt,0)
    num1 = np.size(Traces_evt, 1)
    # 获取一个长度为num1的汉明窗，ndarray,1*nf
    windows = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (num1 - 1)) for n in range(num1)])
    for ii in range(num):
        Traces_evt[ii,:] = Traces_evt[ii,:]*windows  # 6*n

    Distance = np.round(Distance,1)
    if num1 >= 500:
        displace1 = iomega(Traces_evt[2,:],Sprate,2)
        maxdisp1 = max(abs(displace1[int(0.1*Sprate-1):int(2.5*Sprate)]))
        magnitude11 = (math.log(maxdisp1*10000,10)+1.2*math.log(Distance,10)+5.0*10**-4*Distance-5.0*10**-3*10+0.46)/0.78
        mag_tc1, tc = TaoC(Traces_evt[2,0:int(2.5*Sprate)], Sprate, 'sc')
        # temp = magnitude11
        # Magnitude1 = min(temp)
        Magnitude1 = magnitude11
        # tempM = mag_tc1
        # minM = min(tempM)
        minM = mag_tc1

        #第二个传感器的
        displace2 = iomega(Traces_evt[5,:], Sprate, 2)
        maxdisp2 = max(abs(displace2[int(0.1 * Sprate): int(2.5 * Sprate)]))
        magnitude22 = (math.log(maxdisp2*10000,10) + 1.2 * math.log(
            Distance,10) + 5.0*Distance*10**-4  - 5.0 *10*10**-3 + 0.46) / 0.78
        mag_tc2, tc = TaoC(Traces_evt[5,0:int(2.5*Sprate)], Sprate, 'sc')
        temp = [magnitude11, magnitude22]
        Magnitude1 = min(temp)
        tempM = [mag_tc1, mag_tc2]
        minM = min(tempM)
        Magnitude2 = minM
    else:
        displace1 = iomega(Traces_evt[2,:], Sprate, 2)
        # displace1 = np.array(displace1)
        lld = len(displace1)
        maxdisp1 = max(abs(displace1[int(0.1*Sprate-1-lld):]))
        Magnitude11 = (math.log(maxdisp1*10000,10) + 1.2 * math.log(
            Distance,10) + 5.0 * Distance * 10**-4 - 5.0 * 10* 10**-3  + 0.46) / 0.78
        # temp = Magnitude11
        # Magnitude1 = min(temp)
        Magnitude1 =Magnitude11
        [mag_tc1, tc] = TaoC(Traces_evt[2,:], Sprate, 'sc')
        # tempM = [mag_tc1]
        # minM = min(tempM)
        # Magnitude2 = minM
        Magnitude2 = mag_tc1
        if num > 3:  # double sensor
            displace2 = iomega(Traces_evt[5,:], Sprate, 2)
            maxdisp2 = max(abs(displace2[int(0.1*Sprate-1-lld):]))
            Magnitude22 = (math.log(maxdisp2 * 10000,10) + 1.2 *math.log(
                Distance,10) + 5.0 * Distance* 10**-4  - 5.0* 10 * 10**-3  + 0.46) / 0.78

            temp = np.array([Magnitude11, Magnitude22])
            Magnitude1 = min(temp)
            [mag_tc2, tc] = TaoC(Traces_evt[5,:], Sprate, 'sc')
            tempM = np.array([mag_tc1, mag_tc2])
            minM = min(tempM)
            Magnitude2 = minM

    MagnitudeFinal1 = 0.5*(Magnitude1+Magnitude2) # P波预测震级

###############################################################
    displaceUD1 = iomega(Traces_evt[2,:], Sprate, 2)
    displaceUD2 = iomega(Traces_evt[5,:], Sprate, 2)
    lld = len(displaceUD1)
    maxdispUD1 = max(abs(displaceUD1[int(0.1*Sprate-1-lld):]))
    maxdispUD1 = maxdispUD1 * 10000
    maxdispUD2 = max(abs(displaceUD2[int(0.1*Sprate-1-lld):]))
    maxdispUD2 = maxdispUD2 * 10000
    maxA1 = maxdispUD1
    maxA2 = maxdispUD2
    # mmva,ind = min(find(Distance < Rdelta[:, 1]))
    rr = Rdelta[:, 0]
    ind = np.argwhere(np.array(rr) > Distance).min()
    ind1 = ind - 1
    if ind1 <= 0:
        ind1 = 1
    
    delta = Rdelta[ind1, 1]
    if ind.size == 0:
        delta = Rdelta[-1, 1]
    
    Magnitude111 = math.log(maxA1,10) + delta
    Magnitude222 = math.log(maxA2,10) + delta
    MagnitudeFinal2 = fitM(np.true_divide(Magnitude111 + Magnitude222,2)) # 规范计算震级 # 加入3项拟合
    
    if MagnitudeFinal1 > MagnitudeFinal2:
        # disp(['预测震级:'  num2str(MagnitudeFinal1), '  计算震级:', num2str(MagnitudeFinal2)])
        result = np.round(MagnitudeFinal1, 1)
        if beM > result:
            result = beM

            return result
        # disp('预测震级p')
        return result
    else:
        # # 规范计算震级
        result = np.round(MagnitudeFinal2, 1)
        if np.round(MagnitudeFinal1, 1) >= 4.3: # 滤波器切换
            displaceUD1 = iomega(Traces_evt[2,:], Sprate, 2)
            displaceUD2 = iomega(Traces_evt[5,:], Sprate, 2)
            maxdispUD1 = max(abs(displaceUD1[int(0.1 * Sprate):]))
            # try:
            #     maxdispUD1 = max(abs(displaceUD1[int(0.1*Sprate):]))
            # except:
            #     os.system("pause")
            maxdispUD1 = maxdispUD1 * 10000
            maxdispUD2 = max(abs(displaceUD2[int(0.1*Sprate):]))
            maxdispUD2 = maxdispUD2 * 10000
            maxA1 = maxdispUD1
            maxA2 = maxdispUD2
            rr = Rdelta[:, 0]
            rr[rr < Distance]=10000
            sorted_index = rr.argsort() #up
            ind = sorted_index[0]  # rr.index(min(rr))  # list
            ind1 = ind - 1
            ind1=int(ind1)
            if ind1 < 0:
                ind1 = 0
            delta = Rdelta[ind1, 1]
            if rr[ind] == 10000:  # isempty(ind):
                delta = Rdelta[-1, 2]

            Magnitude111 = math.log(maxA1,10) + delta
            Magnitude222 = math.log(maxA2,10) + delta
            # MagnitudeFinal2 = (Magnitude111 + Magnitude222) / 2
            m2 = np.true_divide((Magnitude111 + Magnitude222),2)
            MagnitudeFinal2 = fitM(m2)  # 加入3项拟合
            result = round(MagnitudeFinal2, 1)

        # disp(['预测震级:'  num2str(MagnitudeFinal1), '  计算震级:', num2str(MagnitudeFinal2)])
        if beM > result:
            # disp('计算震级前一报')
            result = beM
        return result
    

def TaoC(a,sprate,area):
    # INPUT:
    # a, 加速度记录,竖直向
    # sprate
    # area: 'sc' southern california
    # OUTPUT:
    # M, magnitude
    # tc, taoC

    M = 0
    tc = 0
    v = iomega(a, sprate, 1)
    d = iomega(a, sprate, 2)
    # v = np.array(v)
    # d = np.array(d)
    r = np.sum(v**2)/np.sum(d**2)
    r1 = math.sqrt(r)
    if not float(r1):  # ~isreal(r1):
        r1=1
    tc = np.true_divide(2 * math.pi, r1)
    # M = p2m(tc, 'TaoC', area)  #
    # 下面行取代p2m函数
    M = 0.4046 * tc * tc + 1.2767 * tc + 2.7713  # 3s效果很好
    return M,tc



def fitM(x):
    result = 0
    # p1 = 0.01518
    # p2 = -0.1842
    # p3 = 1.728
    # p4 = -1.373
    p1 = -0.04941  # (-0.0936, -0.005222)
    p2 = 0.7157  # (0.09173, 1.34)
    p3 = -2.211  # (-5.006, 0.5832)
    p4 = 4.243  # (0.2474, 8.239)
    result = p1 * x**3 + p2 * x**2 + p3 * x + p4
    return result


