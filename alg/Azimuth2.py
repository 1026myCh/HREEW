# -*-coding:UTF-8 -*-
import math
import os

import scipy.signal as filter
# import PCA
from scipy.fftpack import fft, ifft

from function1 import iomega_azi1
from srs_integration import srs_integration, srs_integrationV


def Azimuth2(datain):
    # in:
    # datain:3*n or 6*n
    sprate = 200
    len1=int(np.shape(datain)[1])
    if len1>600:
        tempData = np.array(datain[:][0:401])
    else:
        tempData = np.array(datain)



    dataout10 = iomega_azi1(tempData[0],200,1)#给李慧改
    dataout11 = iomega_azi1(tempData[1], 200, 1)#
    dataout12 = iomega_azi1(tempData[2], 200, 1)#
    b,a=filter.butter(4,20,'lp',fs=200)
    # b = [1.41272653624677e-07, 0, -5.65090614498706e-07, 0, 8.47635921748059e-07,
    #      0, -5.65090614498706e-07, 0, 1.41272653624677e-07]
    # a = [1, -7.89154014671955, 27.2523197336544, -53.7909309665033,
    #      66.3737957621409, -52.4283619864010, 25.8891681521142,
    #      -7.30691593562765, 0.902465387346559]
    dataout10 = filter.filtfilt(b, a, tempData[0])
    dataout11 = filter.filtfilt(b, a, tempData[1])
    dataout12 = filter.filtfilt(b, a, tempData[2])
    # dataout10 = iomega_azi1(tempData[0],200,1)#给李慧改
    # dataout11 = iomega_azi1(tempData[1], 200, 1)#
    # dataout12 = iomega_azi1(tempData[2], 200, 1)#

    # dataout10=  inter.solve_ivp(dataout10,(0,len1),[1],dense_output=True)
    # dataout11 = inter.solve_ivp(dataout11)
    # dataout12 = inter.solve_ivp(dataout12)

    # dataout10 = inter.cumtrapz(dataout10)
    # dataout11 = inter.cumtrapz(dataout11)
    # dataout12 = inter.cumtrapz(dataout12)


    # print(np.size(tempData))
    dataout1 = [dataout10,dataout11,dataout12] #(3, 403)
    dataout1 = np.array(dataout1) # size:3*n

    #dataout1 = dataout1.T #（3，n）
    lentemp = tempData.shape[1] # 只输出行数
    PointEnd = lentemp
    numZero = 0
    PointBeg = 0

        # for k in range(0, lentemp-1, 1):
        #     dk = dataout1[2,k]
        #     dk1 = dataout1[2,k+1]
        #     if np.true_divide(dk, dk1) < 0:
        #         numZero = numZero+1
        #         if numZero == 1:
        #             PointBeg = k-1
        #         if numZero == 2:
        #             PointEnd = k
        #
        # absP = abs(PointEnd-PointBeg)
        # if absP<30:
        #     PointEnd = PointBeg
        #     PointBeg = 0
        # if PointBeg<1:
        #     PointBeg = 0
        # if PointBeg == PointEnd:
        #     dp = dataout1[PointBeg, :]
        # else:
        #     dp = dataout1[0:3,PointBeg:PointEnd]
        # flagd = np.shape(dp)
        # if len(dp)<1:
        #     PointBeg = 1
        #     PointEnd = lentemp
        # if PointEnd<PointBeg:
        #     PointEnd=400
        #     PointBeg=0
    try:
        [ind1,ind2]=first_half_wave_length(dataout1[2])
    except:
        ind1=0
        ind2=1*sprate
    # Azi2 = MyAzimuth1(dataout1[0:3,PointBeg:PointEnd+1],'PCA',sprate)


    # ind1=ind1-20
    # ind2=ind2+40
    if ind1<0:
        ind1=0
    elif ind2>lentemp:
        ind2=lentemp-2
    if ind2 - ind1 < 10:
        ind2=ind1+200
    Azi2 = MyAzimuth1(dataout1[0:3, ind1:ind2], 'PCA', sprate)
    #result = Azi2
    return Azi2

# trace 为外部进行了带通滤波的位移值

def MyAzimuth1(traces, method, sprate):
    tracesD = traces
    result = 0
    azi = -10000
    ANtrace  =  traces[0]
    AEtrace  =  traces[1]
    AZtrace  =  traces[2]

    if method  ==  'PCA':
        azi = -10000
        primary,ddout = pca(tracesD)
        # pca = PCA(n_components = 1)
        # primary  =  pca.fit_transform(tracesD.transpose())
        azi = pca2azi(primary) # check if zero!
        result = azi
    elif method == 'FM':
        pass

    return result



# primaryin：行列值与matlab输入一致，3*3
def pca2azi(primaryin):
    azi = 0
    e = primaryin[0,0]
    n = primaryin[1,0]
    u = primaryin[2,0]
    azi = 180/math.pi*math.atan(abs(np.true_divide(e, n)))
    if (u<0 and n>0 and e>0)or (u>0 and n<0 and e<0):
        pass
    elif (u<0 and n<0 and e>0)or (u>0 and n>0 and e<0):
        azi = 180-azi
    elif (u<0 and n<0 and e<0)or (u>0 and n>0 and e>0):
        azi  =  180 + azi
    elif (u<0 and n>0 and e<0)or (u>0 and n<0 and e>0):
        azi  =  360 - azi
    return azi


# def MyAzimuth(traces ,method,sprate,*args):
#     # 暂时用不到，未调试
#     # 确定窗长
#     tracesA  =  traces
#     result = []
#     azi = []
#     if len(*args) < 1:
#         tracesD = iomega(tracesA[:, 2], sprate, 2)
#         # ind  =  find(diff(sign(diff(abs(tracesD(:, 1)))))   ==   -2)+1  # 极大值位置
#         tr_diff = np.diff(abs(tracesD[:, 0], n=1, axis=-1))
#         tr_diff2 = np.diff(np.sign(tr_diff), n=1, axis=-1)
#         tr_diff2 = list(tr_diff2)
#         ind = tr_diff2.index(-2)
#
#         windowlen = 0
#         if len(ind) < 1:  #Sized
#             windowlen = len(traces) - 1
#         else:
#             ind = 1 + ind
#             V = tracesD[ind-1]
#             (maxv, posv) = max(abs(V))
#             ind2 = V[abs(V) > maxv / 10]  # 最大极值得1 / 10以上，meaningful
#             windowlen = ind[ind2[1]] - 1
#
#     tracesD = [iomega(tracesA[1:1 + windowlen, 0], sprate, 2), iomega(tracesA[1: 1 + windowlen, 1], sprate, 2), iomega(
#         tracesA[1: 1 + windowlen, 2], sprate, 2)] # 由加速度计算到位移
#
#     AEtrace = iomega(tracesA[1:1 + windowlen, 0], sprate, 2)
#     ANtrace = iomega(tracesA[1:1 + windowlen, 1], sprate, 2)
#     AZtrace = iomega(tracesA[1:1 + windowlen, 2], sprate, 2)
#     if method == 'PCA':
#         # k循环为1，不如直接k = 1
#         # [coeff, score, latent] = pca(___)
#         # coeff: X矩阵所对应的协方差矩阵V的所有特征向量组成的矩阵，即变换矩阵或投影矩阵，coeff每列代表一个特征值所对应的特征向量，列的排列方式对应着特征值从大到小排序。
#         # score: 表示原数据在各主成分向量上的投影。但注意：是原数据经过中心化后在主成分向量上的投影。score每行对应样本观测值，每列对应一个主成份(变量)，它的行和列的数目和X的行列数目相同。
#         # latent: 是一个列向量，主成分方差，也就是各特征向量对应的特征值，按照从大到小进行排列。(简洁点说就是X 所对应的协方差矩阵的特征值，latent=diag(cov(score)))
#         [vout, eout, dout] = pca(tracesD[1:1 + windowlen,:])
#         primary = eout[:, 1] # primary component
#         # pca  =  PCA(n_components = 1)
#         # primary  =  pca.fit_transform(tracesD.transpose())
#         ## 待完善PCA！！！！！！！！！！！！！！！！
#         azi = pca2azi(primary)
#         result  =  azi
#     elif method   ==    'FM':
#         pass
#     elif method   ==   'SV':
#         #  选择水平最大点
#         #  We, therefore, choosethe data point corresponding
#         # to the highest horizontal amplitude withinth  n1–n2 data interval.
#
#         [maxn, ind]  =  max(abs(ANtrace))
#         [maxae, ]  =  max(abs(AEtrace))
#         [maxan,  ]  =  max(abs(ANtrace))
#         if maxae > maxan:
#             [mm, ind] = max(abs(AEtrace))
#         AE = AEtrace[ind]
#         AN = ANtrace[ind]
#         AZ = AZtrace[ind]
#         fff  =  math.atan(np.true_divide(AE,AN))
#
#         result  =  np.true_divide(fff * 180, math.pi)
#     elif method  ==  'MA':
#         RZE = 0
#         RZN = 0
#         alpha = 0.99
#         for ii in range(windowlen):
#             AE  =  AEtrace[ii]
#             AN  =  ANtrace[ii]
#             AZ  =  AZtrace[ii]
#             RZE  =  alpha * RZE + AZ * AE
#             RZN  =  alpha * RZN + AZ * AN
#         F  =  math.atan(np.true_divide(RZE, RZN)) + math.pi
#         if RZN < 0:
#             F  =  F+math.pi
#         result = F * np.true_divide(180,math.pi)
#     return result


def filter_matlab(b,a,x):
    # 供iomega_azi调用
    y  =  []
    y.append(b[0] * x[0])
    for i in range(1,len(x)):
        y.append(0)
        for j in range(len(b)):
            if i >= j:
                y[i]  =  y[i] + b[j] * x[i - j]
                j +=  1
        for l in range(len(b)-1):
            if i > l:
                y[i]  =  (y[i] - a[l+1] * y[i -l-1])
                l +=  1
        i +=  1
    return y

# 共多出调用iomega和iomega1，在CoMagnitude中调试
def iomega(datain, sprate, xx):
    # 计算速度和位移
    # # datain ：加速度
    # # sprate: 采样率
    # # xx: 1
    # 为一次积分（速度）， 2为二次积分（位移）
    # # dataout：速度（xx  =  1） / 位移（xx  =  2)
    dt  =  np.true_divide(1, sprate)
    len_datain  =  np.size(datain,0)
    row_datain  =  np.size(datain,1)
    if len_datain <=  0:
        print('iomega.m:输入数组为空')
        return
    # N  =  2 ^ nextpow2(len_datain)
    N = int(math.pow(2,nextpow2(len_datain)))
    df = np.true_divide(1, N * dt)
    Nyq = np.true_divide(1, 2 * dt)

    iomega_array = 1j * 2 * math.pi * np.arange(-Nyq,Nyq-df, df)
    iomega_exp = -xx
    if N - len_datain != 0 and N-row_datain != 0:
        if len_datain > row_datain:
            datain = np.vstack([datain, np.zeros((N - len_datain, 1), dtype = float)])
        else:
            datain = np.hstack([datain, np.zeros((1, N - row_datain), dtype = float)])
    ##############################################
    # [x, y]  =  butter(4, [1, 25] / (sprate / 2)) # butterworth
    x = [0.00891445723916871,0 - 0.0356578289566749,0,0.0534867434350123,0 - 0.0356578289566749,0,0.00891445723916871]
    y = [0.00891445723916871,0 - 0.0356578289566749, 0,0.0534867434350123, 0 - 0.0356578289566749,0,0.00891445723916871]
    datain = filter.filtfilt(x, y, datain)

    #scipy.signal.butter
    A = fft(datain, N)
    ##############################################
    A = np.fft.fftshift(A)
    for j in range(N):
        if iomega_array[j] != 0:
            A[j] = A[j] * (iomega_array[j] ^ iomega_exp)
        else:
            A[j] = complex(0.0, 0.0)

    A = np.fft.ifftshift(A)
    datain = ifft(A)
    #  datain = datain'
    if len_datain > row_datain:
        dataout = np.real(datain[0:len_datain-1, row_datain-1])
    else:
        dataout = np.real(datain[len_datain-1, 0:row_datain-1])
    #  新增，输出前滤波
    # butterworth滤波0.4 - 25hz, 原来是0.1 - 25 hz
    # [x, y]  =  butter(4, [1, 25] / (sprate / 2))
    dataout = filter.filtfilt(x, y, dataout)
    dataout = np.array(dataout)
    return dataout

def nextpow2(n):
    # 求最接近数据长度的2的整数次方
    # An integer equal to 2 that is closest to the length of the data
    # Eg:
    # nextpow2(2)  =  1
    # nextpow2(2**10+1)  =  11
    # nextpow2(2**20+1)  =  21
    return np.ceil(np.log2(np.abs(n))).astype('long')

def pca(x):
    # # do PCA on image patches jiang
    # [V, E, D] = pca(X) princomp matlab
    # INPUT variables:
    # X  matrix with image patches as columns,ndarray:(3,66)
    #
    # OUTPUT variables:
    # V  whiteningm matrix
    # E  principal component transformation(orthogonal)
    # D  variances of the principal components, eigenvalues

    # Calculate the eigenvalues and eigenvectors of the new covariance matrix.
    rowx = np.size(x, 1)
    # covarianceMatrix = np.true_divide(np.dot(x,np.transpose(x)),rowx)
    covarianceMatrix = np.cov(x)
    [dd, ee] = np.linalg.eigh(covarianceMatrix)
    # try:
        #print(rowx)
        # covarianceMatrix = np.cov(x)
    # except:
    #     os.system("pause")

    # try:
    #     [dd, ee] = np.linalg.eigh(covarianceMatrix)
    # except:
    #     os.system("pause")
    # ee 第一列和第三列，与matlab结果的符号相反？？？？？？？？

    # Sort the eigenvalues and recompute matrices
    # a.sort(key=lambda x: x[1], reverse=True)
    ind0 = []
    # sorted_index = dd.argsort() #升序
    sorted_index = dd.argsort()[::-1] #将数组降序排列，但不改变数组，且返回对应的索引
    dsqrtinv = np.real(pow(dd, -0.5))
    dsqrtinv1 = np.diag(dsqrtinv)
    ee = ee[:,sorted_index]
    ss = -ee
    ee1 = np.array([ss[:,0],ee[:,1],ss[:,2]])
    ee1 = ee1.transpose()
    dd1 = dd[sorted_index]  #np.diag(dd(sorted_index))
    # vv1 = np.dot(dsqrtinv1, ee.transpose())
    return ee1, dd1


# 调用

import numpy as np


def first_half_wave_length(signal):
    rising_edge_index = None
    falling_edge_index = None
    temp_signal=np.diff(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    peak = np.where(np.diff(np.sign(temp_signal)))[0]
    temp_peak=np.abs(signal[peak])
    temp_max=np.max(temp_peak)
    temp_ind=temp_peak>0.08*temp_max #im0.1
    peak=peak[temp_ind]
    len=np.size(np.array(signal))
    peak1 =np.array(np.where(peak>zero_crossings[0]))[0]
    rising_edge_index=zero_crossings[0]
    len_zeroc=np.size(zero_crossings)
    if peak1.size==1 :
        falling_edge_index=len
    else:
        if len_zeroc>1:
            if zero_crossings[0]>peak[peak1[0]]:
                falling_edge_index = peak[peak1[0]]
            else:
                # falling_edge_index = zero_crossings[0]

                falling_edge_index=zero_crossings[zero_crossings>peak[0]][0]

        elif len_zeroc==1:
            falling_edge_index=peak[peak1[0]]




    # for i in range(len(signal) - 1):
    #     if signal[i] <= 0 and signal[i + 1] > 0:
    #         rising_edge_index = i
    #         break
    # # for i in range(rising_edge_index, len(signal) - 1):
    # for k in range(len(signal) - 1):
    #     if signal[k] >= 0 and signal[k + 1] < 0:
    #         falling_edge_index = k
    #         break
    if rising_edge_index==None:
        rising_edge_index=0
    if falling_edge_index==None:
        falling_edge_index=0
    temp=[rising_edge_index,falling_edge_index]
    if temp[1]-temp[0]<10:
        zero_crossings
    ind1 = min(temp)
    ind2 = max(temp)
    if ind2>ind1+200:
        ind2=ind1+200
    return ind1, ind2



