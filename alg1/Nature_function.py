#-*-coding:UTF-8 -*-
import os
import numpy as np
from scipy.signal import butter, filtfilt, sosfreqz,  sosfilt, sosfilt_zi, sosfiltfilt,lfilter
from scipy.fftpack import fft
from scipy.integrate import simps
from function1 import B_delta
import pywt as wavelet
from scipy.optimize import curve_fit, least_squares
from function1 import *

# def cal_distance(trace, sprate):
#     result = -1
#     wname = 'db9'
#     level = 3
#     trace1 = for_one(trace)
#     w=wavelet.Wavelet(wname)
#     temp=wavelet.WaveletPacket(trace1,w,mode='symmetric',maxlevel=level)
#     # temp =wavelet.wpdec(trace1, level, wname)#调试
#     r = temp#调试
#
#
#     resultf = Fourier(trace, 200)
#
#     freqmax_ind,_ = max(enumerate(resultf[1]), key=lambda x: x[1])
#     maxfreq = resultf[0][freqmax_ind]
#     dvide_ind = np.where(resultf[0] >= 10)[0]
#     dvide_ind22 = np.where(resultf[0] >= 40)[0]
#     dvide_ind2 = dvide_ind22[0]
#     dvide_ind1 = dvide_ind[0]
#     ratio = np.sum(resultf[1][0:dvide_ind1]) / np.sum(resultf[1][0:dvide_ind2])
#     ratio2 = r[3]
#     trace1 = trace
#
#     if ratio < 0.39 and ratio2 > 2.1:
#         b = np.array([0.2483, 0.4967, 0.2483])
#         a = np.array([1.0000, -0.1842, 0.1776])
#         trace1 = lfilter(trace, b, a)
#         B, A = b_delta2(trace1)
#         C, dist1 = Ct(trace1)
#         dist = dist1
#
#         if C < 0.08 and B < 0.08:
#             b = np.array([0.000416599204406597, 0.00166639681762639, 0.00249959522643958, 0.00166639681762639,
#                           0.000416599204406597])
#             a = np.array([1.0000, -3.1806, 3.8612, -2.1122, 0.4383])
#             trace2 = trace
#             velocity = simpson(trace2)
#             velocity = lfilter(velocity, b, a)
#             tp_max, _ = TaoP(velocity, 0.99, 200)
#             Ap = np.max(np.abs(trace))
#             dist22 = 10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)
#
#             if dist22 < 65:
#                 dist = dist22
#             else:
#                 dist = -2.949e+12 * dist22 ** (-5.576) + 250.7
#     elif ratio > 0.45 and ratio2 < 2.5:
#         if np.max(np.abs(trace)) > 0.8:
#             C, dist1 = Ct(trace1)
#             C2, A2, dist2, mag = B_delta(trace, 1, sprate)
#             dist = 0.3 * dist2 + 0.1 * dist1
#         else:
#             b = np.array([0.000416599204406597, 0.00166639681762639, 0.00249959522643958, 0.00166639681762639,
#                           0.000416599204406597])
#             a = np.array([1.0000, -3.1806, 3.8612, -2.1122, 0.4383])
#             trace2 = trace
#             velocity = simpson(trace2)
#             velocity = lfilter(velocity, b, a)
#             tp_max, _ = TaoP(velocity, 0.99, 200)
#             Ap = np.max(np.abs(trace))
#             dist22 = 10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)
#
#             if dist22 < 65:
#                 dist = dist22
#             else:
#                 dist = -2.949e+12 * dist22 ** (-5.576) + 250.7
#     else:
#         b = np.array(
#             [0.000416599204406597, 0.00166639681762639, 0.00249959522643958, 0.00166639681762639, 0.000416599204406597])
#         a = np.array([1.0000, -3.1806, 3.8612, -2.1122, 0.4383])
#         trace2 = trace
#         velocity = simpson(trace2)
#         velocity = lfilter(velocity, b, a)
#         tp_max, _ = TaoP(velocity, 0.99, 200)
#         Ap = np.max(np.abs(trace))
#         B,_=b_delta2(trace1)
#         if tp_max<2.5:
#             _,dist1=Ct(trace1)
#             dist22=10**(np.log10(B)-6.11)/-4.58
#             dist=(dist1+dist22)/2
#         else:
#             _,_,dist11,_=B_delta(trace1,200)
#             dist1=10**(np.log10(B)-6.11/-3.24)
#             dist3 = (10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)) * 1.5
#             dist = (dist1 + dist11 + dist3) / 3
#
#     if dist < 20:
#         dist = 20
#
#     result = dist
#     return result


def Ct(data):
    # CT value
    # Used for high-frequency signal judgment
    dist = 0
    trace = data[:150]

    interval = 20
    abst = np.abs(trace)
    tempdata = []

    for i in range(1, len(abst), interval):
        max_value = np.max(abst[i:min(i + interval - 1, len(abst))])
        tempdata[i:min(i + interval - 1, len(abst))] = max_value

    log_amplitude = np.log(np.abs(tempdata))

    Y = log_amplitude
    L = len(Y)
    x = np.arange(1, L + 1) / 200

    def myFunction(x, a):
        return np.log(a * x)

    # Define initial parameter guess
    initialGuess = [0.5]

    # Perform curve fitting
    coefficients, _ = curve_fit(myFunction, x, Y, p0=initialGuess)

    # Constants from the MATLAB code
    a = 2.839
    b = 3.076

    # Calculate 'dist'
    dist = 10 ** (-(np.log10(coefficients[0]) - b) / a)

    return coefficients[0], dist


def b_delta2(seismic_data):
    seismic_data = seismic_data - np.mean(seismic_data)
    t = np.abs(seismic_data)
    tempValue = []
    tempdata = []

    for i in range(0, len(t), 10):
        max_value = np.max(t[i:min(i + 10, len(t))])
        tempdata[i:min(i + 10, len(t))] = max_value

    tempdata1 = np.abs(tempdata)
    ind1 = np.where(tempdata1 > 0.05)[0][0]

    log_amplitude = np.log(tempdata1[ind1:])
    Y = log_amplitude
    L = len(log_amplitude)
    x = np.arange(1, L + 1) / 200

    myFunction = lambda coefficients, x: coefficients[0] + np.log(x) + coefficients[1] * x
    initialGuess = [0.5, 0.5]
    coefficients = least_squares(myFunction, initialGuess, args=(x, Y)).x


def for_one(Data):
    t = Data
    m = len(t)
    temp1 = Data
    temp = (temp1 - min(temp1)) / (max(temp1) - min(temp1))
    temp = temp - temp.mean()
    r = temp
    return r

def simpson(data):
    # 计算采样时间间隔
    acceleration = data

    dt = 1 / 200  # 假设采样频率为10 Hz，即每隔0.1秒进行一次采样

    # 初始化速度向量
    velocity = np.zeros_like(acceleration)

    # 使用辛普森法求解速度
    for i in range(1, len(acceleration) - 1):
        velocity[i] = velocity[i - 1] + (acceleration[i - 1] + 4 * acceleration[i] + acceleration[i + 1]) * (dt / 6)

    return velocity

def co_magnitude1(Traces_evt, Distance, beM, Sprate, Debug, Stime, flagarea, flag):
    result = -1  # Mag震级初始值
    if beM == -1:
        beM = 0
    if flag is not None:
        if flag == 1:
            return result
    if flagarea is None:
        flagarea = 3
    if Stime <= 0 and Distance > 25:
        if Distance <= 0:
            return result
        ans1 = np.diff(Traces_evt[:, 2])
        ans2 = np.diff(Traces_evt[:, 5])
        ind1 = np.argmax(ans1)
        ind2 = np.argmax(ans2)
        Sprate = Sprate[0]
        num = Traces_evt.shape[1]
        lent = len(Traces_evt[:, 2])
        B, A = signal.butter(2, 0.075 * 2 / 200, 'high')
        B1, A1 = signal.butter(2, 15 * 2 / 200, 'low')
        if lent >= 500:
            ud1 = filter_matlab(B, A, Traces_evt[0:2.5 * Sprate, 2])
            ud1 = filter_matlab(B1, A1, ud1)
            v1 = simpson1(ud1)
            disp1 = simpson1(v1)
            d1 = np.max(np.abs(disp1))
            Magnitude11 = (np.log10(d1 * 10000) + 1.2 * np.log10(Distance) + 5.0 * 10 ** -4 * Distance - 5.0 * 10 ** -3 * 10 + 0.46) / 0.72
            mag_tc1, tc = TaoC(Traces_evt[0:2.5 * Sprate, 2], Sprate[0], 'sc', 2, 0)
            temp = [Magnitude11]
            Magnitude1 = np.min(temp)
            tempM = [mag_tc1]
            minM = np.min(tempM)
            if num > 3:
                ud2 = filter_matlab(B, A, Traces_evt[0:2.5 * Sprate, 5])
                v2 = simpson1(ud2)
                disp2 = simpson1(v2)
                d2 = np.max(np.abs(disp2))
                Magnitude22 = (np.log10(d2 * 10000) + 1.2 * np.log10(Distance) + 5.0 * 10 ** -4 * Distance - 5.0 * 10 ** -3 * 10 + 0.46) / 0.72
                mag_tc2, tc = TaoC(Traces_evt[0:2.5 * Sprate, 5], Sprate[0], 'sc', 2, 0)
                temp = [Magnitude11, Magnitude22]
                Magnitude1 = np.min(temp)
                tempM = [mag_tc1, mag_tc2]
                minM = np.min(tempM)
            Magnitude2 = minM
        else:
            ud1 = filter_matlab(B, A, Traces_evt[:, 2])
            ud1 = filter_matlab(B1, A1, ud1)
            v1 = simpson1(ud1)
            disp1 = simpson1(v1)
            d1 = np.max(np.abs(disp1))
            Magnitude11 = (np.log10(d1 * 10000) + 1.2 * np.log10(Distance) + 5.0 * 10 ** -4 * Distance - 5.0 * 10 ** -3 * 10 + 0.46) / 0.78
            mag_tc1, tc = TaoC(Traces_evt[:, 2], Sprate[0], 'sc', 2, 0)
            temp = [Magnitude11]
            Magnitude1 = np.min(temp)
            tempM = [mag_tc1]
            minM = np.min(tempM)
            Magnitude2 = minM
            if num > 3:
                ud2 = filter_matlab(B, A, Traces_evt[:, 5])
                ud2 = filter_matlab(B1, A1, ud2)
                v2 = simpson1(ud2)
                disp2 = simpson1(v2)
                d2 = np.max(np.abs(disp2))
                Magnitude22 = (np.log10(d2 * 10000) + 1.2 * np.log10(Distance) + 5.0 * 10 ** -4 * Distance - 5.0 * 10 ** -3 * 10 + 0.46) / 0.78
                mag_tc2, tc = TaoC(Traces_evt[:, 5], Sprate[0], 'sc', 2, 0)
                temp = [Magnitude11, Magnitude22]
                Magnitude1 = np.min(temp)
                tempM = [mag_tc1, mag_tc2]
                minM = np.min(tempM)
        deltM = np.abs(Magnitude1 - Magnitude2)
        MagnitudeFinal1 = 0.5 * (Magnitude1 + Magnitude2)
        result = np.round(MagnitudeFinal1, -1)
        if result <= 3.5:
            mag_tc1, tc = TaoC(Traces_evt[:, 2], Sprate[0], 'sc', 0, 0)
            mag_tc2, tc = TaoC(Traces_evt[:, 5], Sprate[0], 'sc', 0, 0)
            result = np.mean([mag_tc1, mag_tc2])
        elif result > 5.5:
            if len(Traces_evt[:, 5]) <= 600:
                mag_tc2, tc = TaoC(Traces_evt[:, 5], Sprate[0], 'sc', 1, 0)
                mag_tc1, tc = TaoC(Traces_evt[:, 2], Sprate[0], 'sc', 1, 0)
                B, A = signal.butter(2, 0.075 * 2 / 200, 'high')
                B1, A1 = signal.butter(2, 15 * 2 / 200, 'low')
                ud1 = filter_matlab(B, A, Traces_evt[:, 2])
                ud1 = filter_matlab(B1, A1, ud1)
                ud2 = filter_matlab(B, A, Traces_evt[:, 5])
                ud2 = filter_matlab(B1, A1, ud2)
                v1 = simpson1(ud1)
                d1 = np.max(np.abs(simpson1(v1)))

def simpson1(data):
    # 输入加速度信号和时间步长
    acceleration = data  # 替换为您的加速度数据
    dt = 1/200  # 采样时间间隔为1/200秒 (200Hz采样率)

    # 初始化速度数组
    velocity = np.zeros_like(acceleration)

    # 使用辛普森法进行积分计算速度
    for i in range(1, len(acceleration)-2, 2):
        velocity[i+1] = velocity[i-1] + (acceleration[i] + 4*acceleration[i+1] + acceleration[i+2]) * dt / 3

    # 处理奇数索引
    for i in range(2, len(acceleration)-1, 2):
        velocity[i+1] = velocity[i-1] + (acceleration[i-1] + 4*acceleration[i] + acceleration[i+1]) * dt / 3

    return velocity


def TaoC(a, sprate, area, Debug, flag=None):
    tc = 0

    if flag == 1 if flag is not None else False:
        return None, tc

    if Debug == 1:  # 对应大震
        b, a = signal.butter(2, 20 * 2 / 200, 'low')
        b1, a1 = signal.butter(2, 0.5 * 2 / 200, 'low')
        a = signal.lfilter(b1, a1, a)
        a = signal.lfilter(b, a, a)
        v = simpson(a)
        v = signal.lfilter(b1, a1, v)
        d = simpson(v)
        r = np.sum(v ** 2) / np.sum(d ** 2)
        r1 = np.sqrt(r)
        if not np.isreal(r1):
            r1 = 1
        tc = 2 * np.pi / r1
        M = 2.94 * np.log10(tc) + 5.26 - 0.62
    elif Debug == 2:  # 对应小震
        v = iomega(a, sprate, 1, Debug)
        d = iomega(a, sprate, 2, Debug)
        r = np.sum(v ** 2) / np.sum(d ** 2)
        r1 = np.sqrt(r)
        if not np.isreal(r1):
            r1 = 1
        tc = 2 * np.pi / r1
        M = 2.94 * np.log10(tc) + 5.26 + 0.3
    elif Debug == 0:
        v = iomega(a, sprate, 1, Debug)
        d = iomega(a, sprate, 2, Debug)
        r = np.sum(v ** 2) / np.sum(d ** 2)
        r1 = np.sqrt(r)
        if not np.isreal(r1):
            r1 = 1
        tc = 2 * np.pi / r1
        period = tc
        M = 0.4046 * period ** 2 + 1.2767 * period + 2.7713

    return M, tc

def Wavelet_energy(data,fs,wavelet,maxlevel):
    wp = wavelet.WaveletPacket(data, wavelet, mode='symmetric', maxlevel=maxlevel)
    freq=[node.path for node in wp.get_level(maxlevel,'freq')]
    


