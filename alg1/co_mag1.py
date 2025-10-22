#-*-coding:UTF-8 -*-
import numpy as np
import math
import scipy
from function1 import Rdelta,iomega,simpson,filter_matlab # 73*2,ndarray
import scipy.signal as signal

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
                B, A = scipy.signal.butter(2, 0.075 * 2 / 200, 'high')
                B1, A1 = scipy.signal.butter(2, 15 * 2 / 200, 'low')
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
        a = signal.filtfilt(b1, a1, a)
        a = signal.filtfilt(b, a, a)
        v = simpson(a)
        v = signal.filtfilt(b1, a1, v)
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