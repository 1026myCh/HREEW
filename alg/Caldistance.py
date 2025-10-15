import numpy as np
from scipy.optimize import curve_fit
import pywt

def cal_distance(trace, sprate, flag=None):
    result = -1
    if flag == 1:
        return result

    wname = 'db9'  # 小波基函数名称
    level = 3  # 小波分解级数
    trace1 = forone(trace)
    temp = wpdec(trace1, level, wname)
    r = wenergy(temp)
    resultf = fourier(trace, 200)
    freqmax_ind = np.argmax(resultf[1])
    maxfreq = resultf[1, freqmax_ind]
    dvide_ind = np.where(resultf[0] >= 10)[0][0]  # 频率分界值 10Hz
    dvide_ind2 = np.where(resultf[0] >= 40)[0][0]
    ratio = np.sum(resultf[1, :dvide_ind]) / np.sum(resultf[1, :dvide_ind2])
    ratio2 = r[3]
    trace1 = trace

    if ratio < 0.39 and ratio2 > 2.1:  # 初判为近震
        b = np.array([0.2483, 0.4967, 0.2483])
        a = np.array([1.0, -0.1842, 0.1776])
        trace1 = filtfilt(b, a, trace)
        B, A = b_delta2(trace1)
        C, dist1 = ct(trace1)
        dist = dist1
        if C < 0.08 and B < 0.08:  # 误判决策，按远震计算
            b = np.array([0.000416599204406597, 0.00166639681762639, 0.00249959522643958,
                          0.00166639681762639, 0.000416599204406597])
            a = np.array([1.0, -3.1806, 3.8612, -2.1122, 0.4383])
            trace2 = trace.copy()
            velocity = simpson(trace2)
            velocity = filtfilt(b, a, velocity)
            tp_max, _ = tao_p(velocity, 0.99, 200)
            Ap = np.max(np.abs(trace))
            dist22 = 10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)
            if dist22 < 65:
                dist = dist22
            else:
                dist = -2.949e+12 * dist22 ** (-5.576) + 250.7

    elif ratio > 0.45 and ratio2 < 2.5:  # 初判为远震
        if np.max(np.abs(trace)) > 0.8:
            C, dist1 = ct(trace1)
            B, A = b_delta2(trace1)
            C2, A2, dist2, mag = b_delta(trace, 1, sprate)
            dist = 0.3 * dist2 + 0.1 * dist1
        else:
            b = np.array([0.000416599204406597, 0.00166639681762639, 0.00249959522643958,
                          0.00166639681762639, 0.000416599204406597])
            a = np.array([1.0, -3.1806, 3.8612, -2.1122, 0.4383])
            trace2 = trace.copy()
            velocity = simpson(trace2)
            velocity = filtfilt(b, a, velocity)
            tp_max, _ = tao_p(velocity, 0.99, 200)
            Ap = np.max(np.abs(trace))
            dist22 = 10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)
            if dist22 < 65:
                dist = dist22
            else:
                dist = -2.949e+12 * dist22 ** (-5.576) + 250.7
    else:  # 模糊判断
        C, dist1 = ct(trace1)
        B, A = b_delta2(trace1)
        b = np.array([0.000416599204406597, 0.00166639681762639, 0.00249959522643958,
                      0.00166639681762639, 0.000416599204406597])
        a = np.array([1.0, -3.1806, 3.8612, -2.1122, 0.4383])
        trace2 = trace.copy()
        velocity = simpson(trace2)
        velocity = filtfilt(b, a, velocity)
        tp_max, _ = tao_p(velocity, 0.99, 200)
        Ap = np.max(np.abs(trace))
        B, A = b_delta2(trace1)
        dist = 0
        if tp_max < 2.5:
            C, dist1 = ct(trace1)
            dist22 = 10 ** ((np.log10(B) - 6.11) / -4.58)
            dist = (dist1 + dist22) / 2
        else:
            C2, A2, dist11, mag = b_delta(trace, 1, sprate)
            dist1 = 10 ** ((np.log10(B) - 6.11) / -3.24)
            dist3 = (10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)) * 1.5
            dist = (dist1 + dist11 + dist3) / 3

    if dist < 20:
        dist = 20

    result = dist
    return result

def Ct(data):
    # CT值
    # 用于高频信号判断
    dist = 0
    trace = data[:150]

    interval = 20
    abst = np.abs(trace)
    tempdata = np.zeros_like(abst)
    for i in range(0, len(abst), interval):
        max_value = np.max(abst[i:min(i + interval, len(abst))])  # 找到每个间隔内的最大值
        tempdata[i:min(i + interval, len(abst))] = max_value  # 将最大值赋给相应的索引位置

    log_amplitude = np.log(np.abs(tempdata))

    x = np.arange(1, len(log_amplitude) + 1) / 200

    # 定义拟合函数
    def my_function(x, coefficients):
        return np.log(coefficients * x)

    # 定义初始参数猜测值
    initial_guess = [0.5]  # 假设初始参数猜测值为 [a, b]

    # 使用曲线拟合求解系数
    coefficients, _ = curve_fit(my_function, x, log_amplitude, p0=initial_guess)
    C = coefficients[0]

    a = 2.839
    b = 3.076
    # a = 0.7232
    # b = 0.1738
    dist = 10**(-(np.log10(C) - b) / a)

    return C, dist

def forone(Data):
    t = Data
    m = len(t)
    temp1 = Data[:]
    temp = (temp1 - np.min(temp1)) / (np.max(temp1) - np.min(temp1))
    temp = temp - np.mean(temp)
    r = temp
    return r




def b_delta2(seismic_data):
    # 1. 去除零位移（直流）分量
    seismic_data = seismic_data - np.mean(seismic_data)
    t = np.abs(seismic_data)

    # 2. 计算参数A和B
    interval = 10
    abst = np.abs(seismic_data)
    tempdata = np.zeros_like(abst)
    for i in range(0, len(abst), interval):
        max_value = np.max(abst[i:i + interval])
        idx = np.argmax(abst[i:i + interval])
        tempdata[i:i + interval] = max_value

    tempdata1 = np.abs(tempdata)

    ind1 = 0  # Initial index
    log_amplitude = np.log(tempdata1[ind1:])

    Y = log_amplitude
    L = len(log_amplitude)
    x = np.arange(1, L + 1) / 200

    # 定义拟合函数
    def myFunction(x, a, b):
        return a + np.log(x) + b * x

    # 初始参数猜测值
    initialGuess = [0.5, 0.5]

    # 使用 curve_fit 进行拟合
    coefficients, _ = curve_fit(myFunction, x, Y, p0=initialGuess)

    A = -coefficients[1]
    B = np.exp(coefficients[0])

    return B, A
