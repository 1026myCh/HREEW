#-*-coding:UTF-8 -*-
import pywt as wavelet
from scipy.optimize import curve_fit
from scipy.signal import butter,filtfilt
# from matplotlib import pyplot as plt

from function1 import *
def cal_distance(trace1, sprate):
    if np.shape(trace1)[0]>4000:
        result=np.shape(trace1)[0]/sprate*8
        return result
    if np.shape(trace1)[0]>1100 and np.max(trace1)>0.5 :
        result=np.shape(trace1)[0]/sprate*8
        return result

    if np.shape(trace1)[0]>600:
        trace=np.copy(trace1[:600])
    else:
        trace = np.copy(trace1)


    result = -1
    wname = 'db9'
    level = 3
    trace11 = for_one(trace)
    w=wavelet.Wavelet(wname)
    wp=wavelet.WaveletPacket(trace11,w,mode='symmetric',maxlevel=level)
    # temp =wavelet.wpdec(trace1, level, wname)#调试
    # r = temp#调试
    # freqTree = [node.path for node in temp.get_level(3, 'freq')]  # 频谱由高到低
    r=Wavelet_energy(wp)
    resultf = Fourier(trace, 200)
    resultf=np.array(resultf)
    freqlist=resultf[0,:]
    freqValue=resultf[1,:]
    maxfreq=freqlist[np.argmax(freqValue)]
    # maxfreq = resultf[1, freqmax_ind]
    dvide_ind = np.where(resultf[0, :] >= 10)[0]
    dvide_ind22 = np.where(resultf[0, :] >= 40)[0]
    dvide_ind2 = dvide_ind22[0]
    dvide_ind1 = dvide_ind[0]
    ratio = np.sum(resultf[1][0:dvide_ind1]) / np.sum(resultf[1, 0:dvide_ind2])
    ratio2 = r[2]

    tracelen=np.shape(trace1)[0]
    if (ratio < 0.44 and ratio2 > 2.1 and tracelen<1100) or (tracelen<500 and ratio<0.7 and ratio2>0.7) or (tracelen>1200 and r[1]>10 and r[1]<20):
        b = np.array([0.2483, 0.4967, 0.2483])
        a = np.array([1.0000, -0.1842, 0.1776])
        trace1 = filter_matlab( b, a,trace)
        B, A = b_delta2(trace1)
        C, dist1 = Ct(trace1)
        dist = dist1
        if C < 0.045 and B < 0.045:  #0.08
            b = np.array([0.000416599204406597, 0.00166639681762639, 0.00249959522643958, 0.00166639681762639,
                          0.000416599204406597])
            a = np.array([1.0000, -3.1806, 3.8612, -2.1122, 0.4383])
            trace2 = trace
            velocity = simpson(trace2)
            velocity = filter_matlab(b, a,velocity)
            tp_max= TaoP(velocity, 0.99, 200)
            Ap = np.max(np.abs(trace))
            # dist22 = 10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)
            dist22 = 10 ** (-0.51118 * np.log10(0.5 / tp_max) - 0.12 * np.log10(Ap) + 1.59766)
            if dist22 < 65:
                dist = dist22
            else:
                dist = -2.949e+12 * dist22 ** (-5.576) + 250.7
    elif (ratio > 0.49 and ratio2 < 2.9) or (r[0]>60 and ratio>0.42 ):
        if np.max(np.abs(trace)) > 0.8:#0.8
            C, dist1 = Ct(trace1)
            # B, A = b_delta2(trace1)
            C2, A2, dist2 = B_delta(trace,sprate)
            dist = 0.5 * dist2 + 0.5 * dist1
            dist=dist
        else:
            b = np.array([0.000416599204406597, 0.00166639681762639, 0.00249959522643958, 0.00166639681762639,
                          0.000416599204406597])
            a = np.array([1.0000, -3.1806, 3.8612, -2.1122, 0.4383])
            trace2 = trace
            velocity = simpson(trace2)
            velocity = filter_matlab(b, a,velocity)
            tp_max= TaoP(velocity, 0.99, 200)
            Ap = np.max(np.abs(trace))
            # dist22 = 10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766)
            dist22 = 10 ** (-0.51118 * np.log10(0.5 / tp_max) - 0.12 * np.log10(Ap) + 1.59766)
            if dist22 < 65:
                dist = dist22
            else:
                # dist = -2.949e+12 * dist22 ** (-5.576) + 250.7
                dist = -4.54e+12 * dist22 ** (-5.576) + 250.7
    else:
        b = np.array(
            [0.000416599204406597, 0.00166639681762639, 0.00249959522643958, 0.00166639681762639, 0.000416599204406597])
        a = np.array([1.0000, -3.1806, 3.8612, -2.1122, 0.4383])
        trace2 = trace
        velocity = simpson(trace2)
        velocity = filter_matlab(b, a,velocity)
        tp_max = TaoP(velocity, 0.99, 200)
        Ap = np.max(np.abs(trace))
        B,_=b_delta2(trace1)
        B=B*1.2
        if tp_max<2.5:
            _,dist1=Ct(trace1)
            dist22=10**(np.log10(B)-6.11)/-4.58
            dist=(dist1+dist22)/2
        else:
            _,_,dist11=B_delta(trace1,200)
            dist1=10**(np.log10(B)-6.11/-3.24)
            dist3 = (10 ** (-0.51118 * np.log10(1 / tp_max) - 0.18298 * np.log10(Ap) + 1.59766))/2 #* 1.5
            if dist11<150:
                dist=dist11
            else:
                dist = (0.4*dist1 +0.2* dist11 +0.4* dist3)

    if dist < 20:
        dist = 20

    result = dist
    return result


def Ct(data):
    # CT value
    # Used for high-frequency signal judgment
    dist = 0
    trace = data[:150]
    interval = 20
    abst = np.abs(trace)
    tempdata = np.zeros(np.shape(trace))
    for i in range(0, len(abst), interval):
        max_value = np.max(abst[i:min(i + interval - 1, len(abst))])
        tempdata[i:min(i + interval, len(abst))] = max_value

    log_amplitude = np.log(np.abs(tempdata))
    Y = log_amplitude
    L = len(Y)
    x = np.arange(1, L + 1) / 200

    def myFunction(x, a):
        return np.log(a * x)

    # Define initial parameter guess
    # initialGuess = [0.5]

    # Perform curve fitting
    coefficients, _ = curve_fit(myFunction, x, Y)

    # Constants from the MATLAB code
    a = 2.839
    b = 3.076
    # Calculate 'dist'
    dist = 10 ** (-(np.log10(coefficients[0]) - b) / a)

    return coefficients[0], dist
def b_delta2(seismic_data):
    seismic_data = seismic_data - np.mean(seismic_data)
    t = np.abs(seismic_data)
    tempdata = np.zeros(np.shape(seismic_data))
    for i in range(0, len(t), 10):
        max_value = np.max(t[i:min(i + 10, len(t))])
        tempdata[i:min(i + 10, len(t))] = max_value
    tempdata1 = np.copy(np.abs(tempdata))
    log_amplitude = np.log(tempdata1)
    Y = log_amplitude
    L = len(log_amplitude)
    X = np.arange(1, L + 1) / 200
    # scipy.optimize.curve_fit
    # myFunction = lambda coefficients, x: coefficients[0] + np.log(x) + coefficients[1] * x
    # initialGuess = [0.5, 0.5]
    def func1(x, a, b):
        return a + np.log(x) + b * x
    coefficients,_ = curve_fit(func1,X,Y)
    A=-coefficients[1]
    B=np.exp(coefficients[0])
    return B,A

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

    dt = 1 / 200  #采样频率为200 Hz，即每隔0.05秒进行一次采样
    # 初始化速度向量
    velocity = np.zeros_like(acceleration)

    # 使用辛普森法求解速度
    for i in range(1, len(acceleration) - 1):
        velocity[i] = velocity[i - 1] + (acceleration[i - 1] + 4 * acceleration[i] + acceleration[i + 1]) * (dt / 6)

    return velocity

def Wavelet_energy(wp):
    maxlevel=wp.maxlevel
    freqTree=[node.path for node in wp.get_level(maxlevel,'freq')]#频谱由高到低
    totalEn=0
    Enlist=[]
    for i in range(len(freqTree)):
        # tempEn=np.linalg.norm((wp[freqTree[i]].data))
        tempEn = np.linalg.norm( (wp[freqTree[i]].data))
        tempEn=np.square(tempEn)
        totalEn=totalEn+tempEn
        Enlist.append(tempEn)
    Enlist_ndarry=np.array(Enlist)
    percentEn=Enlist_ndarry/totalEn*100
    return percentEn


