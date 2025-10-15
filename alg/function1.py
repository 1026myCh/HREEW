# 其他各类函数 画图
#-*-coding:UTF-8 -*-
import math
import signal
from datetime import *
from scipy import signal
from scipy.signal import lfiltic,lfilter_zi
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import sklearn
# from matplotlib import pyplot as plt
import pywt as wavelet

def Ct1(data):
    # CT value
    # Used for high-frequency signal judgment
    dist = 0
    trace = data[:240]
    interval = 5 #8 good reuslt
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
    # a = 2.839
    # b = 3.076
    a=-0.493
    b=1.826#1.826
    # Calculate 'dist'
    # dist = 10 ** (-(np.log10(coefficients[0]) - b) / a)
    dist = 10 ** (a*np.log10(coefficients[0]) + b )
    return coefficients[0], dist

def LTW_reset(Buffer, Sprate):
    # 自相关最低的时候认为噪声，确定信号的自相关较高
    Len = len(Buffer)
    if Len == 1400:
        return Buffer
    LTW = Buffer[0:1400]
    return LTW


def MyFilter(fs, data, L_fs, H_fs):
    filtered_data = np.copy(data)
    if L_fs <= 0 and H_fs <= 0:
        if data is None:
            return filtered_data
    # fb,fa=signal.butter(4,[2*L_fs/fs,2*H_fs/fs],'bandpass',output='sos')
    fb,fa=signal.butter(4,[2*L_fs/fs,2*H_fs/fs],'bandpass')
    # fb = [0.000401587597758709, 0,
    #       -0.00160635039103484, 0,
    #       0.00240952558655226, 0,
    #       -0.00160635039103484, 0,
    #       0.000401587597758709]
    # fa = [1, -7.18522612270076, 22.6153766287987,
    #       -40.7334658923449, 45.9266056466201,
    #       -33.1963263771614, 15.0231035453242,
    #       -3.89199799726802, 0.441930568732715]
    # from matplotlib import pyplot as plt

    # y_input=data[:100]
    zi = lfilter_zi(fb, fa)
    filtered_data = signal.lfilter(fb, fa, data,zi=zi*data[0])
    # filtered_data = signal.filtfilt(fb, fa, data)
    # filtered_data=signal.sosfilt(sos,filtered_data)
    filtered_data = np.array(filtered_data[0])
    return filtered_data

def MyFilter1(fs, data, L_fs, H_fs):
    filtered_data = np.copy(data)
    if L_fs <= 0 and H_fs <= 0:
        if data is None:
            return filtered_data
    # fb,fa=signal.butter(4,[2*L_fs/fs,2*H_fs/fs],'bandpass',output='sos')
    fb,fa=signal.butter(4,[2*L_fs/fs,2*H_fs/fs],'bandpass')
    # fb = [0.000401587597758709, 0,
    #       -0.00160635039103484, 0,
    #       0.00240952558655226, 0,
    #       -0.00160635039103484, 0,
    #       0.000401587597758709]
    # fa = [1, -7.18522612270076, 22.6153766287987,
    #       -40.7334658923449, 45.9266056466201,
    #       -33.1963263771614, 15.0231035453242,
    #       -3.89199799726802, 0.441930568732715]
    # filtered_data = signal.lfilter(fb, fa, data)
    filtered_data = signal.filtfilt(fb, fa, data)
    # filtered_data=signal.sosfilt(sos,filtered_data)
    filtered_data = np.array(filtered_data)
    return filtered_data

def Fourier(Trace_evt, Sprate):
    Fs = Sprate
    DataLen = np.size(Trace_evt)
    NFFT = 2 ** nextpow2(DataLen)
    Yud = np.fft.fft(Trace_evt, NFFT) / DataLen
    f = Fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # fig=plt.flag('fig1')
    # plt.plot(Yud)
    x_axis = f
    y_axis = 2 * np.abs(Yud[0:int(NFFT / 2)])
    return x_axis, y_axis

    # data_len = len(traces_evt)
    # nfft = 2 ** np.ceil(np.log2(data_len)).astype(int)
    # yud = np.fft.fft(traces_evt, nfft) / data_len
    # f = sprate / 2 * np.linspace(0, 1, nfft // 2 + 1)
    # amplitude_spectrum = 2 * np.abs(yud[:nfft // 2 + 1])
    # result_f = np.vstack((f, amplitude_spectrum))
    # return result_f


def nextpow2(n):
    # 求最接近数据长度的2的整数次方
    # An integer equal to 2 that is closest to the length of the data
    # Eg:
    # nextpow2(2) = 1
    # nextpow2(2**10+1) = 11
    # nextpow2(2**20+1) = 21
    return np.ceil(np.log2(np.abs(n))).astype(
        'long')


def filter_matlab(b, a, data_in):
    x=np.copy(data_in)
    zi = lfilter_zi(b, a)

    y = signal.lfilter(b, a, x,zi=zi*data_in[0])
    y=np.array(y[0])
    # y=signal.filtfilt(b,a,x)
    return y

def cmp_vector(Traces_evt_input):
    PGA1 = 0.0
    PGA2 = 0.0
    PGApos1 = 0.0
    PGApos2 = 0.0
    Traces_evt=np.copy(Traces_evt_input)
    len = np.shape(Traces_evt)[0]
    lendata = np.shape(Traces_evt)[1]
    if lendata < 1:
        return PGA1, PGA2,PGApos1,PGApos2
    else:
        tempdata1 = np.sqrt(
            Traces_evt[0] ** 2 + Traces_evt[1] ** 2 +
            Traces_evt[2] ** 2)
        tempdata1 = [np.round(x, 3) for x in tempdata1]
        PGA1 = np.max(tempdata1)
        PGApos1 = np.where(tempdata1==PGA1)[0][0]
        if len > 3:
            tempdata2 = np.sqrt(
                Traces_evt[3] ** 2 + Traces_evt[
                    4] ** 2 + Traces_evt[5] ** 2)
            tempdata2 = [np.round(x, 3) for x in tempdata2]
            PGA2 = np.max(tempdata2)
            PGApos2 = np.where(tempdata2==PGA2)[0][0]
            return PGA1, PGA2,PGApos1,PGApos2
        else:
            PGA2 = 0
            return PGA1, PGA2,PGApos1,PGApos2


def EEW_Alarm(PGA1, PGA2, PGAt,ThreshGal,Alarm):
    Alarm=Alarm
    ThreshGal3 = float(ThreshGal[2])
    ThreshGal2 = float(ThreshGal[1])
    ThreshGal1 = float(ThreshGal[0])
    AlarmLevel = 0
    if PGA1<PGA2:
        Alarm.PGA = PGA1
    else:
        Alarm.PGA = PGA2
    if Alarm.PGA > ThreshGal3:
        AlarmLevel = 3
    elif Alarm.PGA > ThreshGal2:
        AlarmLevel = 2
    elif Alarm.PGA > ThreshGal1:
        AlarmLevel = 1
    Alarm.AlarmLevel = AlarmLevel
    Alarm.recordtime = np.round(PGAt,3)  # 根据收到的StartT推算的
    currtime = datetime.now()
    # ans_time = np.round(currtime.timestamp(), 3)
    Alarm.Alarmtime = np.round(currtime.timestamp(), 3)  # 输出本消息的时刻
    Alarm.delT = Alarm.Alarmtime-Alarm.recordtime # Ararmtime-recordtime
    return Alarm



def autocorrelation(x, lags):  # 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:] - x[i:].mean(), x[:n - i] - x[
                                                             :n - i].mean())[
                  0] \
              / (x[i:].std() * x[:n - i].std() * (
            n - i)) \
              for i in range(1, lags + 1)]
    return result


def iomega(datain_input, sprate, xx):
    # datain:数据 1*n
    # sprate：采样率
    # xx:积分次数
    # 1-25hz 4阶
    datain = np.copy(datain_input)
    dt = 1 / sprate
    tempnum = np.shape(datain)[0]
    N = 2 ** nextpow2(tempnum)
    df = 1 / (N * dt)
    Nyq = 1 / (2 * dt)  # 奈奎斯特采样率
    temp = np.linspace(-Nyq, Nyq - df, N)
    iomega_arry = 1j * 2 * math.pi * temp
    iomega_exp = -xx
    [size1] = np.shape(datain)
    x = [0.00891445723916871, 0,
         -0.0356578289566749, 0,
         0.0534867434350123, 0,
         -0.0356578289566749, 0,
         0.00891445723916871]
    y = [1, -5.97019036284279, 15.6760061739522,
         -23.7534053193241, 22.8038401706724,
         -14.2293982265302, 5.63370270828677,
         -1.29223569702625, 0.131680715038691]
    if (N - size1) != 0:
        datain = np.hstack([datain,
                            np.zeros((N - size1),
                                     dtype=float)])
    tempdatain = filter_matlab(x, y, datain)
    A = np.fft.fft(tempdatain, N)
    A = np.fft.fftshift(A)
    for j in range(N):
        if iomega_arry[j] != 0:
            A[j] = A[j] * (iomega_arry[
                               j] ** iomega_exp)
        else:
            A[j] = np.array([0 + 0 * j],
                            dtype=complex)

    A = np.fft.ifftshift(A)
    datain = np.fft.ifft(A)


    dataout = np.real(datain[0:size1])
    dataout = filter_matlab(x, y, dataout)
    return dataout


def iomega_azi(datain_in, sprate, xx):
    datain = np.copy(datain_in)
    dt = 1 / sprate
    tempnum = np.shape(datain)[0]
    N = 2 ** nextpow2(tempnum)
    df = 1 / (N * dt)
    Nyq = 1 / (2 * dt)  # 奈奎斯特采样率
    temp = np.linspace(-Nyq, Nyq - df, N)
    iomega_arry = 1j * 2 * math.pi * temp
    iomega_exp = -xx
    [size1] = np.shape(datain)
    # b = [0.028,0.053,0.071,0.053,0.028]#HD
    # a = [1.000,-2.026,2.148,-1.159,0.279]# 没有反！！！HD
    # b,a=signal.butter(4,[75,100],fs=200,btype='band')
    sos = np.array([[1, 0, -1, 1, -1.97502583381370, 0.978717118573789],
                    [1, 0, -1, 1, -1.99091380463892, 0.991498547783860],
                    [1, 0, -1, 1, -1.97096688012118, 0.971910194847182],
                    [1, 0, -1, 1, -1.95463362814574, 0.956874871079503]])
    if (N - size1) != 0:
        datain = np.hstack([datain, np.zeros((N - size1), dtype=float)])
    tempdatain = signal.sosfilt(sos, datain)
    A = np.fft.fft(tempdatain, N)
    A = np.fft.fftshift(A)
    for j in range(N):
        if iomega_arry[j] != 0:
            A[j] = A[j] * (iomega_arry[
                               j] ** iomega_exp)
        else:
            A[j] = np.array([0 + 0 * j],
                            dtype=complex)

    A = np.fft.ifftshift(A)
    datain = np.fft.ifft(A)

    dataout = np.real(datain[0:size1])
    dataout = signal.sosfilt(sos, dataout)
    return dataout


def FilterV(Data, Fs):
    data_in = np.copy(Data[2])
    sos = np.array([1, -1, 0, 1, -0.984491960028794, 0])
    UD_high_pass_value_1 = signal.sosfilt(sos, data_in)
    UD_high_pass_value_1 = np.array(UD_high_pass_value_1)
    bb = [0.000194303927727353,
          -0.000124310499575406,
          -0.00161719199400788,
          -0.00428857840136351,
          -0.00621293793681084, \
          -0.00405810468220371,
          0.00394827019242057, 0.0143567095268667,
          0.0181408384666999, 0.00643377065912978,
          -0.0199852393508868, \
          -0.0450050327165390,
          -0.0426227899981891,
          0.00669433824641129, 0.0988030174770648,
          0.201508141442756, 0.269270407060813, \
          0.269270407060813, 0.201508141442756,
          0.0988030174770648, 0.00669433824641129,
          -0.0426227899981891,
          -0.0450050327165390, \
          -0.0199852393508868,
          0.00643377065912978, 0.0181408384666999,
          0.0143567095268667, 0.00394827019242057, \
          -0.00405810468220371,
          -0.00621293793681084,
          -0.00428857840136351,
          -0.00161719199400788,
          -0.000124310499575406,
          0.000194303927727353]
    L = np.size(bb)
    fir_tap = int(np.floor(L / 2))
    full_data = np.zeros(fir_tap)
    UD_high_pass_value_1 = np.hstack([UD_high_pass_value_1, full_data])

    UD_low_pass_value_1 = signal.lfilter(bb, [1], UD_high_pass_value_1)  # 需注意
    Filterdata1 = UD_low_pass_value_1[fir_tap + 1:]
    return Filterdata1


def Dis_compt_0726(data, fs):
    flag_dis = 0
    A2 = -1
    if np.shape(data)[0] < 6:
        data_ud = FilterV(data, fs)
        [B1, A1, Distance] = B_delta(data_ud, fs)
        len = np.shape(data)[1]
        if len == 2 * fs:
            Distance = (69.8231 * math.log(10, B1) * math.log(10, B1) - 21.2595 * math.log(19, B1) + 22.8587)
        elif len == 2.56 * fs:
            Distance = (70.9925 * math.log(10, B1) * math.log(10, B1) - 18.6683 * math.log(19, B1) + 22.6376)
        dis_2s = Distance
    else:
        data_ud1 = FilterV(data[0:3], fs)
        data_ud2 = FilterV(data[3:6], fs)
        [B1, A1, Distance] = B_delta(data_ud1, fs)
        [B2, A2, Distance] = B_delta(data_ud2, fs)

        if np.shape(data)[1] == 2 * fs:
            Distance1 = (69.8231 * math.log10(B1) * math.log10(B1) - 21.2595 * math.log10(B1) + 22.8587)  # 2s 数据长度
            Distance2 = (69.8231 * math.log10(B2) * math.log10(B2) - 21.2595 * math.log10(B2) + 22.8587)
            Distance = (Distance1 + Distance2) / 2
        elif np.shape(data)[1] == 2.56 * fs:
            Distance1 = (70.9925 * math.log10(B1) * math.log10(B1) - 18.6683 * math.log10(B1) + 22.6376)  # 2.56s 数据长度
            Distance2 = (70.9925 * math.log10(B2) * math.log10(B2) - 18.6683 * math.log10(B2) + 22.6376)
            Distance = (Distance1 + Distance2) / 2

        dis_2s = Distance
    return dis_2s, A1, A2


def cmp_t(lent, sprate, startT):
    # in:
    # lent:生成时间序列的长度
    # sprate, 采样率
    # startT：给定时间序列起点，不需要时赋0.

    # if isinstance(sprate, int):
    #     sprate=200
    t = np.arange(0, lent, 1)
    tout = np.true_divide(t, sprate)
    # 存在起点时间的话，用起点时间生成
    if startT > 0:
        tout = tout + startT
    return tout
def B_delta(UD_trace, sprate):
    # UD_trace: n*1,ndarray,float
    # Ptrig:  no use ?
    # sprate: sampling rate,int

    c2 = 0.001
    A2 = 0
    dist = 0
    mag = 0

    # Ct method
    len1 = np.size(UD_trace, 0)
    try:
        row1 = np.shape(UD_trace)[1]
    except:
        row1 = 1
    # 检查输入，如果不对就转换为行向量
    if row1 > 1 and len1>1:
        print("b_delta数据纬度不满足要求")
        return -1,-1,-1
    if len1 > 1 and row1 == 1:  # n*1
        ud = UD_trace.transpose()  # 1*n

    # length of UD trace data
    lent = int(max([len1, row1]))
    if lent < 0.5 * sprate:
        # dispall('数据不足0.5s')
        return -1,-1,-1

    ud1 = ud[0:int(0.5 * sprate)]
    len32 = int(0.5 * sprate)
    t1 = cmp_t(len32, sprate, 0)
    yy32 = np.zeros(len32)

    for j in range(len32):
        udj = ud1[0:j + 1]
        udj1 = np.abs(udj)
        ss = max(udj1)
        yy32[j] = ss

    maxC2 = max(yy32)

    c2 = maxC2 / t1[-1]
    if c2 <= 0:
        c2 = 0.001

    # return c2, A2,  mag
    def equations(rbc):
        ppp = np.array([1.52925505016417,
                        -0.0205438355292372,
                        -1.309963826451])
        p0 = ppp[0]
        p1 = ppp[1]
        p2 = ppp[2]
        if rbc <= 0:
            rbc = 50
        dist = rbc
        try:
            tempdata = math.log(c2, 10) - p0 * math.log(dist, 10) - p1 * dist - p2
        except:
            tempdata = math.log(0.001, 10) - p0 * math.log(dist, 10) - p1 * dist - p2
        return tempdata

    distout = fsolve(equations, 50, maxfev=300)
    return c2, A2, distout


def Ct(data,distance_scaler,svm_regression_model):
    # CT值
    # 用于高频信号判断
    dist = 0
    data=data[:,0:400]
    trace = data[2,0:400]
    # [b,a]=signal.butter(4,[10*2/200,20*2/200],fs=200,btype='band')
    # filterdata=filtfilt(b,a,trace)
    traceNS=data[0,0:400]
    traceEW = data[1, 0:400]
    interval = 20
    abst = np.abs(trace)
    tempdata = np.zeros_like(abst)
    for i in range(0, len(abst), interval):
        max_value = np.max(abst[i:min(i + interval, len(abst))])  # 找到每个间隔内的最大值
        tempdata[i:min(i + interval, len(abst))] = max_value  # 将最大值赋给相应的索引位置

    log_amplitude = np.log10(np.abs(tempdata))

    x = np.arange(1, len(log_amplitude) + 1) / 200

    # 定义拟合函数
    def my_function(x, coefficients):
        return np.log(coefficients * x)

    # 定义初始参数猜测值
    initial_guess = [0.5]  # 假设初始参数猜测值为 [a, b]

    # 使用曲线拟合求解系数
    coefficients, _ = curve_fit(my_function, x, log_amplitude, p0=initial_guess)
    C = coefficients[0]
    # dist=10**(-0.493*np.log10(C)+1.826)
    # dist=10**(-1*np.log10(C)-)
    # def equation(delta):
    #     return np.log10(C) + 0.96 * np.log10(delta) - 2.28 + 3.7 /1000 * delta
    #      return np.log10(C) + 0.96 * np.log10(delta) - 2.28 + 3.7 /1000 * delta
    # # 使用 fsolve 求解方程
    # delta_solution = fsolve(equation, 1.0)
    # dist=delta_solution
    from Nature_Distance import Wavelet_energy,for_one
    wname = 'db9'
    level = 4
    trace1 = for_one(trace)
    w = wavelet.Wavelet(wname)
    wp = wavelet.WaveletPacket(trace1, w, mode='symmetric', maxlevel=level)
    # temp =wavelet.wpdec(trace1, level, wname)#调试
    # r = temp#调试
    # freqTree = [node.path for node in temp.get_level(3, 'freq')]  # 频谱由高到低
    r = Wavelet_energy(wp)
    # Kur_ud=kurtosis(data[2])
    # Sk_ud=skew(data[2])
    max_ud = np.max(np.abs(data[2]))
    tc_ud = TaoC(data[2], 400)
    tp_max_ud = TaoP(data[2], 0.99, 400)
    Eud = sum(np.abs(data[2]))
    parameter = np.hstack((
        max_ud, tc_ud, tp_max_ud, r, C, Eud))
    scaler=distance_scaler
    svr_model=svm_regression_model
    data2d=parameter.reshape(1,-1)
    new_data_scaled=scaler.transform(data2d)  # [:,0:-1]
    dist=svr_model.predict(new_data_scaled)
    dist=np.round(dist)
    return parameter, dist

def trace_info(traces, sprate):
    len = np.shape(traces)[1]
    baseinfo0430 = np.zeros((1, 5))
    if len < sprate * 1.5:
        return baseinfo0430
    if len > sprate * 1.5:
        trace1 = traces[:, 1:int(1.5 * sprate + 1)]
        Z = trace1[2]
    pga = np.max(trace1[0] ** 2 + trace1[1] ** 2 + trace1[2] ** 2)
    pgaZ = np.max(np.abs(Z))
    tc = TaoC(Z, sprate)
    tp_max = TaoP(Z, 0.99, sprate)
    c2, A2, distout = B_delta(Z, sprate)
    baseinfo0430 = np.array([[pga, pgaZ, tp_max, tc, c2]])
    return baseinfo0430


def TaoC(a_input, sprate):
    # % compute the TaoC max according to Allen in berkeley
    # % refer to paper  'A comparison of ?c and ?p  max for magnitude
    # %                  estimation in earthquake early warning'
    # %  in this study
    # % INPUT:    a,加速度记录 竖直向
    # %   sprate
    # %   area: 'sc'southern california
    # % OUTPUT:
    # %   M, magnitude
    # %   tc, taoC
    # % v=cmp_vel(a,sprate);
    # % v=high_butter(v,0.01*2/sprate); % 频率采用0.075hz的效果不好
    # % d = cmp_dis(a,sprate); % get displacement from velocity
    # % d=high_butter(d,0.01*2/sprate);
    tc = 0
    a = np.copy(a_input)
    v = iomega(a, sprate, 1)
    d = iomega(a, sprate, 2)
    r = np.sum(v ** 2) / np.sum(d ** 2)
    tc = 2 * np.pi / np.sqrt(r)
    # M=P2
    return tc


def TaoP(x_input, aa, sprate):
    # % compute the TaoC max according to Allen in berkeley
    # % refer to paper  'A comparison of ?c and ?p  max for magnitude
    # %                  estimation in earthquake early warning'
    # % a is a 1 sec smoothing constant,set to 0.99 in this study
    # % x is velocity data
    # % spr is sample rate
    x = np.copy(x_input)
    b = [0.993979467408589, -7.95183573926871, 27.8314250874405, -55.6628501748810, 69.5785627186012, -55.6628501748810, 27.8314250874405, -7.95183573926871, 0.993979467408589]#
    a = [1, -7.98792254578512, 27.9155307411736, -55.7468106999317, 69.5783814853248, -55.5787446629730, 27.7474644201712, -7.91589391960956, 0.987995181629862]#
    x = filter_matlab(b, a, x)
    len = np.size(x)
    difx = np.hstack((0, np.diff(x)))
    X = np.zeros(len)
    D = np.zeros(len)

    for i in range(1, len):
        X[i] = aa * X[i - 1] + x[i] ** 2
        D[i] = aa * D[i - 1] + (difx[i] * sprate) ** 2 + 0.0001
        if isinstance(D[i], complex):  # 判断num是否为复数类型
            D[i] = abs(D[i].real)
            # tp = 2 * np.pi * np.sqrt(X / D)
    try:
        D = D + 0.00001
        tp = 2 * np.pi * np.sqrt(X / D)
    except RuntimeWarning:
        print("error:function1,TaoP")

    tp_indarry = np.isnan(tp)
    ind = np.where(tp_indarry == False)
    tp = tp[ind]
    tp_max = np.max(tp)
    return tp_max


def LonLat(slon, slat, azi1, dis):
    # LONLAT Summary of this function goes here
    # in
    #  slon : 经度
    #  slat ： 维度
    #  azi1 ：方位角，度!!
    #  dis ：震中距，km
    # out
    #  lon:震源经度
    #  lat:震源纬度

    lat = -1
    lon = -1
    if dis < 0 or azi1 < 0:
        return lon, lat

    #  RN=6356.755;
    #  RE=6378.140;
    #  azi1=azi1*pi/180;
    #  lon=slon+dis*sin(azi1)*180/(pi*RE)
    #  lat=slat+dis*cos(azi1)*180/(pi*RN)

    azi1 = np.true_divide(azi1 * math.pi, 180)  # %弧度
    lat = slat + np.true_divide(dis, 111.195) * np.cos(azi1)  # 度
    rlatp = np.true_divide((lat + slat), 180) * np.true_divide(math.pi, 2)  # 弧度
    lon = slon + dis / np.cos(rlatp) / 111.195 * np.sin(azi1)  # 度
    # lon1 =slon+np.true_divide(dis*np.sin(azi1),np.cos(rlatp)*111.195)
    return lon, lat

def simpson(data):
    # 计算采样时间间隔 辛普森积分
    acceleration = data

    dt = 1 / 200  # 假设采样频率为200 Hz，即每隔0.05秒进行一次采样

    # 初始化速度向量
    velocity = np.zeros_like(acceleration)

    # 使用辛普森法求解速度
    for i in range(1, len(acceleration) - 1):
        velocity[i] = velocity[i - 1] + (acceleration[i - 1] + 4 * acceleration[i] + acceleration[i + 1]) * (dt / 6)

    return velocity

def trainidentify(data):
    # 平滑数据
    #不进行滤波操作
    UDAmp = data[:, 2]
    EWAmp = data[:, 0]
    NSAmp = data[:, 1]

    VHmax = np.max(np.abs(UDAmp) / np.sqrt(EWAmp ** 2 + NSAmp ** 2))

    # 低通滤波器
    b, a = signal.butter(8, 5 / (200 / 2), 'low')
    lowfreqSignal = filter_matlab(b, a, UDAmp)

    # 高通滤波器
    b1, a1 = signal(16, 30 / (200 / 2), 'high')
    highfreqSignal = filter_matlab(b1, a1, UDAmp)

    Rud = np.sum(np.abs(highfreqSignal)) / np.sum(np.abs(lowfreqSignal))

    if VHmax >= 1.07 and Rud <= 6:
        result = 1  # earthquake
    else:
        result = 0  # noise

    return result, VHmax, Rud


# import numpy as np
from scipy.signal import filtfilt
from scipy.fftpack import fft, ifft, fftshift, ifftshift


def iomega_azi1(datain, sprate, xx):
    datain = np.copy(datain)
    dt = 1 / sprate
    tempnum = np.shape(datain)[0]
    N = 2 ** nextpow2(tempnum)
    df = 1 / (N * dt)
    Nyq = 1 / (2 * dt)  # 奈奎斯特采样率
    temp = np.linspace(-Nyq, Nyq - df, N)
    iomega_arry = 1j * 2 * math.pi * temp
    iomega_exp = -xx
    [size1] = np.shape(datain)
    # b = [0.028,0.053,0.071,0.053,0.028]#HD
    # a = [1.000,-2.026,2.148,-1.159,0.279]# 没有反！！！HD
    # b,a=signal.butter(4,[75,100],fs=200,btype='band')
    b = [1.41272653624677e-07, 0, -5.65090614498706e-07, 0, 8.47635921748059e-07,
         0, -5.65090614498706e-07, 0, 1.41272653624677e-07]
    a = [1, -7.89154014671955, 27.2523197336544, -53.7909309665033,
         66.3737957621409, -52.4283619864010, 25.8891681521142,
         -7.30691593562765, 0.902465387346559]
    if (N - size1) != 0:
        datain = np.hstack([datain, np.zeros((N - size1), dtype=float)])
    tempdatain = signal.filtfilt(b,a,datain)
    A = np.fft.fft(tempdatain, N)
    A = np.fft.fftshift(A)
    for j in range(N):
        if iomega_arry[j] != 0:
            A[j] = A[j] * (iomega_arry[
                               j] ** iomega_exp)
        else:
            A[j] = np.array([0 + 0 * j],
                            dtype=complex)

    A = np.fft.ifftshift(A)
    datain = np.fft.ifft(A)

    dataout = np.real(datain[0:size1])
    # tempdatain = signal.filtfilt(b,a,datain)
    return dataout

Rdelta = np.array([[5.000, 10.000, 15.000, 20.000, 25.000, 30.000, 35.000,
                    40.000, 45.000, 50.000, 55.000, 60.000, 70.000, 75.000, 85.000,
                    90.000, 100.000, 110.000, 120.000, 130.000, 140.000, 150.000, 160.000,
                    170.00, 180.000, 190.000, 200.000, 210.000, 220.000, 230.000, 240.000,
                    250.000, 260.000, 270.000, 280.000, 270.000, 280.000, 290.000, 300.000,
                    310.000, 320.000, 330.000, 340.000, 350.000, 360.000, 370.000, 380.000,
                    390.000, 400.000, 420.000, 430.000, 440.000, 450.000, 460.000, 470.000,
                    500.000, 510.000, 530.000, 540.000, 550.000, 560.000, 570.000, 580.000,
                    600.000, 610.000, 620.000, 650.000, 700.000, 750.000, 800.000, 850.000, 900.000, 1000.000],
                   [2.000, 2.000, 2.100, 2.200, 2.400, 2.600, 2.700, 2.800, 2.900, 3.000, 3.100, 3.200, 3.200, 3.300
                       , 3.300, 3.400, 3.400, 3.500, 3.500, 3.600, 3.600, 3.700, 3.700, 3.800, 3.800, 3.900, 3.900,
                    3.900, 3.900, 4.000, 4.000, 4.000, 4.100, 4.200, 4.100, 4.200, 4.100, 4.200, 4.300, 4.400, 4.400, 4.500,
                    4.500, 4.500, 4.500, 4.500, 4.600, 4.600, 4.700, 4.700, 4.800, 4.800, 4.800, 4.800, 4.800, 4.800, 4.900,
                    4.900, 4.900, 4.900, 4.900, 4.900, 4.900, 4.900, 5.000, 5.000, 5.100, 5.200, 5.200, 5.200, 5.200, 5.300, 5.300]])
Rdelta = Rdelta.transpose()

import numpy as np


def snr(Package, Buffer,StartT,P_time,mod, flag=None):
    # (Sta_vars1.Package,Sta_vars1.Buffer,Sta_vars1.StartT,Sta_vars1.P_time,mod, flag=None):
    """
    计算信号噪声比（SNR），支持两种计算模式：SVD分解法（未启用）、能量比法

    参数说明：20251010
    ----------
    Sta_vars1 : 类/字典对象
        需包含以下属性/键：
        - P_time : float，信号起始时间（若<0则直接返回[-1,-1,-1]）
        - StartT : float，数据起始时间
        - Package : 序列，数据包相关数据（用于计算长度LP）
        - Buffer : 2D numpy数组，数据缓冲区（核心计算数据来源）
    mod : int
        计算模式标识：
        - 1 : SVD分解法（原MATLAB中为注释状态，Python中保持注释，暂不启用）
        - 2 : 能量比法（核心启用功能，计算EW/NS/UD三个方向的SNR）
    flag : int, 可选
        控制标志：若flag==1则直接返回[-1,-1,-1]

    返回值：
    ----------
    result : list
        长度为3的列表，依次为 [EW方向SNR, NS方向SNR, UD方向SNR]
        若计算条件不满足（如数据长度不足），返回初始值[-1,-1,-1]
    """
    # 初始化返回结果（默认无效值）
    result = [-1, -1, -1]

    # 1. 标志位判断：若flag存在且为1，直接返回无效值
    if flag is not None:
        if flag == 1:
            return result

    # 2. 时间有效性判断：P_time<0时无有效数据，返回无效值
    if P_time < 0:
        return result

    # 3. 基础参数计算：数据长度与时间差
    LP = len(Package)  # Package的长度（原MATLAB length函数）
    BL = Buffer.shape[0]  # Buffer的行数（原MATLAB length(Buffer)）
    deltatime = StartT - P_time  # 时间差

    # --------------------------
    # 模式1：SVD分解法（原MATLAB注释，暂不启用）
    # --------------------------
    if mod == 1:
        # 以下为原MATLAB逻辑的Python注释版本，如需启用可解除注释并调试
        # # 计算数据起始索引（P_time前5秒 或 Buffer前2000点）
        # ind1 = int(np.floor(deltatime * 200 + LP + 5 * 200))
        # if ind1 > BL:
        #     ind1 = 2000
        # # 提取Buffer中指定范围的数据
        # data = Buffer[BL - ind1 :, :]
        # # 构建三个方向的矩阵（UD/EW/NS）
        # matrixud = data[:, [2, 5]]  # UD方向：第3列（索引2）、第6列（索引5）
        # matrixew = data[:, [0, 3]]  # EW方向：第1列（索引0）、第4列（索引3）
        # matrixns = data[:, [1, 4]]  # NS方向：第2列（索引1）、第5列（索引4）
        # # SVD分解（仅需奇异值矩阵S）
        # _, Sud, _ = np.linalg.svd(matrixud)
        # _, Sew, _ = np.linalg.svd(matrixew)
        # _, Sns, _ = np.linalg.svd(matrixns)
        # # 计算奇异值的平方（能量）
        # sigmaud = np.diag(Sud) ** 2
        # sigmaew = np.diag(Sew) ** 2
        # sigmans = np.diag(Sns) ** 2
        # # 计算信号能量（Es）与噪声能量（En）
        # L = len(sigmaud)
        # Esud = sigmaud[0, 0] - np.sum(sigmaud[1:]) / (L - 1)
        # Enud = np.sum(sigmaud[1:]) / (L - 1)
        # Esew = sigmaew[0, 0] - np.sum(sigmaew[1:]) / (L - 1)
        # Enew = np.sum(sigmaew[1:]) / (L - 1)
        # Esns = sigmans[0, 0] - np.sum(sigmans[1:]) / (L - 1)
        # Enns = np.sum(sigmans[1:]) / (L - 1)
        # # 计算SNR（能量比，未转dB）
        # snrud = Esud / Enud
        # snrew = Esew / Enew
        # snrns = Esns / Enns
        # # 赋值结果（EW, NS, UD顺序）
        # result = [snrew, snrns, snrud]
        return result

    # --------------------------
    # 模式2：能量比法（核心启用功能，计算dB值）
    # --------------------------
    elif mod == 2:
        # 计算信号与噪声的索引范围（基于采样率200Hz：200点=1秒）
        plen = deltatime * 200 + LP  # 信号数据总长度（点）
        pind = BL - plen  # 信号起始索引（Buffer的行索引）
        pend = pind + 200 * 2 - 1  # 信号结束索引（2秒信号：200*2点）
        pbef = pind - 200 * 5 - 1  # 噪声起始索引（5秒噪声：200*5点，在信号前）

        # 数据长度有效性判断：噪声起始索引<1 或 信号结束索引>Buffer长度 → 无效
        if pend >= BL or pbef < 0:  # Python索引从0开始，故pbef<0而非<1
            return result

        # --------------------------
        # 1. 提取信号与噪声数据（按方向分2组通道）
        # --------------------------
        # EW方向：通道1（列0）、通道2（列3）
        signalew1 = Buffer[pind:pend + 1, 0]  # 信号1（Python切片左闭右开，需+1）
        signalew2 = Buffer[pind:pend + 1, 3]  # 信号2
        noiseew1 = Buffer[pbef:pind + 1, 0]  # 噪声1
        noiseew2 = Buffer[pbef:pind + 1, 3]  # 噪声2

        # NS方向：通道1（列1）、通道2（列5）
        signalns1 = Buffer[pind:pend + 1, 1]
        signalns2 = Buffer[pind:pend + 1, 4]
        noisens1 = Buffer[pbef:pind + 1, 1]
        noisens2 = Buffer[pbef:pind + 1, 4]

        # UD方向：通道1（列2）、通道2（列6）
        signalud1 = Buffer[pind:pend + 1, 2]
        signalud2 = Buffer[pind:pend + 1, 5]
        noiseud1 = Buffer[pbef:pind + 1, 2]
        noiseud2 = Buffer[pbef:pind + 1, 5]

        # --------------------------
        # 2. 计算时间长度（秒）：点数 / 采样率（200Hz）
        # --------------------------
        len_signal = len(signalew1) / 200  # 信号长度（所有信号通道长度一致）
        len_noise = len(noiseew1) / 200  # 噪声长度（所有噪声通道长度一致）

        # --------------------------
        # 3. 计算每个通道的SNR（10*log10(信号功率/噪声功率)，单位dB）
        # 功率 = 能量 / 时间长度 → 信号功率/噪声功率 = (信号能量/len_signal) / (噪声能量/len_noise)
        # --------------------------
        # UD方向SNR（2通道平均）
        snrud1 = 10 * np.log10((np.sum(signalud1 ** 2) / len_signal) / (np.sum(noiseud1 ** 2) / len_noise))
        snrud2 = 10 * np.log10((np.sum(signalud2 ** 2) / len_signal) / (np.sum(noiseud2 ** 2) / len_noise))
        snrud = (snrud1 + snrud2) / 2

        # NS方向SNR（2通道平均）
        snrns1 = 10 * np.log10((np.sum(signalns1 ** 2) / len_signal) / (np.sum(noisens1 ** 2) / len_noise))
        snrns2 = 10 * np.log10((np.sum(signalns2 ** 2) / len_signal) / (np.sum(noisens2 ** 2) / len_noise))
        snrns = (snrns1 + snrns2) / 2

        # EW方向SNR（2通道平均）
        snrew1 = 10 * np.log10((np.sum(signalew1 ** 2) / len_signal) / (np.sum(noiseew1 ** 2) / len_noise))
        snrew2 = 10 * np.log10((np.sum(signalew2 ** 2) / len_signal) / (np.sum(noiseew2 ** 2) / len_noise))
        snrew = (snrew1 + snrew2) / 2

        # 赋值最终结果（EW, NS, UD顺序，与原MATLAB一致）
        result = [snrew, snrns, snrud]
        return result

    # 若mod既不是1也不是2，返回无效值
    return result


def ef_udcmp(traces_org):
    """
    等效于MATLAB中的Ef_UDcmp函数，计算6列数据的分段频域能量（UD方向，0-55Hz每5Hz一段）

    参数:
        traces_org (np.ndarray): 输入6列数据（对应MATLAB的traces_org）
                                 列索引：0=EW1, 1=NS1, 2=UD1, 3=EW2, 4=NS2, 5=UD2

    返回:
        ef_ud (np.ndarray): 11个分段的能量值（对应0-5,5-10,...,50-55Hz）
                           若不满足分段条件（n5*11>频率点数量），返回空数组
    """
    # 1. 对6列数据的指定列执行Fourier变换（采样率200Hz，正常运行flag=0）
    freq_ud1 = Fourier(traces_org[:, 2], sprate=200, flag=0)  # UD1：第3列（MATLAB索引3→Python索引2）
    freq_ns1 = Fourier(traces_org[:, 1], sprate=200, flag=0)  # NS1：第2列（MATLAB索引2→Python索引1）
    freq_ew1 = Fourier(traces_org[:, 0], sprate=200, flag=0)  # EW1：第1列（MATLAB索引1→Python索引0）
    freq_ud2 = Fourier(traces_org[:, 5], sprate=200, flag=0)  # UD2：第6列（MATLAB索引6→Python索引5）
    freq_ns2 = Fourier(traces_org[:, 4], sprate=200, flag=0)  # NS2：第5列（MATLAB索引5→Python索引4）
    freq_ew2 = Fourier(traces_org[:, 3], sprate=200, flag=0)  # EW2：第4列（MATLAB索引4→Python索引3）

    # 2. 找到频率>5Hz的第一个索引（对应MATLAB的find(freq_ud1(1,:)>5,1)）
    # freq_ud1[0,:]为频率序列（第1行），np.argwhere返回满足条件的索引，取第一个
    pos = np.argwhere(freq_ud1[0, :] > 5).flatten()[0] if np.any(freq_ud1[0, :] > 5) else len(freq_ud1[0, :])
    n5 = pos - 1  # 0-5Hz的频率点数量（对应MATLAB的n5=pos-1）

    # 3. 检查分段有效性（确保n5*11不超过频率点总数，避免索引越界）
    freq_len = len(freq_ud1[0, :])  # 频率序列的长度（所有freq_*的长度一致）
    if n5 * 11 > freq_len:
        print("分段条件不满足（n5*11 > 频率点数量），无法计算分段能量")
        return np.array([])

    # 4. 计算UD方向各分段能量（0-5,5-10,...,50-55Hz，共11段）
    # 每段取freq_ud1和freq_ud2的振幅谱均值（对应MATLAB的(sum(...) + sum(...))/2）
    eta_ud_1 = (np.sum(freq_ud1[1, 0:n5]) + np.sum(freq_ud2[1, 0:n5])) / 2  # 0-5Hz
    eta_ud_5 = (np.sum(freq_ud1[1, n5:n5 * 2]) + np.sum(freq_ud2[1, n5:n5 * 2])) / 2  # 5-10Hz
    eta_ud_10 = (np.sum(freq_ud1[1, n5 * 2:n5 * 3]) + np.sum(freq_ud2[1, n5 * 2:n5 * 3])) / 2  # 10-15Hz
    eta_ud_15 = (np.sum(freq_ud1[1, n5 * 3:n5 * 4]) + np.sum(freq_ud2[1, n5 * 3:n5 * 4])) / 2  # 15-20Hz
    eta_ud_20 = (np.sum(freq_ud1[1, n5 * 4:n5 * 5]) + np.sum(freq_ud2[1, n5 * 4:n5 * 5])) / 2  # 20-25Hz
    eta_ud_25 = (np.sum(freq_ud1[1, n5 * 5:n5 * 6]) + np.sum(freq_ud2[1, n5 * 5:n5 * 6])) / 2  # 25-30Hz
    eta_ud_30 = (np.sum(freq_ud1[1, n5 * 6:n5 * 7]) + np.sum(freq_ud2[1, n5 * 6:n5 * 7])) / 2  # 30-35Hz
    eta_ud_35 = (np.sum(freq_ud1[1, n5 * 7:n5 * 8]) + np.sum(freq_ud2[1, n5 * 7:n5 * 8])) / 2  # 35-40Hz
    eta_ud_40 = (np.sum(freq_ud1[1, n5 * 8:n5 * 9]) + np.sum(freq_ud2[1, n5 * 8:n5 * 9])) / 2  # 40-45Hz
    eta_ud_45 = (np.sum(freq_ud1[1, n5 * 9:n5 * 10]) + np.sum(freq_ud2[1, n5 * 9:n5 * 10])) / 2  # 45-50Hz
    eta_ud_50 = (np.sum(freq_ud1[1, n5 * 10:n5 * 11]) + np.sum(freq_ud2[1, n5 * 10:n5 * 11])) / 2  # 50-55Hz

    # 5. 组合结果（11个分段能量，对应MATLAB的Ef_UD）
    ef_ud = np.array([
        eta_ud_1, eta_ud_5, eta_ud_10, eta_ud_15, eta_ud_20,
        eta_ud_25, eta_ud_30, eta_ud_35, eta_ud_40, eta_ud_45, eta_ud_50
    ])

    return ef_ud


# ------------------- 测试用例（验证功能正确性）-------------------
if __name__ == "__main__":
    # 生成模拟的6列输入数据（长度1000，符合原代码数据格式）
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    traces_org = np.random.randn(1000, 6)  # 1000行6列的随机数据（模拟振动/信号数据）

    # 调用ef_udcmp计算分段频域能量
    ef_ud_result = ef_udcmp(traces_org)

    # 输出结果
    if len(ef_ud_result) > 0:
        print("UD方向分段频域能量结果（对应0-5,5-10,...,50-55Hz）：")
        freq_bands = [f"{5 * i}-{5 * (i + 1)}Hz" for i in range(11)]
        for band, energy in zip(freq_bands, ef_ud_result):
            print(f"{band:10s}: {energy:.6f}")