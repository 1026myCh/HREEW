import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from iomega import iomega
from srs_integration import srs_integration, srs_integrationV


def cal_magnitude(Traces_evt_org, Distance, beM, Sprate, Debug, Stime, flagarea, PGAold, PGAnew, flag_Sfirst, in_model,
                 Gain, flag=None):
    """
    P波和S波震级计算函数
    参数说明与原始MATLAB函数一致
    """
    result = -1.0  # 初始化震级结果
    org_Distance = Distance
    P_ud_m = 0.0
    TC = 0.0
    sAvghm = 0.0

    # 条件检查1
    if Stime > 0 and abs(PGAold - PGAnew) < 0.001 and flag_Sfirst < 1:
        result = beM
        return result,P_ud_m,TC,sAvghm

    # 数据长度检查
    if Traces_evt_org.shape[1] < 2 * Sprate:
        return result,P_ud_m,TC,sAvghm

    # 距离修正逻辑
    try:
        if 20 < Distance < 55 and Stime > 0:
            pass  # 保持原距离
        elif 0 < Distance <= 50 and Stime > 0:
            Distance = np.sqrt(Distance ** 2 + 20 ** 2)
    except Exception:
        Distance = org_Distance

    if not np.isreal(Distance):
        Distance = org_Distance

    # 数据预处理
    singlen = Traces_evt_org.shape[1]
    Traces_evt = Traces_evt_org - np.mean(Traces_evt_org, axis=1, keepdims=True)

    # 初始化处理
    if beM == -1:
        beM = 0.0
    if flag == 1:
        return result,P_ud_m,TC,sAvghm
    if flagarea is None:
        flagarea = 3
    if Traces_evt.shape[1] < Sprate * 0.1:
        return result,P_ud_m,TC,sAvghm

    # ====================== P波处理 (Stime <= 0) ======================
    if Stime <= 0 and Distance > 25:
        if Distance <= 0:
            return result,P_ud_m,TC,sAvghm

        num = Traces_evt.shape[1]
        Distance = np.round(Distance, 1)
        lent = Traces_evt.shape[1]

        # 设计滤波器
        b, a = butter(2, 0.075 * 2 / 200, 'high')
        b1, a1 = butter(2, 15 * 2 / 200, 'low')

        if lent >= 500:
            # 传感器1处理
            ud1 = lfilter(b, a, Traces_evt[2][0:int(2.5 * Sprate)])
            ud1 = lfilter(b1, a1, ud1)
            v1 = Simpson1(ud1)
            disp1 = Simpson1(v1)
            d1 = np.max(np.abs(disp1))
            Magnitude11 = (np.log10(d1 * 10000) + 1.2 * np.log10(Distance) +
                           5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.72

            # 陶茨法计算震级
            mag_tc1, tc1 = TaoC(Traces_evt[2][0:int(2.5 * Sprate)], Sprate, 'sc', 2, 0)
            temp = [Magnitude11]
            Magnitude1 = np.min(temp)
            tempM = [mag_tc1]
            minM = np.min(tempM)

            # 双传感器处理
            if num > 3:
                ud2 = lfilter(b, a, Traces_evt[5][0:int(2.5 * Sprate)])
                ud2 = lfilter(b1, a1, ud2)
                v2 = Simpson1(ud2)
                disp2 = Simpson1(v2)
                d2 = np.max(np.abs(disp2))
                Magnitude22 = (np.log10(d2 * 10000) + 1.2 * np.log10(Distance) +
                               5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.72
                mag_tc2, tc2 = TaoC(Traces_evt[5][0:int(2.5 * Sprate)], Sprate, 'sc', 2, 0)
                temp = [Magnitude11, Magnitude22]
                Magnitude1 = np.min(temp)
                tempM = [mag_tc1, mag_tc2]
                minM = np.min(tempM)

            Magnitude2 = minM

        else:  # 数据长度<500
            ud1 = lfilter(b, a, Traces_evt[2][:])
            ud1 = lfilter(b1, a1, ud1)
            v1 = Simpson1(ud1)
            disp1 = Simpson1(v1)
            d1 = np.max(np.abs(disp1))
            d1 = np.max([1e-12,d1])
            Magnitude11 = (np.log10(d1 * 10000) + 1.2 * np.log10(Distance) +
                           5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.78
            Magnitude1 = Magnitude11
            mag_tc1, tc1 = TaoC(Traces_evt[2][:], Sprate, 'sc', 2, 0)
            minM = mag_tc1
            Magnitude2 = minM

            if num > 3:  # 双传感器
                ud2 = lfilter(b, a, Traces_evt[5][:])
                ud2 = lfilter(b1, a1, ud2)
                v2 = Simpson1(ud2)
                disp2 = Simpson1(v2)
                d2 = np.max(np.abs(disp2))
                d2 = np.max([1e-12, d2])
                Magnitude22 = (np.log10(d2 * 10000) + 1.2 * np.log10(Distance) +
                               5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.78
                temp = [Magnitude11, Magnitude22]
                Magnitude1 = np.min(temp)
                mag_tc2, tc2 = TaoC(Traces_evt[5][:], Sprate, 'sc', 2, 0)
                tempM = [mag_tc1, mag_tc2]
                minM = np.min(tempM)
                Magnitude2 = minM

        # 结果融合
        deltM = abs(Magnitude1 - Magnitude2)
        # MagnitudeFinal1 = Magnitude1 * 0.4 + Magnitude2 * 0.6
        temp111 = [Magnitude1, Magnitude2]
        MagnitudeFinal1 = max(temp111) * 0.4 + min(temp111) * 0.6
        TC = (tc1 + tc2)/2
        P_ud_m = (d1 + d2)/2 # 竖向位移

        if deltM >= 2:
            MagnitudeFinal1 = min(Magnitude1, Magnitude2)

        #
        PGA1 = np.max(Traces_evt[0][:] ** 2 + Traces_evt[1][:] ** 2 + Traces_evt[2][:] ** 2)
        PGA2 = np.max(Traces_evt[3][:] ** 2 + Traces_evt[4][:] ** 2 + Traces_evt[5][:] ** 2)

        if Distance > 200 and PGA1 < 5 and PGA2 < 5 and MagnitudeFinal1 > 5.5 and Stime < 0:
            MagnitudeFinal1 = min(Magnitude1, Magnitude2)
            if MagnitudeFinal1 > 5.5:
                MagnitudeFinal1 = 5.0

        # 最终结果处理
        result = np.round(MagnitudeFinal1, 1)

        if result <= 3.5:
            mag_tc1, tc1 = TaoC(Traces_evt[2][:], Sprate, 'sc', 0, 0)
            mag_tc2, tc2 = TaoC(Traces_evt[5][:], Sprate, 'sc', 0, 0)
            result = np.mean([mag_tc1, mag_tc2])
        elif result > 5.5:
            # 大震特殊处理
            if Traces_evt.shape[1] <= 600:
                mag_tc2, tc2 = TaoC(Traces_evt[5][:], Sprate, 'sc', 1, 0)
                mag_tc1, tc1 = TaoC(Traces_evt[2][:], Sprate, 'sc', 1, 0)
                b, a = butter(2, 0.075 * 2 / 200, 'high')
                b1, a1 = butter(2, 15 * 2 / 200, 'low')

                ud1 = lfilter(b, a, Traces_evt[2][:])
                ud1 = lfilter(b1, a1, ud1)
                ud2 = lfilter(b, a, Traces_evt[5][:])
                ud2 = lfilter(b1, a1, ud2)

                v1 = Simpson1(ud1)
                d1 = np.max(np.abs(Simpson1(v1)))
                Mag1 = (np.log10(d1 * 10000) + 1.2 * np.log10(Distance) +
                        5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.78
                v2 = Simpson1(ud2)
                d2 = np.max(np.abs(Simpson1(v2)))
                Mag2 = (np.log10(d2 * 10000) + 1.2 * np.log10(Distance) +
                        5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.78
                Taoc_mag = min(mag_tc2, mag_tc1)
                Pdmax_mag = min(Mag1, Mag2)
                result = (Taoc_mag + Pdmax_mag) / 2
            else:
                # 长数据特殊处理
                mag_tc2, tc2 = TaoC(Traces_evt[5][0:int(3 * Sprate)], Sprate, 'sc', 1, 0)
                mag_tc1, tc1 = TaoC(Traces_evt[2][0:int(3 * Sprate)], Sprate, 'sc', 1, 0)
                b, a = butter(2, 0.075 * 2 / 200, 'high')
                b1, a1 = butter(2, 15 * 2 / 200, 'low')

                ud1 = lfilter(b, a, Traces_evt[2][:])
                ud1 = lfilter(b1, a1, ud1)
                ud2 = lfilter(b, a, Traces_evt[5][:])
                ud2 = lfilter(b1, a1, ud2)

                v1 = Simpson1(ud1)
                d1 = np.max(np.abs(Simpson1(v1)))
                Mag1 = (np.log10(d1 * 10000) + 1.2 * np.log10(Distance) +
                        5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.78
                v2 = Simpson1(ud2)
                d2 = np.max(np.abs(Simpson1(v2)))
                Mag2 = (np.log10(d2 * 10000) + 1.2 * np.log10(Distance) +
                        5e-4 * Distance - 5e-3 * 10 + 0.46) / 0.78
                Taoc_mag = min(mag_tc2, mag_tc1) - 0.5
                Pdmax_mag = min(Mag1, Mag2) - 0.5
                result = (Taoc_mag + Pdmax_mag) / 2
        TC = (tc1 + tc2) / 2
    # ====================== S波处理 (Stime > 0) ======================
    elif Stime > 0:
        Rdelta = RRdelta(flagarea)
        idx = np.where(Distance < Rdelta[:, 0])[0]
        if len(idx) == 0:
            return result,P_ud_m,TC,sAvghm
        ind = idx[0]
        ind1 = ind - 1 if ind > 0 else 0
        delta = Rdelta[ind1, 1]

        # 响应补偿 (简化为直接使用原始数据)
        Traces_evt1 = np.zeros_like(Traces_evt)
        Traces_evt1[0][:] = respoff(Traces_evt[0][:],Sprate,in_model) # respoff未实现
        Traces_evt1[3][:] = respoff(Traces_evt[3][:],Sprate,in_model)  # 简化为原始数据
        Traces_evt1[1][:] = respoff(Traces_evt[1][:],Sprate,in_model)
        Traces_evt1[4][:] = respoff(Traces_evt[4][:],Sprate,in_model)

        # 带通滤波
        b, a = butter(4, [0.1, 20], btype='bandpass', fs=200)
        EW1F = lfilter(b, a, Traces_evt1[0][:])
        EW2F = lfilter(b, a, Traces_evt1[3][:])
        NS1F = lfilter(b, a, Traces_evt1[1][:])
        NS2F = lfilter(b, a, Traces_evt1[4][:])

        # 位移计算 (简化为双积分)
        displaceEW1 = srs_integration(EW1F, Sprate, Distance)
        displaceEW2 = srs_integration(EW2F, Sprate, Distance)
        displaceNS1 = srs_integration(NS1F, Sprate, Distance)
        displaceNS2 = srs_integration(NS2F, Sprate, Distance)

        # 计算最大位移
        disp_maxEW1 = np.max(np.abs(displaceEW1))
        disp_maxNS1 = np.max(np.abs(displaceNS1))
        disp_maxEW2 = np.max(np.abs(displaceEW2))
        disp_maxNS2 = np.max(np.abs(displaceNS2))

        # 震级计算
        maxA1_h1 = (disp_maxEW1 + disp_maxNS1) / 2
        maxA1_h2 = (disp_maxNS2 + disp_maxEW2) / 2
        sAvghm = (disp_maxEW1 + disp_maxNS1 + disp_maxNS2 + disp_maxEW2) / 4
        Magnitude111_h = np.log10(maxA1_h1 * 10000) + delta
        Magnitude222_h = np.log10(maxA1_h2 * 10000) + delta
        result = (Magnitude111_h + Magnitude222_h) / 2

        # 首报处理
        if flag_Sfirst >= 1 and beM > 0:
            if beM > result:
                result = (result + beM) / 2

        # 增益调整
        result = np.round(result, 1) + Gain

    # ====================== 近距离处理 (Distance <= 25) ======================
    elif Stime <= 0 and Distance <= 25:
        # 位移计算 (简化为双积分)
        displaceUD1 = iomega(Traces_evt[2][:], Sprate, 2, Debug, 0)
        displaceUD2 = iomega(Traces_evt[5][:], Sprate, 2, Debug, 0)
        # displaceUD1 = srs_integration(Traces_evt[2, :], Sprate, 0)
        # displaceUD2 = srs_integration(Traces_evt[5, :], Sprate, 0)

        P_ud_m = (max(abs(displaceUD1)) + max(abs(displaceUD2))) / 2
        # 计算最大位移
        maxdispUD1 = np.max(np.abs(displaceUD1[int(0.1 * Sprate):]))
        maxdispUD2 = np.max(np.abs(displaceUD2[int(0.1 * Sprate):]))

        # 震级计算
        Rdelta = RRdelta(flagarea)
        idx = np.where(Distance < Rdelta[:, 0])[0]
        if len(idx) == 0:
            return result,P_ud_m,TC,sAvghm
        ind = idx[0]
        ind1 = ind - 1 if ind > 0 else 0
        delta = Rdelta[ind1, 1]

        Magnitude111 = np.log10(maxdispUD1 * 10000) + delta
        Magnitude222 = np.log10(maxdispUD2 * 10000) + delta
        result = (Magnitude111 + Magnitude222) / 2
        result = fitM(result, 0)
        result = np.round(result, 1)

    # ====================== 后处理逻辑 ======================
    if Stime < 0 and beM > 0 and result < beM:
        result = beM
    elif Stime > 0 and beM > 0 and result - beM < 0 and flag_Sfirst < 1:
        result = beM

    return result,P_ud_m,TC,sAvghm


# ====================== 辅助函数 ======================
def fitM(x, flag=None):
    """
    多项式计算函数

    参数:
        x: 输入值
        flag: 控制标志 (可选)
            1: 直接返回0
            其他值或未提供: 计算多项式

    返回:
        多项式计算结果 (若 flag==1 则返回0)
    """
    # 如果提供了 flag 参数且值为 1，直接返回 0
    if flag is not None and flag == 1:
        return 0

    # 多项式系数 (来自未注释的MATLAB参数)
    p1 = -0.04941
    p2 = 0.7157
    p3 = -2.211
    p4 = 4.243

    # 计算三次多项式: p1*x^3 + p2*x^2 + p3*x + p4
    result = p1 * x ** 3 + p2 * x ** 2 + p3 * x + p4

    return result

def Simpson1(data):
    """
    使用辛普森法则对加速度数据进行积分得到速度
    """
    dt = 1 / 200.0  # 采样间隔
    n = len(data)
    velocity = np.zeros(n)

    # 偶数索引处理
    for i in range(1, n - 2, 2):
        velocity[i + 1] = velocity[i - 1] + (data[i] + 4 * data[i + 1] + data[i + 2]) * dt / 3

    # 奇数索引处理
    for i in range(2, n - 1, 2):
        velocity[i + 1] = velocity[i - 1] + (data[i - 1] + 4 * data[i] + data[i + 1]) * dt / 3

    return velocity


def TaoC(a, sprate, area, Debug, flag):
    """
    陶茨法计算震级
    """
    if flag == 1:
        return 0, 0

    # 大震处理
    if Debug == 1:
        b1, a1 = butter(2, 20 * 2 / 200, 'low')
        b2, a2 = butter(2, 0.5 * 2 / 200, 'low')
        a_filt = lfilter(b2, a2, a)
        a_filt = lfilter(b1, a1, a_filt)
        v = Simpson1(a_filt)
        v = lfilter(b2, a2, v)
        d = Simpson1(v)
        # filtfilt()
        r = np.sum(v ** 2) / np.sum(d ** 2)
        r1 = np.sqrt(r)
        if not np.isreal(r1):
            r1 = 1
        tc = 2 * np.pi / r1
        M = 2.94 * np.log10(tc) + 5.26 - 0.62

    # 小震处理
    elif Debug == 2:
        # v = iomega(a, sprate, 1, Debug, 0)
        # d = iomega(a, sprate, 2, Debug, 0)
        v = srs_integrationV(a, sprate)
        d = srs_integration(a, sprate, 0)
        r = np.sum(v ** 2) / np.sum(d ** 2)
        r1 = np.sqrt(r)
        if not np.isreal(r1):
            r1 = 1
        tc = 2 * np.pi / r1
        M = 2.94 * np.log10(tc) + 5.26 + 0.3

    # 默认处理
    else:
        # v = iomega(a, sprate, 1, Debug, 0)
        # d = iomega(a, sprate, 2, Debug, 0)
        v = srs_integrationV(a, sprate)
        d = srs_integration(a, sprate, 0)
        r = np.sum(v ** 2) / np.sum(d ** 2)
        r1 = np.sqrt(r)
        if not np.isreal(r1):
            r1 = 1
        tc = 2 * np.pi / r1
        period = tc
        M = 0.4046 * period ** 2 + 1.2767 * period + 2.7713

    # if M>8:
    #     print('out!!!!')
    return M, tc


def RRdelta(flagarea):
    """
    区域校正值查询表
    """
    if flagarea == 1:  # R11
        return np.array([
            [5, 1.9], [10, 2], [15, 2.2], [20, 2.3], [25, 2.5], [30, 2.7], [35, 2.9],
            [40, 2.9], [45, 3.0], [50, 3.1], [55, 3.2], [60, 3.3], [70, 3.3], [75, 3.4],
            [85, 3.3], [90, 3.4], [100, 3.4], [110, 3.5], [120, 3.5], [130, 3.6],
            [140, 3.6], [150, 3.7], [160, 3.7], [170, 3.8], [180, 3.8], [190, 3.9],
            [200, 3.9], [210, 3.9], [220, 3.9], [230, 4.0], [240, 4.1], [250, 4.1],
            [260, 4.1], [270, 4.2], [280, 4.2], [290, 4.3], [300, 4.2], [310, 4.3],
            [320, 4.3], [330, 4.4], [340, 4.4], [350, 4.4], [360, 4.5], [370, 4.5],
            [380, 4.5], [390, 4.5], [400, 4.6], [420, 4.6], [430, 4.6], [440, 4.6],
            [450, 4.6], [460, 4.6], [470, 4.6], [500, 4.8], [510, 4.8], [530, 4.8],
            [540, 4.8], [550, 4.8], [560, 4.9], [570, 4.9], [580, 4.9], [600, 4.9],
            [610, 5], [620, 5], [650, 5.1], [700, 5.2], [750, 5.2], [800, 5.2],
            [850, 5.2], [900, 5.3], [1000, 5.3]
        ])
    # 其他区域类似实现
    # ...
    else:  # 默认R13
        return np.array([
            [5, 2.0], [10, 2.0], [15, 2.1], [20, 2.2], [25, 2.4], [30, 2.6], [35, 2.7],
            [40, 2.8], [45, 2.9], [50, 3.0], [55, 3.1], [60, 3.2], [70, 3.2], [75, 3.3],
            [85, 3.3], [90, 3.4], [100, 3.4], [110, 3.5], [120, 3.5], [130, 3.6],
            [140, 3.6], [150, 3.7], [160, 3.7], [170, 3.8], [180, 3.8], [190, 3.9],
            [200, 3.9], [210, 3.9], [220, 3.9], [230, 4.0], [240, 4.0], [250, 4.0],
            [260, 4.1], [270, 4.2], [280, 4.1], [290, 4.2], [300, 4.3], [310, 4.4],
            [320, 4.4], [330, 4.5], [340, 4.5], [350, 4.5], [360, 4.5], [370, 4.5],
            [380, 4.6], [390, 4.6], [400, 4.7], [420, 4.7], [430, 4.8], [440, 4.8],
            [450, 4.8], [460, 4.8], [470, 4.8], [500, 4.8], [510, 4.9], [530, 4.9],
            [540, 4.9], [550, 4.9], [560, 4.9], [570, 4.9], [580, 4.9], [600, 4.9],
            [610, 5], [620, 5], [650, 5.1], [700, 5.2], [750, 5.2], [800, 5.2],
            [850, 5.2], [900, 5.3], [1000, 5.3]
        ])


# # ====================== 未实现函数的占位 ======================
# def iomega(data, sprate, order, debug, flag):
#     """
#     积分函数 (需要根据实际需求实现)
#     此处简化为使用累积梯形积分
#     """
#     dt = 1 / sprate
#     if order == 1:  # 加速度->速度
#         return np.cumsum(data) * dt
#     elif order == 2:  # 加速度->位移
#         velocity = np.cumsum(data) * dt
#         return np.cumsum(velocity) * dt
#     else:
#         return data

#
# def srs_integration(data, sprate, distance):
#     """
#     频域积分函数 (简化实现)
#     """
#     dt = 1 / sprate
#     # 第一次积分 (加速度->速度)
#     velocity = np.cumsum(data) * dt
#     # 第二次积分 (速度->位移)
#     displacement = np.cumsum(velocity) * dt
#     return displacement


def respoff(data, fs, in_model):
    """
    去除地震仪器的响应特性
    :param data: 输入加速度信号 (m/s²)
    :param fs: 采样率 (Hz)
    :param in_model: 仪器模型类型 (1: EST, 2: 武汉, 3: 泰德TDE)
    :return: 校正后的加速度信号 (m/s²)
    """
    # 将输入数据转换为电压值 (V)
    Vdata = data / 100.0  # 转换为 cm/s²

    # 根据仪器模型设置参数
    if in_model == 1:  # EST 模型
        poles = np.array([-981 + 1009j, -981 - 1009j, -3290 + 1263j, -3290 - 1263j])
        K = 2.4597e13 * 1.0197  # 增益
        has_zeros = False
    elif in_model == 2:  # 武汉模型
        zeros_point = np.array([3.80233e3, 3.25142e2])
        poles = np.array([-2.29821e2 + 4.90498e2j, -2.29821e2 - 4.90498e2j,
                          -1.92328e3, 2.38227e2])
        K = -1.094939e5 * 2.551  # 增益
        has_zeros = True
    elif in_model == 3:  # 泰德TDE模型
        poles = np.array([-888 + 888j, -888 - 888j, -3020 + 1159j, -3020 - 1159j])
        K = 2.5 * 1.081e13  # 增益
        has_zeros = False
    else:  # 默认使用泰德TDE模型
        poles = np.array([-888 + 888j, -888 - 888j, -3020 + 1159j, -3020 - 1159j])
        K = 2.5 * 1.081e13  # 增益
        has_zeros = False

    n = len(Vdata)
    nfft = n  # FFT长度

    # 计算FFT
    Y = np.fft.fft(Vdata, nfft)

    # 创建频率轴 (rad/s)
    freq = np.fft.fftfreq(nfft, 1 / fs)
    w = 2 * np.pi * freq

    # 初始化传递函数数组
    H = np.zeros(nfft, dtype=complex)

    # 计算传递函数响应
    for k in range(nfft):
        s = 1j * w[k]  # 复频率 s = jω

        if has_zeros and in_model == 2:  # 武汉模型有零点
            numerator = K * (s - zeros_point[0]) * (s - zeros_point[1])
            denominator = (s - poles[0]) * (s - poles[1]) * (s - poles[2]) * (s - poles[3])
        else:  # 其他模型无零点
            numerator = K
            denominator = (s - poles[0]) * (s - poles[1]) * (s - poles[2]) * (s - poles[3])

        # 避免除以零
        if abs(denominator) > 1e-10:
            H[k] = numerator / denominator
        else:
            H[k] = 0.0

    # 频域反卷积（去除仪器响应）
    Y_corrected = Y / H

    # 时域重建（保持实数信号）
    V_reacc = np.real(np.fft.ifft(Y_corrected, nfft))

    # 转换为 gal (1 gal = 0.01 m/s²)
    # 10V/g * 980 gal/g = 9800 V/(m/s²)
    reacc = V_reacc * 980.0 / 10.0  # 转换为 m/s²

    return reacc