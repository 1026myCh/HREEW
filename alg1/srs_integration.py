import numpy as np
from scipy.signal import cont2discrete, lfilter, TransferFunction
# from control import tf as control_tf  # 需要安装 control 库
from scipy.signal import cont2discrete


def srs_integration(acc, fs, dist):
    """
    地震信号积分函数，模拟 DD-1 地震记录仪特性
    :param acc: 输入加速度信号 (单位 cm/s²)
    :param fs: 采样率 (Hz)
    :param dist: 震中距 (km)
    :return: 位移信号
    """
    # 直接使用原始加速度数据 (简化处理)
    acc_dd1 = acc.copy()
    
    # 系统参数设置
    T0 = 0.2  # 周期 (秒)
    zeta = 0.707  # 阻尼比
    
    # 时间步长
    dt = 1.0 / fs
    n = len(acc_dd1)
    disp = np.zeros(n)
    
    # 系统参数计算
    omega0 = 2 * np.pi / T0  # 自然圆频率 (rad/s)
    omegad = omega0 * np.sqrt(1 - zeta**2)  # 阻尼圆频率
    
    # 递归系数计算
    beta = zeta * omega0 * dt
    b1 = 2 * np.exp(-beta) * np.cos(omegad * dt)
    b2 = -np.exp(-2 * beta)
    S0 = (1 - b1 - b2) / (omega0 * dt)**2
    
    # δ值 (固定值)
    delta = 0.0913
    
    # 初始化前两步 (假设初始位移为0)
    disp[0] = 0.0
    disp[1] = 0.0
    
    # 递归计算位移
    for j in range(2, n):
        term = delta * acc_dd1[j] + (1 - 2 * delta) * acc_dd1[j-1] + delta * acc_dd1[j-2]
        disp[j] = b1 * disp[j-1] + b2 * disp[j-2] - S0 * dt**2 * term
    
    # 构建 DD-1 地震仪的传递函数
    # 拾震器部分: s^3/((s^2 + 5.655s + 39.48)(s + 4.545))
    # 记录笔部分: 15791/(s^2 + 177.7s + 15791)

    # # old
    # # 使用 control 库创建传递函数
    # s = control_tf('s')
    # H_analog = (s**3) / ((s**2 + 5.655*s + 39.48) * (s + 4.545)) * (15791) / (s**2 + 177.7*s + 15791)
    #
    # # 转换为离散传递函数 (双线性变换)
    # num = H_analog.num[0][0]  # 分子系数
    # den = H_analog.den[0][0]  # 分母系数
    #
    # dt = 1/fs
    # discrete_system1 = cont2discrete((num, den), dt, method='bilinear')
    # num_d1, den_d1 = discrete_system1[0], discrete_system1[1]
    # num_d1 = num_d1.reshape(-1)
    # disp_filtered1 = lfilter(num_d1, den_d1, disp)

    # new，#
    # 获取系数并离散化
    num_analog, den_analog = expand_transfer_function()
    fs = 1000  # 设置采样频率
    dt = 1 / fs
    discrete_system = cont2discrete((num_analog, den_analog), dt, method='bilinear')
    num_d, den_d = discrete_system[0], discrete_system[1]
    
    # 应用离散滤波器
    num_d = num_d.reshape(-1)
    # 结果比control 库传递函数结果小
    disp_filtered = lfilter(num_d, den_d, disp)
    
    return disp_filtered


# def linear_interp_to_double(arr):
#     """
#     将一维数组线性插值为两倍长度
#
#     参数:
#         arr: 输入的一维数组 (1×n)
#
#     返回:
#         插值后的数组 (1×2n)
#     """
#     n = len(arr)
#     if n == 0:
#         return np.array([])
#     if n == 1:
#         return np.array([arr[0], arr[0]])
#
#     # 原始索引
#     x_orig = np.arange(n)
#
#     # 目标索引 (步长减半)
#     x_target = np.linspace(0, n - 1, 2 * n)
#
#     # 线性插值
#     interp_arr = np.interp(x_target, x_orig, arr)
#
#     return interp_arr


def srs_integrationV(acc, fs):
    """
    根据加速度计算速度的积分算法

    参数:
        acc: 加速度数组 (单位 cm/s²)
        fs: 采样率 (Hz)

    返回:
        vv: 计算得到的速度数组
    """
    # 确保输入是numpy数组
    acc = np.array(acc)
    n = len(acc)  # 数据点数量

    # 初始化速度数组
    vv0 = np.zeros(n)

    # 如果输入数据太短，直接返回零数组
    if n < 3:
        return vv0

    # 固定参数
    T0 = 0.2
    zeta = 0.707
    delta = 0.0913  # δ值

    # 计算时间步长
    dt = 1.0 / fs

    # 系统参数计算
    omega0 = 2 * np.pi / T0  # 自然圆频率 (rad/s)
    omegad = omega0 * np.sqrt(1 - zeta ** 2)  # 阻尼圆频率

    # 递归系数计算
    beta = zeta * omega0 * dt
    b1 = 2 * np.exp(-beta) * np.cos(omegad * dt)
    b2 = -np.exp(-2 * beta)
    S0 = (1 - b1 - b2) / (omega0 * dt) ** 2

    # 对加速度进行插值处理
    # 1. 线性插值为2倍长度,1*n to 1*2n;
    # acc_interp = linear_interp_to_double(acc)
    n = len(acc)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([acc[0], acc[0]])
    # 原始索引
    x_orig = np.arange(n)
    # 目标索引 (步长减半)
    x_target = np.linspace(0, n - 1, 2 * n)
    # 线性插值
    acc_interp = np.interp(x_target, x_orig, acc)

    # 2. 在末尾添加一个点（复制最后一个点）
    acc1 = np.append(acc_interp, acc_interp[-1])

    # 递归计算位移
    for j in range(2, n):  # 从第三个点开始计算
        # 计算插值加速度的索引
        # 注意：MATLAB索引从1开始，Python从0开始
        idx1 = 2 * j + 1  # 对应MATLAB的2*j+1
        idx2 = 2 * j - 1  # 对应MATLAB的2*j-1
        idx3 = 2 * j - 3  # 对应MATLAB的2*j-3
        idx4 = 2 * j - 5  # 对应MATLAB的2*j-5

        # 确保索引有效
        if idx4 < 0 or idx1 >= len(acc1):
            continue

        # 计算公式中的项
        term = (delta * acc1[idx1] +
                (1 - 3 * delta) * acc1[idx2] -
                (1 - 3 * delta) * acc1[idx3] -
                delta * acc1[idx4])

        # 递归计算
        vv0[j] = b1 * vv0[j - 1] + b2 * vv0[j - 2] - S0 * dt * term

        # # 使用 control 库创建传递函数
        # s = control_tf('s')
        # H_analog = (s ** 3) / ((s ** 2 + 5.655 * s + 39.48) * (s + 4.545)) * (15791) / (s ** 2 + 177.7 * s + 15791)
        #
        # # 转换为离散传递函数 (双线性变换)
        # num = H_analog.num[0][0]  # 分子系数
        # den = H_analog.den[0][0]  # 分母系数
        #
        # # 使用 scipy 的 cont2discrete 进行转换
        # dt = 1/fs
        # discrete_system = cont2discrete((num, den), dt, method='bilinear')
        # num_d, den_d = discrete_system[0], discrete_system[1]
        #
        # # 应用离散滤波器
        # num_d = num_d.reshape(-1)
        # vv0 = lfilter(num_d, den_d, vv0)

    return vv0


# 直接展开多项式计算系数
def expand_transfer_function():
    # 分子: 15791 * s^3
    num = [15791, 0, 0, 0]  # 15791 * s^3

    # 分母: (s^2 + 5.655s + 39.48) * (s + 4.545) * (s^2 + 177.7s + 15791)
    # 展开多项式乘积
    den1 = [1, 5.655, 39.48]  # s^2 + 5.655s + 39.48
    den2 = [1, 4.545]  # s + 4.545
    den3 = [1, 177.7, 15791]  # s^2 + 177.7s + 15791

    # 逐步计算多项式乘积
    temp = np.polymul(den1, den2)
    den = np.polymul(temp, den3)

    return num, den


