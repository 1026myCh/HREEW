import numpy as np
from scipy.signal import lfilter

def iomega(datain, sprate, xx, Debug, flag):
    """
    在频域进行积分或微分操作
    :param datain: 输入信号 (加速度数据)
    :param sprate: 采样率 (Hz)
    :param xx: 操作类型 (1: 积分, 2: 二次积分)
    :param Debug: 调试标志 (未使用)
    :param flag: 执行标志 (1: 跳过执行, 0: 正常执行)
    :return: 处理后的信号 (速度或位移)
    """
    # 如果 flag=1，直接返回 -1
    if flag == 1:
        return -1
    
    # 获取输入数据的形状
    if len(datain.shape) == 1:
        len_datain = datain.shape[0]
        row_datain = 1
        datain = datain.reshape(-1, 1)  # 转为列向量
    else:
        len_datain, row_datain = datain.shape
    
    # 检查输入数据是否为空
    if len_datain <= 0:
        print('iomega.m: 输入数据为空')
        return -1
    
    # 计算时间步长和FFT参数
    dt = 1.0 / sprate
    N = 2 ** int(np.ceil(np.log2(len_datain)))  # 下一个2的幂
    df = 1.0 / (N * dt)                       # 频率分辨率
    Nyq = 1.0 / (2 * dt)                      # 奈奎斯特频率
    
    # 创建频率数组 (从 -Nyq 到 Nyq-df)
    iomega_array = 1j * 2 * np.pi * np.arange(-Nyq, Nyq, df)
    iomega_exp = -xx  # 积分操作的指数
    
    # 零填充输入数据
    if len_datain > row_datain:  # 列向量
        datain_padded = np.vstack([datain, np.zeros((N - len_datain, row_datain))])
    else:  # 行向量
        datain_padded = np.hstack([datain, np.zeros((len_datain, N - row_datain))])
    
    # 预定义滤波器系数 (带通滤波器 0.1-25Hz)
    x = np.array([
        0.00891445723946303, 0, -0.0356578289578521, 0, 0.0534867434367782, 
        0, -0.0356578289578521, 0, 0.00891445723946303
    ])
    
    y = np.array([
        1, -5.97019036284279, 15.6760061739522, -23.7534053193241, 
        22.8038401706724, -14.2293982265302, 5.63370270828677, 
        -1.29223569702625, 0.131680715038691
    ])
    
    # 第一次滤波
    datain_filtered = lfilter(x, y, datain_padded, axis=0)
    
    # 执行FFT
    A = np.fft.fft(datain_filtered, n=N, axis=0)
    
    # FFT移位
    A_shifted = np.fft.fftshift(A, axes=0)
    
    # 频域操作 (积分/微分)
    for j in range(N):
        if iomega_array[j] != 0:
            A_shifted[j] = A_shifted[j] * (iomega_array[j] ** iomega_exp)
        else:
            A_shifted[j] = 0.0 + 0.0j
    
    # 逆FFT移位
    A_ishifted = np.fft.ifftshift(A_shifted, axes=0)
    
    # 执行逆FFT
    dataout = np.fft.ifft(A_ishifted, axis=0)
    dataout = np.real(dataout)
    
    # 裁剪到原始长度
    if len_datain > row_datain:
        dataout = dataout[:len_datain, :]
    else:
        dataout = dataout[:, :row_datain]
    
    # 第二次滤波
    dataout = lfilter(x, y, dataout, axis=0)
    
    return dataout.squeeze()  # 移除单维度