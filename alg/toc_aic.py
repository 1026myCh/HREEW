import numpy as np
import math
import math

import numpy as np


# Data :单方向数据，一列
# TrigS ；S波在当前数据中的点位
def  TOC_AIC(Data):
    TrigS = -1
    delt_AIC = -1
    N = len(Data)-1
    AIC2 = np.zeros((N),dtype=float)
    Data_m2 = Data**2
    Data_m1 = Data
    total_m1 = sum(Data_m1[0:N+2]) # AIC段全部数据和
    total_m2 = sum(Data_m2[0:N+2]) # AIC段全部数据平方和
    temp_m1 = Data_m1[0]
    temp_m2 = Data_m2[0]
    for k in range(N-2):#N-2？
        K=k+1
        temp_m1 = temp_m1+Data_m1[K] #和
        temp_m2 = temp_m2+Data_m2[K] #平方和
        try:
            a=(total_m2-temp_m2)/(N-k)-(total_m1-temp_m1)**2/(N-k)**2
            AIC2[k] = (K+1)*math.log(temp_m2/(K+1)-(temp_m1/(K+1))**2,10)+(N-k)*math.log((total_m2-temp_m2)/(N-k)-(total_m1-temp_m1)**2/(N-k)**2,10)
        except:
            # plt.plot(Data)
            continue

        AIC_min = np.min(AIC2[2:-2])
        TrigS=np.argmin(AIC2[2:-2])

        #ma是最小值，Ka是取得最小值时的节点
    # delt_AIC = AIC2(3)-AIC_min
        if  AIC2[2]-AIC_min < 500:
            TrigS=-1
        return TrigS, delt_AIC
