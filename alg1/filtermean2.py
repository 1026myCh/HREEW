import numpy as np

def filtermean2(data):
    """
    中值滤波 平滑处理。20220921
    in:
     data: 多列数据，列序：传感器1UDdata、传感器2UDdata……
     flag: 是否空运行，1：空运行. 0:正常运行
    out:
     result: 滤波后的数据，与输入维度相同
    """
    nlines, nrows = data.shape
    n = nrows // 10
    origal_data = data.copy()  # 原始数据
    data = data[:n*10, :]  # 长度取整了，10的倍数
    result = data.copy()


    # 如果单次输入太长了，不滤波了
    if nlines >= 3001:
        return result

    # 滤波
    Lpackage = int(nrows/2)  # 10/20
    step = 1

    for i in range(Lpackage, nrows, step):
        data_peace1 = data[0][i-Lpackage:i]
        data_peace2 = data[1][i-Lpackage:i]
        data_peace3 = data[2][i-Lpackage:i]  # sensor1ud
        data_peace4 = data[3][i-Lpackage:i]
        data_peace5 = data[4][i-Lpackage:i]
        data_peace6 = data[5][i-Lpackage:i]  # sensor2ud

        n = len(data_peace3)
        temp_data1 = np.abs(data_peace1)
        temp_data2 = np.abs(data_peace2)
        temp_data3 = np.abs(data_peace3)
        temp_data4 = np.abs(data_peace4)
        temp_data5 = np.abs(data_peace5)
        temp_data6 = np.abs(data_peace6)

        ind_max1 = np.argmax(temp_data1)
        ind_max2 = np.argmax(temp_data2)
        ind_max3 = np.argmax(temp_data3)
        ind_max4 = np.argmax(temp_data4)
        ind_max5 = np.argmax(temp_data5)
        ind_max6 = np.argmax(temp_data6)

        data_peace1[ind_max1] = 0
        data_peace2[ind_max2] = 0
        data_peace3[ind_max3] = 0
        data_peace4[ind_max4] = 0
        data_peace5[ind_max5] = 0
        data_peace6[ind_max6] = 0

        ind_max3 = np.argmax(temp_data3)
        ind_max6 = np.argmax(temp_data6)
        data_peace3[ind_max3] = 0
        data_peace6[ind_max6] = 0


        sum_value1 = np.sum(data_peace1)
        sum_value2 = np.sum(data_peace2)
        sum_value3 = np.sum(data_peace3)
        sum_value4 = np.sum(data_peace4)
        sum_value5 = np.sum(data_peace5)
        sum_value6 = np.sum(data_peace6)
    
        r1 = sum_value1 / (n - 1)
        r2 = sum_value2 / (n - 1)
        r3 = sum_value3 / (n - 2)
        r4 = sum_value4 / (n - 1)
        r5 = sum_value5 / (n - 1)
        r6 = sum_value6 / (n - 2)

        result[0][i] = r1
        result[1][i] = r2
        result[2][i] = r3
        result[3][i] = r4
        result[4][i] = r5
        result[5][i] = r6

    return result
