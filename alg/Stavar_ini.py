
def Stavar_ini(Sta_vars1,Alarm,MaxEEW_times,FirstStart):
    Sta_vars_out=Sta_vars1
    Alarmout=Alarm

    Sta_vars_out.Triged = -1
    Sta_vars_out.NewInfo = -1
    Sta_vars_out.Duration = -1 #从P到end的时间
    Sta_vars_out.DurCurr = -1 # ?
    Sta_vars_out.M = -1
    Sta_vars_out.S = -1
    Sta_vars_out.Azimuth = -1000 # 一定要小于-1000，不然会有冲突。
    ## 负数角度0~-360度表示建设的结果很好。
    Sta_vars_out.P_time = -1
    Sta_vars_out.S_time = -1 ## ？？？
    Sta_vars_out.S_timeold = -1
    Sta_vars_out.Distance_Pred = -1
    Sta_vars_out.Magnitude = -1
    Sta_vars_out.Epi_Long = -1
    Sta_vars_out.Epi_Lat = -1
    Sta_vars_out.S_time_cal = -1
    Sta_vars_out.Epi_time = -1
    Sta_vars_out.N_pnsr = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3] #分母

    Sta_vars_out.Is_EQK = -1 # 初值：-1，触发：0，地震：1。
    Sta_vars_out.theta = -1
    Sta_vars_out.EEW_times = MaxEEW_times ##递减，最多发送25报。


    #首次/强制启动时全部初始化
    if FirstStart == 1:
        ###20230303, 改为不更新
        # Sta_vars_out.BaseLine = [] #20200409改为存储原始数据
        # Sta_vars_out.Package = []
        # Sta_vars_out.Buffer = []
        # Sta_vars_out.StartT = -1
        # Sta_vars_out.rout = []
        Sta_vars_out.End_time = -1 #在首次启动，才更新endtime为 - 1.


    Sta_vars_out.LastEEWTime = -1 #更新上次返回时间
    Sta_vars_out.LastAlarmTime = -1
    Sta_vars_out.PGA_Curr = -1
    Sta_vars_out.PGD_Curr = -1
    Sta_vars_out.PGV_Curr = -1
    Sta_vars_out.Traces_evt = [] #Sta_vars.Traces_evt是原始数据未去基线哦
    Sta_vars_out.buffer_Traces_evt = [] #20230215, 新增
    Sta_vars_out.LTA_evt = [] #20210607新增
    Sta_vars_out.meanhead = [] #20200630新增
    # Sta_vars_out.identifyCnt = [] #
    # Sta_vars_out.BaseLine = []
    Sta_vars_out.AlarmLevel = -1
    Sta_vars_out.Alarmflag = 0 #本次事件中是否发生过阈值报警，1：有，0：无。

    ##20230816 new add
    Sta_vars_out.SNout = [0, 0, 0] #仅赋值一次不再改变，结束时初始化
    Sta_vars_out.corrENZ = [0, 0, 0] #3
    Sta_vars_out.Ef_UD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #11

    ##20230913 add
    Sta_vars_out.Sbegin = -1

    ##20230928
    Sta_vars_out.probsum = 0
    Sta_vars_out.probty_sta = 0
    Sta_vars_out.probty_aic = 0

    Sta_vars_out.Distance_ori = -1
    Sta_vars_out.End_time0 = -1

    Alarm.ErrInfo = -1
    Alarm.AlarmLevel = -1 # Alarm = struct('ErrInfo', '-1', 'AlarmLevel', -1, 'PGA', -1)
    Alarm.PGA = -1
    return Alarm,Sta_vars_out
