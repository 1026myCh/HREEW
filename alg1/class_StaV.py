import numpy as np
from class_obj import AlarmS
class Sta_V():
    def __init__(self):
        self.meanhead = []
        self.LastAlarmTime = -1000
        self.LastEEWTime = None
        self.STA_Long = -1.0
        self.STA_Lat = -1.0
        self.Epi_Long = -1.0
        self.Epi_Lat = -1.0
        self.sprate = -1
        self.StarT = -1
        self.Magnitude = -1
        self.Magnitude_real = -1
        self.STA_Name = "xxx"
        self.Triged = 0
        self.NewInfo = -1
        self.Duration = -1.0
        self.P_time = -1
        self.S_time = -1
        self.End_time = -1.0
        self.PGA_Real = -1.0
        self.Distance_real = -1.0
        self.Distance_Pred = -1.0
        self.Azimuth_real = -1.0
        self.Epi_Depth = -1
        self.Epi_Depth_real = -1
        self.PGA_Pred = -1.0
        self.Trig_first = -1
        self.M = -1
        self.S = -1
        self.Buffer = np.empty([3, 0],
                               dtype=float)
        self.S_Theory_time = -1
        self.theta = np.empty(shape=(3, 0))
        self.Package = np.empty(shape=(3, 0))
        self.Traces_evt = np.empty(shape=(3, 0))
        self.B = -1.0
        self.A = -1.0
        self.Delta = np.empty(shape=(3, 0))
        self.tc = np.empty(shape=(3, 0))
        self.pd = np.empty(shape=(3, 0))
        self.PGV_Curr = 0.0
        self.PGD_Curr = 0.0
        self.DurCurr = 0.0
        self.LTA_evt = np.empty((1, 0),
                                dtype=float)
        self.filter_Buffer = np.empty([1, 0],
                                      dtype=float)
        self.Epi_time = -1.0
        self.Epi_time_real = -1.0
        self.epiLon_real = -1.0
        self.epiLat_real = -1.0
        self.M = 0
        self.S = 0
        self.Is_EQK = 0
        self.Azimuth = -10000.0
        self.BaseLine = np.empty(shape=(3, 0))
        self.EEW_times = 25 #
        self.AlarmLevel = 0
        self.PGA_Curr = 0
        self.Buffer_len = 0
        self.S_time_cal=-1
        self.triger_fine=-1
        self.PGAEEW_times=0
        self.AlarmFlag = 0
        self.S_timeold = 0
        # 20250819
        self.AZIold = 0
        self.distold = -1
        self.PGAcurrold = -1
        self.PGAold = -1  # Alarm匹配
        # 20251011
        self.corrENZ = np.array([0.0, 0.0, 0.0], dtype=float)
        self.Ef_UD = np.empty([11, 0], dtype=float)
        self.SNout = np.empty([3, 0], dtype=float)
        # 20251013 add
        self.flagarea = 3
        # # :P波时竖直位移峰值,值填入PGD_Curr
        # self.P_ud_m = 0
        # :事件数据长度(点数)
        self.lenCh6 = 0
        # :TaoC值均值
        self.TC = 0
        # :S波时水平向位移峰值(双传感器平均)
        self.sAvgH_m = 0




    def ini2(self, MaxEEW_times,FirstStart, object=None):

        # self.End_time = -1  # 不可在此初始化结束时间
        self.Triged = 0
        self.NewInfo = -1
        self.Duration = 0
        self.M = 0
        self.S = 0
        self.Azimuth = -1000.0
        self.P_time = -1
        self.S_time = -1
        self.Distance_Pred = -1.0
        self.Magnitude = -1
        self.Epi_Long = -1.0
        self.Epi_Lat = -1.0
        self.S_time_cal = -1
        self.Epi_time = -1.0
        self.Is_EQK = 0
        self.theta = []
        self.AlarmLevel = 0
        self.EEW_times = MaxEEW_times
        self.Duration = -1.0
        self.M = 0
        self.S = 0
        self.LastAlarmTime = -1000
        self.PGA_Curr = 0.0
        self.PGD_Curr = 0.0
        self.PGV_Curr = 0.0
        self.Traces_evt = np.empty(shape=(3, 0))
        self.meanhead = []
        self.PGA_Pred=-1
        self.PGAEEW_times = 0
        self.AlarmFlag = 0
        if FirstStart==1:
            self.End_time = -1;
        # 20250819
        self.AZIold = 0
        self.distold = -1
        self.PGAcurrold = -1
        self.PGAold = -1  # Alarm匹配
        self.corrENZ = np.array([0.0, 0.0, 0.0], dtype=float)
        self.Ef_UD = np.empty([11, 0], dtype=float)
        self.SNout = np.empty([3, 0], dtype=float)
        # 20251013 add
        self.flagarea = 3
        # # :P波时竖直位移峰值
        # self.P_ud_m = 0
        # :事件数据长度(点数)
        self.lenCh6 = 0
        # :TaoC值均值
        self.TC = 0
        # :S波时水平向位移峰值(双传感器平均)
        self.sAvgH_m = 0

        if object is not None and isinstance(object,AlarmS):
            object.AlarmLevel = -1
            object.PGA = -1
            object.Alarmtime = -1
            object.recordtime = -1
            object.delT = -1



# class global_Sta_V(Sta_V):
# # Sta_vars1.STA_Name=StCode
# # Sta_vars1.STA_Long=Lon
# # Sta_vars1.STA_Lat=Lat
# # Sta_vars1.sprate=sprate
# # Sta_vars1.PGA_Real=PGA_Real1
# # Sta_vars1.Distance_real=epi_dist
# # Sta_vars1.Magnitude_real=magnitude
# # Sta_vars1.Epi_time_real=epi_time
# # Sta_vars1.Aziumth_real=Azimuth_real
# # Sta_vars1.epiLat_real=epiLat
# # # Sta_vars1.epiLon_real=epiLon
#      global StCode
# #     global STA_Lon
# #     global STA_Lat
# # global sprate
# # global PGA_Real
#      StCode=Sta_V.STA_Name
# # STA_Lon=object.STA_Long
# class gobal_Value():
class StationInfosStruct():
    def __init__(self):
        self.STA_Long = -1
        self.STA_Lat = -1
        self.Epi_Long = -1
        self.Epi_Lat = -1
        self.sprate = -1
        self.StartT = -1
        self.Magnitude = -1
        self.STA_Name = "xxx"
        self.NewInfo = -1
        self.Duration = -1
        self.P_time = -1
        self.S_time = -1
        self.End_time = -1
        self.Distance = -1
        self.Azimuth = -1
        self.Epi_Depth = -1
        self.PGA_Pred = -1
        self.Trig_first = -1
        self.PGA_Curr= -1
        self.PGV_Curr = 0
        self.PGD_Curr = 0
        self.DurCurr = 0
        self.Epi_time = -1
        self.S_time_cal=-1