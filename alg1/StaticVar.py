class StaticVar():
    StartT=-1
    StartT1=-1
    position=-1
    FirstStart=-1
    First3s=-1
    First3s2=-1
    first_fig=-1
    First_filter=-1
    flag_nature=-1 #原Nature
    PGAold=-1
    AZIold=-1
    distold=-1
    AlarmLevel=-1
    beMagnitude=-1
    retout=-1
    Flag=None
    tempret1=None
    tempret2=None
    retout=None
    dist_time=-1.0
    P_time_Pre=-1.0
    S_time_Pre=-1.0
    end_time_Pre=-1.0
    PGAoldout=0.00  #仅输出报次记录用

    # P_time_Pre=[-1.0,-1.0] #传感器1、2 Triger
    # S_time_Pre = [-1.0,-1.0] #传感器1、2 Triger
    # end_time_Pre=[-1.0,-1.0] #传感器1、2 Triger
    N_pnsr1=[-1.0,-1.0,-1.0] #Triger
    N_pnsr2=[-1.0,-1.0,-1.0] #Triger

class Static_EEW_Params():
    VersionNo=-1
    Debug=-1
    Pspeed=-1
    Sspeed=-1
    ThreshGals=[40,80,120]
    Buffer_seconds = -1
    EEW_Refresh_Second = -1
    Alarm_Refresh_Second = -1
    MaxEEW_times = -1
    EEW_Time_After_S = -1
    Second2judge=1.5
    Epi_Depth_rand=[10,5]
    S_Pick_Thresh=10
    Mag_converge=[6.5,3]
    MinDuration=15
    LongestNonEqk=15
    MaxDur=150
    CodaUper=3.5
    ForceEndThresh=0.05
    EndStaLta=1.2
    EndRatio=15
    pins=10
    ratio=0.8
    Back=10
    Fowrd=1
    thresh=32# STA/LTA thresh
    STW= 0.4#seconds of short time windoweeX
    LTW= 7#7seconds of long  time window
    iflilter=1
    MinThresh=0.5#触发的最小加速度值0.5gal，避免台阶触发
    spanDeOddSecond=0.2
    L_fs=0.1
    H_fs=10
    IfCkCorr=1
    MinCorrEW=0.99
    MinCorrNS=0.99
    MinCorrUD=0.99
    MinCorrEW_Nature=0.9
    MinCorrNS_Nature=0.9
    MinCorrUD_Nature=0.95
    FreqRange=[0.05,35]
    MaxCorrInSens=0.95
    MaxHorientalFirstSecond=100
    MaxPredSum=120
    MaxPredSingle=45
    SteadyRatioMax=0.6
    MaxHVrationFirstSecond=3
    Mag_M=[3.373,5.787]
    Sprate=200
    flagarea=3
    in_type=1
    Gain=0
    NatureMode="0"

    #需添加Nature 静态子类
    @staticmethod
    def Var_Copy(self,object):
        #setattr 转换出来为字符串
        setattr(self,"VersionNo",object["EEW_Single"]["VersionNo"])

        setattr(self,"Debug",object["EEW_Single"]["Debug"])
        setattr(self,"Sprate",object["EEW_Single"]["Sprate"])
        setattr(self,"Pspeed",object["EEW_Single"]["Pspeed"])
        setattr(self,"Sspeed",object["EEW_Single"]["Sspeed"])
        setattr(self,"ThreshGals",object["EEW_Single"]["ThreshGals"])
        setattr(self,"Buffer_seconds",object["EEW_Single"]["Buffer_seconds"])
        setattr(self,"EEW_Refresh_Second",object["EEW_Single"]["EEW_Refresh_Second"])
        setattr(self,"MaxEEW_times",object["EEW_Single"]["MaxEEW_times"])
        setattr(self,"EEW_Time_After_S",object["EEW_Single"]["EEW_Time_After_S"])
        setattr(self,"Second2judge",object["EEW_Single"]["Second2judge"])
        setattr(self,"Epi_Depth_rand",object["EEW_Single"]["Epi_Depth_rand"])
        setattr(self,"S_Pick_Thresh",object["EEW_Single"]["S_Pick_Thresh"])
        setattr(self,"Mag_converge",object["EEW_Single"]["Mag_converge"])

        setattr(self, "in_type", object["EEW_Single"]["in_model"])
        setattr(self, "FlagArea", object["EEW_Single"]["FlagArea"])
        setattr(self, "Gain", object["EEW_Single"]["Gain"])
        setattr(self, "NatureMode", object["EEW_Single"].get("NatureMode", "0"))

        setattr(self,"MinDuration",object["EEW_Triger"]["MinDuration"])
        setattr(self,"LongestNonEqk",object["EEW_Triger"]["LongestNonEqk"])
        setattr(self,"MaxDur",object["MyEnder"]["MaxDur"])
        setattr(self,"CodaUper",object["MyEnder"]["CodaUper"])
        setattr(self,"EndStaLta",object["MyEnder"]["EndStaLta"])
        setattr(self,"EndRatio",object["MyEnder"]["EndRatio"])
        setattr(self,"CodaUper",object["MyEnder"]["CodaUper"])
        setattr(self,"EndStaLta",object["MyEnder"]["EndStaLta"])
        setattr(self,"EndRatio",object["MyEnder"]["EndRatio"])
        setattr(self,"CodaUper",object["MyEnder"]["CodaUper"])
        setattr(self,"EndStaLta",object["MyEnder"]["EndStaLta"])
        setattr(self,"EndRatio",object["MyEnder"]["EndRatio"])
        setattr(self,"Back",object["MyPickerAIC"]["Back"])
        setattr(self,"Fowrd",object["MyPickerAIC"]["Fowrd"])
        setattr(self,"thresh",object["MyPickerAIC"]["thresh"])
        setattr(self,"STW",object["MyPickerAIC"]["STW"])
        setattr(self,"LTW",object["MyPickerAIC"]["LTW"])
        setattr(self,"iflilter",object["MyPickerAIC"]["iflilter"])
        setattr(self,"MinThresh",object["MyPickerAIC"]["MinThresh"])
        setattr(self,"spanDeOddSecond",object["MyPicker"]["spanDeOddSecond"])
        setattr(self,"L_fs",object["MyPicker"]["L_fs"])
        setattr(self,"H_fs",object["MyPicker"]["H_fs"])
        setattr(self,"IfCkCorr",object["Judges"]["IfCkCorr"])
        setattr(self,"MinCorrEW",object["Judges"]["MinCorrEW"])
        setattr(self,"MinCorrNS",object["Judges"]["MinCorrNS"])
        setattr(self,"MinCorrUD",object["Judges"]["MinCorrUD"])
        setattr(self,"FreqRange",object["Judges"]["FreqRange"])
        setattr(self,"MaxCorrInSens",object["Judges"]["MaxCorrInSens"])
        setattr(self,"MaxHorientalFirstSecond",object["Judges"]["MaxHorientalFirstSecond"])
        setattr(self,"MaxPredSum",object["Judges"]["MaxPredSum"])
        setattr(self,"MaxPredSingle",object["Judges"]["MaxPredSingle"])
        setattr(self,"SteadyRatioMax",object["Judges"]["SteadyRatioMax"])
        setattr(self,"MaxHVrationFirstSecond",object["Judges"]["MaxHVrationFirstSecond"])
        setattr(self,"Mag_M",object["Judges"]["Mag_M"])




