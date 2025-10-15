# -*-coding:UTF-8 -*-
"""Wrapper for Nature mode that keeps computing when Judges returns non-earthquake."""

from Nature_Single import Nature_Single


def Nature_Single1(Data_now, StartT, MaxEEW_times, StationInfo, NewInfo, Debug, Sprate, ThreshGals,
                   S_time2, Buffer_len, EEW_Time_After_S, Pspeed, Sspeed, Alarm, Sta_vars1,
                   Sta_vars2, Alarm1, Alarm2, flagarea, in_model, Gain):
    """Invoke :func:`Nature_Single` while keeping processing for non-earthquake cases.

    This mode is used for testing scenarios where Judges returns ``result = 0``.
    """

    return Nature_Single(Data_now, StartT, MaxEEW_times, StationInfo, NewInfo, Debug, Sprate,
                         ThreshGals, S_time2, Buffer_len, EEW_Time_After_S, Pspeed, Sspeed,
                         Alarm, Sta_vars1, Sta_vars2, Alarm1, Alarm2, flagarea, in_model,
                         Gain, continue_on_non_eqk=True)

