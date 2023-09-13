import pandas as pd
import talib


def get_indicators(time_series):
    df = pd.DataFrame()
    # Overlap Studies Functions

    upperband, middleband, lowerband = talib.BBANDS(time_series, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df["bbands_upperband"] = upperband
    df["bbands_middleband"] = middleband
    df["bbands_lowerband"] = lowerband

    # real = talib.DEMA(time_series, timeperiod=30)
    # df["DEMA"] = real
    # real = talib.EMA(time_series, timeperiod=30)
    # df["EMA"] = real
    # real = talib.HT_TRENDLINE(time_series)
    # df["HT_TRENDLINE"] = real
    # # real = talib.KAMA(time_series, timeperiod=30)
    # # df["KAMA"] = real
    # real = talib.MA(time_series, timeperiod=30, matype=0)
    # df["MA"] = real
    # real = talib.MIDPOINT(time_series, timeperiod=14)
    # df["MIDPOINT"] = real
    # real = talib.SMA(time_series, timeperiod=30)
    # df["SMA"] = real
    # # real = talib.T3(time_series, timeperiod=5, vfactor=0)
    # # df["T3"] = real
    # real = talib.TEMA(time_series, timeperiod=30)
    # df["TEMA"] = real
    # real = talib.TRIMA(time_series, timeperiod=30)
    # df["TRIMA"] = real
    # real = talib.WMA(time_series, timeperiod=30)
    # df["WMA"] = real

    #mama, fama = MAMA(close, fastlimit=0, slowlimit=0)
    #real = talib.MAVP(time_series, periods=2, minperiod=2, maxperiod=30, matype=0)

    # Momentum Indicator Functions

    # real = talib.APO(time_series, fastperiod=12, slowperiod=26, matype=0)
    # df["APO"] = real
    # real = talib.CMO(time_series, timeperiod=14)
    # df["CMO"] = real
    # macd, macdsignal, macdhist = talib.MACD(time_series, fastperiod=12, slowperiod=26, signalperiod=9)
    # df["MACD_macd"] = macd
    # df["MACD_macdsignal"] = macdsignal
    # df["MACD_macdhist"] = macdhist
    # macd, macdsignal, macdhist = talib.MACDEXT(time_series, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    # df["MACDEXT_macd"] = macd
    # df["MACDEXT_macdsignal"] = macdsignal
    # df["MACDEXT_macdhist"] = macdhist
    # macd, macdsignal, macdhist = talib.MACDFIX(time_series, signalperiod=9)
    # df["MACDFIX_macd"] = macd
    # df["MACDFIX_macdsignal"] = macdsignal
    # df["MACDFIX_macdhist"] = macdhist
    # real = talib.MOM(time_series, timeperiod=10)
    # df["MOM"] = real
    # real = talib.PPO(time_series, fastperiod=12, slowperiod=26, matype=0)
    # df["PPO"] = real
    # real = talib.ROC(time_series, timeperiod=10)
    # df["ROC"] = real
    # real = talib.ROCP(time_series, timeperiod=10)
    # df["ROCP"] = real
    # real = talib.ROCR(time_series, timeperiod=10)
    # df["ROCR"] = real
    real = talib.ROCR100(time_series, timeperiod=10)
    df["ROCR100"] = real
    real = talib.RSI(time_series, timeperiod=14)
    df["RSI"] = real
    # fastk, fastd = talib.STOCHRSI(time_series, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    # df["STOCHRSI_fastk"] = fastk
    # df["STOCHRSI_fastd"] = fastd
    # real = talib.TRIX(time_series, timeperiod=30)
    # df["TRIX"] = real

    # Cycle Indicator

    # real = talib.HT_DCPERIOD(time_series)
    # df["HT_DCPERIOD"] = real
    # real = talib.HT_DCPHASE(time_series)
    # df["HT_DCPHASE"] = real
    # inphase, quadrature = talib.HT_PHASOR(time_series)
    # df["HT_PHASOR_inphase"] = inphase
    # df["HT_PHASOR_quadrature"] = quadrature
    # sine, leadsine = talib.HT_SINE(time_series)
    # df["HT_SINE_sine"] = sine
    # df["HT_SINE_leadsine"] = leadsine
    # integer = talib.HT_TRENDMODE(time_series)
    # df["HT_TRENDMODE"] = integer

    # Statistic Functions

    # real = talib.LINEARREG(time_series, timeperiod=14)
    # df["LINEARREG"] = real
    # real = talib.LINEARREG_ANGLE(time_series, timeperiod=14)
    # df["LINEARREG_ANGLE"] = real
    # real = talib.LINEARREG_INTERCEPT(time_series, timeperiod=14)
    # df["LINEARREG_INTERCEPT"] = real
    # real = talib.LINEARREG_SLOPE(time_series, timeperiod=14)
    # df["LINEARREG_SLOPE"] = real
    # real = talib.STDDEV(time_series, timeperiod=5, nbdev=1)
    # df["STDDEV"] = real
    # real = talib.TSF(time_series, timeperiod=14)
    # df["TSF"] = real
    # real = talib.VAR(time_series, timeperiod=5, nbdev=1)
    # df["VAR"] = real
    # df = pd.DataFrame()
    return df

    #     {
    #     # Overlap Studies Functions
    #     ('bbands_upperband', 'bbands_middleband', 'bbands_lowerband'):
    #         talib.BBANDS(time_series, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0),
    #     'DEMA': talib.DEMA(time_series, timeperiod=30),
    #     'EMA': talib.EMA(time_series, timeperiod=30),
    #     'HT_TRENDLINE': talib.HT_TRENDLINE(time_series),
    #     'KAMA': talib.KAMA(time_series, timeperiod=30),
    #     'MA': talib.MA(time_series, timeperiod=30, matype=0),
    #     'MIDPOINT': talib.MIDPOINT(time_series, timeperiod=14),
    #     'T3': talib.T3(time_series, timeperiod=5, vfactor=0),
    #     'TRIMA': talib.TRIMA(time_series, timeperiod=30),
    #     'WMA': talib.WMA(time_series, timeperiod=30),
    #
    #     # mama, fama = MAMA(close, fastlimit=0, slowlimit=0)
    #     # real = talib.MAVP(time_series, periods=2, minperiod=2, maxperiod=30, matype=0)
    #
    #     # Momentum Indicator Functions
    #     'APO': talib.APO(time_series, fastperiod=12, slowperiod=26, matype=0),
    #     'CMO': talib.CMO(time_series, timeperiod=14),
    #     ('MACD_macd', 'MACD_macdsignal', 'MACD_macdhist'):
    #         talib.MACD(time_series, fastperiod=12, slowperiod=26, signalperiod=9),
    #     ('MACDEXT_macd', 'MACDEXT_macdsignal', 'MACDEXT_macdhist'):
    #         talib.MACDEXT(time_series, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0),
    #     ('MACDFIX_macd', 'MACDFIX_macdsignal', 'MACDFIX_macdhist'):
    #         talib.MACDFIX(time_series, signalperiod=9),
    #     'MOM': talib.MOM(time_series, timeperiod=10),
    #     'PPO': talib.PPO(time_series, fastperiod=12, slowperiod=26, matype=0),
    #     'ROC': talib.ROC(time_series, timeperiod=10),
    #     'ROCP': talib.ROCP(time_series, timeperiod=10),
    #     'ROCR': talib.ROCR(time_series, timeperiod=10),
    #     'ROCR100': talib.ROCR100(time_series, timeperiod=10),
    #     'RSI': talib.RSI(time_series, timeperiod=14),
    #     ('STOCHRSI_fastk', 'STOCHRSI_fastd'):
    #         talib.STOCHRSI(time_series, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0),
    #     'TRIX': talib.TRIX(time_series, timeperiod=30),
    #
    #     # Cycle Indicators
    #     'HT_DCPERIOD': talib.HT_DCPERIOD(time_series),
    #     'HT_DCPHASE': talib.HT_DCPHASE(time_series),
    #     ('HT_PHASOR_inphase', 'HT_PHASOR_quadrature'): talib.HT_PHASOR(time_series),
    #     ('HT_SINE_sine', 'HT_SINE_leadsine'): talib.HT_SINE(time_series),
    #     'HT_TRENDMODE': talib.HT_TRENDMODE(time_series),
    #
    #     # Statistic Functions
    #     'LINEARREG': talib.LINEARREG(time_series, timeperiod=14),
    #     'LINEARREG_ANGLE': talib.LINEARREG_ANGLE(time_series, timeperiod=14),
    #     'LINEARREG_INTERCEPT': talib.LINEARREG_INTERCEPT(time_series, timeperiod=14),
    #     'LINEARREG_SLOPE': talib.LINEARREG_SLOPE(time_series, timeperiod=14),
    #     'STDDEV': talib.STDDEV(time_series, timeperiod=5, nbdev=1),
    #     'TSF': talib.TSF(time_series, timeperiod=14),
    #     'VAR': talib.VAR(time_series, timeperiod=5, nbdev=1),
    # }


# def get_max_time_period_from_indicators(empty_time_series):
#     max_time_period = 0
#     time_periods = ('timeperiod', 'slowperiod')
#     for indicator in get_indicators(empty_time_series).items():
#         # print(f'{getattr(indicator,tp)=}')
#         for tp in time_periods:
#             print(f'{getattr(indicator,tp)=}')
#             # print(f'{indicator=} {tp=}')
#             # if tp in indicator.__dict__.keys() and indicator[tp] > max_time_period:
#             #     max_time_period = indicator[tp]
#             #
#             #     print(f'{max_time_period=}')


# # Overlap Studies Functions
#
# # upperband, middleband, lowerband = talib.BBANDS(time_series, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
# # df["bbands_upperband"] = upperband
# # df["bbands_middleband"] = middleband
# # df["bbands_lowerband"] = lowerband
#
# real = talib.DEMA(time_series, timeperiod=30)
# df["DEMA_real"] = real
# real = talib.EMA(time_series, timeperiod=30)
# df["EMA_real"] = real
# real = talib.HT_TRENDLINE(time_series)
# df["HT_TRENDLINE"] = real
# real = talib.KAMA(time_series, timeperiod=30)
# df["KAMA"] = real
# real = talib.MA(time_series, timeperiod=30, matype=0)
# df["MA"] = real
# real = talib.MIDPOINT(time_series, timeperiod=14)
# df["MIDPOINT"] = real
# real = talib.SMA(time_series, timeperiod=30)
# df["SMA"] = real
# real = talib.T3(time_series, timeperiod=5, vfactor=0)
# df["T3"] = real
# real = talib.TEMA(time_series, timeperiod=30)
# df["TEMA"] = real
# real = talib.TRIMA(time_series, timeperiod=30)
# df["TRIMA"] = real
# real = talib.WMA(time_series, timeperiod=30)
# df["WMA"] = real
#
# #mama, fama = MAMA(close, fastlimit=0, slowlimit=0)
# #real = talib.MAVP(time_series, periods=2, minperiod=2, maxperiod=30, matype=0)
#
# # Momentum Indicator Functions
#
# real = APO(time_series, fastperiod=12, slowperiod=26, matype=0)
# df["APO"] = real
# real = CMO(time_series, timeperiod=14)
# df["CMO"] = real
# macd, macdsignal, macdhist = MACD(time_series, fastperiod=12, slowperiod=26, signalperiod=9)
# df["MACD_macd"] = macd
# df["MACD_macdsignal"] = macdsignal
# df["MACD_macdhist"] = macdhist
# macd, macdsignal, macdhist = MACDEXT(time_series, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
# df["MACDEXT_macd"] = macd
# df["MACDEXT_macdsignal"] = macdsignal
# df["MACDEXT_macdhist"] = macdhist
# macd, macdsignal, macdhist = MACDFIX(time_series, signalperiod=9)
# df["MACDFIX_macd"] = macd
# df["MACDFIX_macdsignal"] = macdsignal
# df["MACDFIX_macdhist"] = macdhist
# real = MOM(time_series, timeperiod=10)
# df["MOM"] = real
# real = PPO(time_series, fastperiod=12, slowperiod=26, matype=0)
# df["PPO"] = real
# real = ROC(time_series, timeperiod=10)
# df["ROC"] = real
# real = ROCP(time_series, timeperiod=10)
# df["ROCP"] = real
# real = ROCR(time_series, timeperiod=10)
# df["ROCR"] = real
# real = ROCR100(time_series, timeperiod=10)
# df["ROCR100"] = real
# real = RSI(time_series, timeperiod=14)
# df["RSI"] = real
# fastk, fastd = STOCHRSI(time_series, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
# df["STOCHRSI_fastk"] = fastk
# df["STOCHRSI_fastd"] = fastd
# real = TRIX(time_series, timeperiod=30)
# df["TRIX"] = real
#
# # Cycle Indicator
#
# real = HT_DCPERIOD(time_series)
# df["HT_DCPERIOD"] = real
# real = HT_DCPHASE(time_series)
# df["HT_DCPHASE"] = real
# inphase, quadrature = HT_PHASOR(time_series)
# df["HT_PHASOR_inphase"] = inphase
# df["HT_PHASOR_quadrature"] = quadrature
# sine, leadsine = HT_SINE(time_series)
# df["HT_SINE_sine"] = sine
# df["HT_SINE_leadsine"] = leadsine
# integer = HT_TRENDMODE(time_series)
# df["HT_TRENDMODE"] = integer
#
# # Statistic Functions
#
# real = LINEARREG(time_series, timeperiod=14)
# df["LINEARREG"] = real
# real = LINEARREG_ANGLE(time_series, timeperiod=14)
# df["LINEARREG_ANGLE"] = real
# real = LINEARREG_INTERCEPT(time_series, timeperiod=14)
# df["LINEARREG_INTERCEPT"] = real
# real = LINEARREG_SLOPE(time_series, timeperiod=14)
# df["LINEARREG_SLOPE"] = real
# real = STDDEV(time_series, timeperiod=5, nbdev=1)
# df["STDDEV"] = real
# real = TSF(time_series, timeperiod=14)
# df["TSF"] = real
# real = VAR(time_series, timeperiod=5, nbdev=1)
# df["VAR"] = real