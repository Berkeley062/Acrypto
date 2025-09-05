import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta, date, timezone
from freqtrade.strategy import IStrategy, informative
from freqtrade.persistence import Trade
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame
from functools import reduce
import random
import time

logger = logging.getLogger(__name__)

class AcryptoWeightedStrategyOp(IStrategy):
    """
    Acrypto - Weighted Strategy v1.6.3 for Freqtrade
    Originally created by Alberto Cuadra on TradingView
    https://github.com/AlbertoCuadra/algo_trading_weighted_strategy
    
    Ported to Freqtrade by Berkeley062
    Enhanced by Berkeley0621 with improved profit taking and stop loss mechanisms
    
    更新: 
    - 增加波动率过滤功能，自动排除波动率最大的N个币
    - 增加亏损后自动冷却功能，亏损交易后自动锁定该币种一段时间
    
    This is a weighted strategy combining:
    - MACD
    - Stochastic RSI
    - RSI
    - Supertrend
    - MA Cross
    
    Features:
    - Daily profit limit (30 USDT)
    - Phased entry (7 immediate slots + 3 gradual slots)
    - Take profit levels
    - Dynamic stop loss
    - Time-based exit
    - Volatility filter (excludes top N volatile pairs)
    - Loss cooldown protection
    
    License: © accry - Creative Commons Attribution-NonCommercial-ShareAlike 4.0
    International License - https://creativecommons.org/licenses/by-nc-sa/4.0/
    """
    
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.12
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.06

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100
    
    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60,
    }

    # Optional time in force for orders
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    # Position sizing parameters
    max_entry_position_adjustment = 1
    
    # Define the parameter space for the strategy
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 波动率黑名单功能
        self.volatility_blacklist = set()          # 存储波动率最高的币对
        self._volatility_last_update = 0           # 最后更新时间戳
        self.volatility_filter_enabled = True      # 是否启用波动率过滤
        self.volatility_exclude_top_n = 3          # 排除波动率最高的N个币
        self.volatility_refresh_period = 60        # 刷新周期(分钟)
        
        # 亏损保护机制
        self.loss_cooldown_enabled = True         # 是否启用亏损后冷却
        self.loss_cooldown_period = 60            # 亏损后冷却时间(分钟)
        self.loss_threshold = -0.01               # 触发冷却的亏损阈值
        self.pair_loss_history = {}               # 记录币对亏损历史
        
        # Type trading
        self.allow_longs = True
        self.allow_shorts = True
        
        # Stop loss
        self.use_custom_stoploss = True
        self.stoploss_perc = 0.06
        self.movestoploss = 'TP-2'  # Options: 'None', 'Percentage', 'TP-1', 'TP-2', 'TP-3'
        self.movestoploss_entry = False  # Move stop loss to entry
        self.move_stoploss_factor = 1.20  # 20% move stop loss factor
        
        # Take profits
        self.take_profits = True
        self.MAX_TP = 6
        self.long_profit_perc = 0.068  # 6.8%
        self.long_profit_qty = 15      # 15% of position
        self.short_profit_perc = 0.13  # 13%
        self.short_profit_qty = 10     # 10% of position
        
        # 增强止损参数
        self.use_atr_stoploss = False   # 使用ATR增强止损
        self.atr_stoploss_factor = 2.0  # ATR乘数
        self.max_stoploss = 0.15       # 最大止损限制
        self.max_stoploss_distance = 0.08  # 最大止损距离

        # 时间基础退出参数
        self.use_time_exit = True     # 启用时间基础退出
        self.max_trade_duration = 4  # 最长持仓时间（小时），7天 = 168小时
        self.time_exit_profit_threshold = -0.02  # 亏损超过此值时触发时间退出
        
        # 每日利润管理
        self.daily_profit_limit = 30  # 每日利润限制（USDT）
        self.daily_profit_reached = False  # 是否达到每日利润限制
        self.current_day = None  # 当前交易日
        self.daily_profit = 0  # 当日累计利润
        
        # 交易数量管理参数
        self.max_open_trades = 10       # 最大同时开仓数量
        self.immediate_entry_slots = 6  # 可立即开启的交易数量
        self.min_time_between_trades = 0.5 # 逐步开仓时的最小时间间隔(小时)
        self.last_trade_time = datetime.now(timezone.utc) - timedelta(hours=24)  # 最后一次开仓时间，带时区
                
        # Delays
        self.delay_macd = 1
        self.delay_srsi = 2
        self.delay_rsi = 2
        self.delay_super = 1
        self.delay_cross = 1
        self.delay_exit = 7
        
        # Weights for strategies
        self.str_0 = True  # Use weighted strategy
        self.weight_trigger = 2  # Weight trigger for entry signal
        self.weight_str1 = 1  # Weight for Strategy 1 (MACD)
        self.weight_str2 = 1  # Weight for Strategy 2 (Stoch RSI)
        self.weight_str3 = 1  # Weight for Strategy 3 (RSI)
        self.weight_str4 = 1  # Weight for Strategy 4 (Supertrend)
        self.weight_str5 = 1  # Weight for Strategy 5 (MA Cross)
        
        # Strategy 1: MACD
        self.str_1 = True
        self.MA1_period_1 = 16
        self.MA1_type_1 = 'EMA'
        self.MA2_period_1 = 36
        self.MA2_type_1 = 'EMA'
        
        # Strategy 2: Stoch RSI
        self.str_2 = True
        self.long_RSI = 70
        self.short_RSI = 27
        self.length_RSI = 14
        self.length_stoch = 14
        self.smoothK = 3
        
        # Strategy 3: RSI
        self.str_3 = True
        self.long_RSI2 = 77
        self.short_RSI2 = 30
        
        # Strategy 4: Supertrend
        self.str_4 = True
        self.periods_4 = 2
        self.multiplier = 2.4
        self.change_ATR = True
        
        # Strategy 5: MA Cross
        self.str_5 = True
        self.MA1_period_5 = 46
        self.MA1_type_5 = 'EMA'
        self.MA2_period_5 = 82
        self.MA2_type_5 = 'EMA'
        
        # Potential TOP/BOTTOM
        self.str_6 = False
        self.top_qty = 30
        self.bottom_qty = 30
        self.long_trail_perc = 1.50
        self.short_trail_perc = 1.50
        
        # Custom variables that need to be tracked
        self.custom_info = {}

    def lock_pair_after_loss(self, pair: str, profit: float, current_time: datetime):
        """
        在亏损交易后锁定币对一段时间
        """
        if not self.loss_cooldown_enabled:
            return
            
        # 只在亏损超过阈值时锁定
        if profit > self.loss_threshold:
            return
            
        # 计算锁定结束时间
        lock_end_time = current_time + timedelta(minutes=self.loss_cooldown_period)
        
        # 记录锁定信息
        self.pair_loss_history[pair] = {
            'locked_until': lock_end_time,
            'profit': profit,
            'lock_time': current_time
        }
        
        # 调用 Freqtrade 的锁定函数
        self.lock_pair(pair, until=lock_end_time, reason="Loss protection")
        
        logger.info(f"{pair} - 亏损交易后锁定: 利润 {profit:.2%}, 锁定至 {lock_end_time}")

    def bot_loop_start(self, **kwargs) -> None:
        """
        在每个机器人循环开始时调用
        可用于记录当前锁定状态
        """
        # 获取所有被锁定的币对
        if hasattr(self.dp, 'get_pair_locked_pairs'):
            locked_pairs = self.dp.get_pair_locked_pairs()
            for pair, until in locked_pairs.items():
                # 检查是否是因为亏损导致的锁定
                history = self.pair_loss_history.get(pair, {})
                if history and 'locked_until' in history and history['locked_until'] >= datetime.now(timezone.utc):
                    remaining = (until - datetime.now(timezone.utc)).total_seconds() / 60
                    logger.info(f"亏损保护中: {pair} 锁定至 {until}, 剩余 {remaining:.1f}分钟, "
                               f"触发亏损: {history.get('profit', 0):.2%}")

    def update_volatility_blacklist(self, pairlist):
        """更新波动率黑名单，使用24小时真实波动率"""
        now = time.time()
        
        # 如果禁用了过滤或者刷新周期未到，则直接返回
        if not self.volatility_filter_enabled:
            return
        if now - self._volatility_last_update < self.volatility_refresh_period * 60 and self.volatility_blacklist:
            return
            
        volatilities = []
        for pair in pairlist:
            try:
                # 使用 DataProvider 正确的 API 获取数据
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if len(dataframe) < 24:
                    continue
                    
                # 取最近24根K线
                recent_data = dataframe.tail(24)
                
                # 计算24小时真实波动率（最高点到最低点的比值）
                period_high = recent_data['high'].max()
                period_low = recent_data['low'].min()
                last_close = recent_data['close'].iloc[-1]
                
                # 计算波动率
                if last_close > 0:
                    volatility = (period_high - period_low) / last_close
                    volatilities.append((pair, volatility))
                    
            except Exception as e:
                logger.debug(f"计算{pair}波动率时出错: {e}")
                continue
                
        if not volatilities:
            logger.warning("无法计算波动率，没有足够数据")
            return
            
        # 按波动率降序排序，取波动率最大的N个
        volatilities.sort(key=lambda x: x[1], reverse=True)
        
        # 限制N不超过可用币对数量的一半
        exclude_n = min(self.volatility_exclude_top_n, len(volatilities) // 2)
        
        # 更新黑名单
        old_blacklist = self.volatility_blacklist.copy()
        self.volatility_blacklist = set([p[0] for p in volatilities[:exclude_n]])
        
        # 记录变化
        added = self.volatility_blacklist - old_blacklist
        removed = old_blacklist - self.volatility_blacklist
        
        if added or removed:
            logger.info(f"波动率黑名单更新 - 排除币种: {list(self.volatility_blacklist)}")
            if added:
                logger.info(f"新增黑名单币种: {list(added)}")
            if removed:
                logger.info(f"移除黑名单币种: {list(removed)}")
            
            # 记录前10名的波动率情况，便于分析
            for i, (p, v) in enumerate(volatilities[:10]):
                logger.info(f"波动率排名 #{i+1}: {p} - 波动率: {v:.4f}")
                
        # 更新时间戳
        self._volatility_last_update = now

    def ma(self, dataframe, ma_type, ma_source, ma_period):
        """
        Calculate moving average based on type
        """
        if ma_type == 'SMA':
            return ta.SMA(ma_source, timeperiod=ma_period)
        elif ma_type == 'EMA':
            return ta.EMA(ma_source, timeperiod=ma_period)
        elif ma_type == 'WMA':
            return ta.WMA(ma_source, timeperiod=ma_period)
        elif ma_type == 'RMA':
            # RMA implementation similar to TradingView
            return ta.EMA(ma_source, timeperiod=ma_period)
        elif ma_type == 'HMA':
            # Hull Moving Average
            return ta.WMA(2 * ta.WMA(ma_source, int(ma_period/2)) - ta.WMA(ma_source, ma_period), 
                          int(np.sqrt(ma_period)))
        elif ma_type == 'DEMA':
            # Double Exponential Moving Average
            ema = ta.EMA(ma_source, ma_period)
            return 2 * ema - ta.EMA(ema, ma_period)
        elif ma_type == 'TEMA':
            # Triple Exponential Moving Average
            ema1 = ta.EMA(ma_source, ma_period)
            ema2 = ta.EMA(ema1, ma_period)
            return 3 * (ema1 - ema2) + ta.EMA(ema2, ma_period)
        elif ma_type == 'VWMA':
            # Volume Weighted Moving Average
            return qtpylib.weighted_moving_average(ma_source, ma_period, dataframe['volume'])
        else:
            return ta.SMA(ma_source, timeperiod=ma_period)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds various indicators to the given DataFrame
        """
        # Calculate mid price and other common values
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        
        # Common indicators
        dataframe['volume_ma20'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['avg_volume'] = dataframe['volume_ma20']  # Alias for the original strategy
        dataframe['volume_strength'] = dataframe['volume'] / dataframe['volume_ma20']
        
        # Strategy 1: MACD
        # Using hl2 and high as sources as in the original
        dataframe['MA1'] = self.ma(dataframe, self.MA1_type_1, dataframe['hl2'], self.MA1_period_1)
        dataframe['MA2'] = self.ma(dataframe, self.MA2_type_1, dataframe['high'], self.MA2_period_1)
        
        dataframe['MACD'] = dataframe['MA1'] - dataframe['MA2']
        dataframe['signal'] = self.ma(dataframe, 'SMA', dataframe['MACD'], 9)
        dataframe['trend'] = dataframe['MACD'] - dataframe['signal']
        
        dataframe['long_1'] = dataframe['MACD'] > dataframe['signal']
        dataframe['short_1'] = dataframe['MACD'] < dataframe['signal']
        dataframe['proportion'] = abs(dataframe['MACD'] / dataframe['signal'])
        
        # Create shifted versions for delay logic
        max_delay = max(self.delay_macd, self.delay_srsi, self.delay_rsi, self.delay_super, self.delay_cross)
        for i in range(1, max_delay + 1):
            dataframe[f'long_1_{i}'] = dataframe['long_1'].shift(i)
            dataframe[f'short_1_{i}'] = dataframe['short_1'].shift(i)
            
        # Strategy 1 conditions - completely rewritten to avoid index issues
        # For MACD, we need previous bar to be in same trend and the bar before that to be in opposite trend
        dataframe['long_signal1'] = False
        dataframe['short_signal1'] = False
        
        if self.delay_macd > 1:
            # If delay is > 1, we check current, (delay-1), and delay bars
            dataframe['long_signal1'] = (dataframe['long_1'] & 
                                        dataframe[f'long_1_{self.delay_macd - 1}'] & 
                                        (dataframe[f'long_1_{self.delay_macd}'] == False))
            
            dataframe['short_signal1'] = (dataframe['short_1'] & 
                                        dataframe[f'short_1_{self.delay_macd - 1}'] & 
                                        (dataframe[f'short_1_{self.delay_macd}'] == False))
        else:
            # When delay is 1, we only need current and previous bar
            dataframe['long_signal1'] = dataframe['long_1'] & (dataframe['long_1_1'] == False)
            dataframe['short_signal1'] = dataframe['short_1'] & (dataframe['short_1_1'] == False)
        
        dataframe['close_long1'] = dataframe['short_1'] & (dataframe['long_1_1'] == False)
        dataframe['close_short1'] = dataframe['long_1'] & (dataframe['short_1_1'] == False)
        
        # Strategy 2: Stoch RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.length_RSI)
        dataframe['srsi'] = ta.STOCHRSI(dataframe['close'], timeperiod=self.length_stoch, 
                                      fastk_period=self.length_stoch, fastd_period=self.smoothK)[0]
        dataframe['k'] = self.ma(dataframe, 'SMA', dataframe['srsi'], self.smoothK)
        
        dataframe['isRsiOB'] = dataframe['k'] >= self.long_RSI
        dataframe['isRsiOS'] = dataframe['k'] <= self.short_RSI
        
        # Create shifted versions
        for i in range(1, self.delay_srsi + 1):
            dataframe[f'isRsiOB_{i}'] = dataframe['isRsiOB'].shift(i)
            dataframe[f'isRsiOS_{i}'] = dataframe['isRsiOS'].shift(i)
            
        # Strategy 2 conditions
        dataframe['long_signal2'] = (dataframe[f'isRsiOS_{self.delay_srsi}'] & 
                                    (dataframe['isRsiOB'] == False))
        
        dataframe['short_signal2'] = (dataframe[f'isRsiOB_{self.delay_srsi}'] & 
                                     (dataframe['isRsiOS'] == False))
                                     
        dataframe['close_long2'] = dataframe['short_signal2']
        dataframe['close_short2'] = dataframe['long_signal2']
        
        # Strategy 3: RSI
        dataframe['isRsiOB2'] = dataframe['rsi'] >= self.long_RSI2
        dataframe['isRsiOS2'] = dataframe['rsi'] <= self.short_RSI2
        
        # Create shifted versions
        for i in range(1, self.delay_rsi + 1):
            dataframe[f'isRsiOB2_{i}'] = dataframe['isRsiOB2'].shift(i)
            dataframe[f'isRsiOS2_{i}'] = dataframe['isRsiOS2'].shift(i)
            
        # Strategy 3 conditions
        dataframe['long_signal3'] = (dataframe[f'isRsiOS2_{self.delay_rsi}'] & 
                                    (dataframe['isRsiOB2'] == False))
                                    
        dataframe['short_signal3'] = (dataframe[f'isRsiOB2_{self.delay_rsi}'] & 
                                     (dataframe['isRsiOS2'] == False))
                                     
        dataframe['close_long3'] = dataframe['short_signal3']
        dataframe['close_short3'] = dataframe['long_signal3']
        
        # Strategy 4: Supertrend
        if self.change_ATR:
            dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.periods_4)
        else:
            dataframe['atr'] = ta.SMA(ta.TRUE_RANGE(dataframe['high'], dataframe['low'], dataframe['close']), timeperiod=self.periods_4)
            
        dataframe['up'] = dataframe['hl2'] - self.multiplier * dataframe['atr']
        dataframe['dn'] = dataframe['hl2'] + self.multiplier * dataframe['atr']
        
        # Calculate Supertrend
        # This needs to be calculated iteratively as it depends on previous values
        trend = [0] * len(dataframe)
        up = [0] * len(dataframe)
        dn = [0] * len(dataframe)
        
        for i in range(1, len(dataframe)):
            up[i] = max(dataframe['up'].iloc[i], up[i-1] if dataframe['close'].iloc[i-1] > up[i-1] else dataframe['up'].iloc[i])
            dn[i] = min(dataframe['dn'].iloc[i], dn[i-1] if dataframe['close'].iloc[i-1] < dn[i-1] else dataframe['dn'].iloc[i])
            
            if trend[i-1] == 1 and dataframe['close'].iloc[i] < dn[i-1]:
                trend[i] = -1
            elif trend[i-1] == -1 and dataframe['close'].iloc[i] > up[i-1]:
                trend[i] = 1
            else:
                trend[i] = trend[i-1]
        
        dataframe['trend_supertrend'] = pd.Series(trend, index=dataframe.index)
        dataframe['up_supertrend'] = pd.Series(up, index=dataframe.index)
        dataframe['dn_supertrend'] = pd.Series(dn, index=dataframe.index)
        
        dataframe['long4'] = dataframe['trend_supertrend'] == 1
        dataframe['short4'] = dataframe['trend_supertrend'] == -1
        
        # Create shifted versions
        for i in range(1, self.delay_super + 1):
            dataframe[f'trend_supertrend_{i}'] = dataframe['trend_supertrend'].shift(i)
            
        # Strategy 4 conditions - rewritten to avoid index issues
        if self.delay_super > 1:
            dataframe['long_signal4'] = ((dataframe['trend_supertrend'] == 1) & 
                                       (dataframe[f'trend_supertrend_{self.delay_super - 1}'] == 1) & 
                                       (dataframe[f'trend_supertrend_{self.delay_super}'] == -1))
                                       
            dataframe['short_signal4'] = ((dataframe['trend_supertrend'] == -1) & 
                                        (dataframe[f'trend_supertrend_{self.delay_super - 1}'] == -1) & 
                                        (dataframe[f'trend_supertrend_{self.delay_super}'] == 1))
        else:
            # Simplified when delay_super is 1
            dataframe['long_signal4'] = ((dataframe['trend_supertrend'] == 1) & 
                                       (dataframe['trend_supertrend_1'] == -1))
                                       
            dataframe['short_signal4'] = ((dataframe['trend_supertrend'] == -1) & 
                                        (dataframe['trend_supertrend_1'] == 1))
                                    
        dataframe['changeCond'] = dataframe['trend_supertrend'] != dataframe['trend_supertrend'].shift(1)
        dataframe['close_long4'] = dataframe['short_signal4']
        dataframe['close_short4'] = dataframe['short_signal4']  # Note: this is the same as in original
        
        # Strategy 5: MA Cross
        dataframe['MA12'] = self.ma(dataframe, self.MA1_type_5, dataframe['close'], self.MA1_period_5)
        dataframe['MA22'] = self.ma(dataframe, self.MA2_type_5, dataframe['close'], self.MA2_period_5)
        
        dataframe['long5'] = dataframe['MA12'] > dataframe['MA22']
        dataframe['short5'] = dataframe['MA12'] < dataframe['MA22']
        
        # Create shifted versions
        for i in range(1, self.delay_cross + 1):
            dataframe[f'long5_{i}'] = dataframe['long5'].shift(i)
            dataframe[f'short5_{i}'] = dataframe['short5'].shift(i)
            
        # Strategy 5 conditions - rewritten to avoid index issues
        if self.delay_cross > 1:
            dataframe['long_signal5'] = ((dataframe['long5']) & 
                                       (dataframe[f'long5_{self.delay_cross - 1}']) & 
                                       (dataframe[f'long5_{self.delay_cross}'] == False))
                                       
            dataframe['short_signal5'] = ((dataframe['short5']) & 
                                        (dataframe[f'short5_{self.delay_cross - 1}']) & 
                                        (dataframe[f'short5_{self.delay_cross}'] == False))
        else:
            # Simplified when delay_cross is 1
            dataframe['long_signal5'] = dataframe['long5'] & (dataframe['long5_1'] == False)
            dataframe['short_signal5'] = dataframe['short5'] & (dataframe['short5_1'] == False)
                                    
        dataframe['close_long5'] = dataframe['short5'] & (dataframe['long5_1'] == False)
        dataframe['close_short5'] = dataframe['long5'] & (dataframe['short5_1'] == False)
        
        # Strategy 6: Potential TOP/BOTTOM
        dataframe['volumeRSI_condition'] = ((dataframe['volume'].shift(2) > dataframe['volume'].shift(3)) & 
                                         (dataframe['volume'].shift(2) > dataframe['volume'].shift(4)) & 
                                         (dataframe['volume'].shift(2) > dataframe['volume'].shift(5)))
                                         
        dataframe['condition_OB1'] = ((dataframe['isRsiOB2']) & 
                                    ((dataframe['isRsiOB']) | (dataframe['volume'] < dataframe['avg_volume'] / 2)) & 
                                    (dataframe['volumeRSI_condition']))
                                    
        dataframe['condition_OS1'] = ((dataframe['isRsiOS2']) & 
                                    ((dataframe['isRsiOS']) | (dataframe['volume'] < dataframe['avg_volume'] / 2)) & 
                                    (dataframe['volumeRSI_condition']))
                                    
        dataframe['condition_OB2'] = ((dataframe['volume'].shift(2) / dataframe['volume'].shift(1) > (1.0 + self.long_trail_perc)) & 
                                    (dataframe['isRsiOB']) & 
                                    (dataframe['volumeRSI_condition']))
                                    
        dataframe['condition_OS2'] = ((dataframe['volume'].shift(2) / dataframe['volume'].shift(1) > (1.0 + self.short_trail_perc)) & 
                                    (dataframe['isRsiOS']) & 
                                    (dataframe['volumeRSI_condition']))
                                    
        dataframe['condition_OB3'] = self.weight_total(dataframe,
                                                   dataframe['MACD'] < dataframe['signal'],
                                                   dataframe['isRsiOB'],
                                                   dataframe['isRsiOB2'],
                                                   dataframe['short4'],
                                                   dataframe['short5']) >= self.weight_trigger
                                                   
        dataframe['condition_OS3'] = self.weight_total(dataframe,
                                                   dataframe['MACD'] > dataframe['signal'],
                                                   dataframe['isRsiOS'],
                                                   dataframe['isRsiOS2'],
                                                   dataframe['long4'],
                                                   dataframe['long5']) >= self.weight_trigger
                                                   
        dataframe['condition_OB'] = dataframe['condition_OB1'] | dataframe['condition_OB2']
        dataframe['condition_OS'] = dataframe['condition_OS1'] | dataframe['condition_OS2']
        
        # Multi-bar logic for condition_OB_several and condition_OS_several
        # Logic: check if condition was true in previous bar and one of bars 2-7
        for i in range(1, 8):
            dataframe[f'condition_OB_{i}'] = dataframe['condition_OB'].shift(i)
            dataframe[f'condition_OS_{i}'] = dataframe['condition_OS'].shift(i)
        
        condition_OB_several = pd.Series(False, index=dataframe.index)
        condition_OS_several = pd.Series(False, index=dataframe.index)
        
        for i in range(2, 8):
            condition_OB_several |= (dataframe['condition_OB_1'] & dataframe[f'condition_OB_{i}'])
            condition_OS_several |= (dataframe['condition_OS_1'] & dataframe[f'condition_OS_{i}'])
            
        dataframe['condition_OB_several'] = condition_OB_several
        dataframe['condition_OS_several'] = condition_OS_several
        
        # Calculate total weights for entries
        if self.str_0:
            dataframe['w_total_long'] = self.weight_total(dataframe,
                                                     dataframe['long_signal1'],
                                                     dataframe['long_signal2'],
                                                     dataframe['long_signal3'],
                                                     dataframe['long_signal4'],
                                                     dataframe['long_signal5'])
                                                     
            dataframe['w_total_short'] = self.weight_total(dataframe,
                                                      dataframe['short_signal1'],
                                                      dataframe['short_signal2'],
                                                      dataframe['short_signal3'],
                                                      dataframe['short_signal4'],
                                                      dataframe['short_signal5'])
        
        return dataframe

    def weight_values(self, signal) -> float:
        """
        Return weight value based on signal
        """
        return 1.0 if signal else 0.0

    def weight_total(self, dataframe, signal1, signal2, signal3, signal4, signal5) -> pd.Series:
        """
        Calculate weighted total of signals
        """
        # Convert boolean series to numeric (1.0 for True, 0.0 for False)
        s1 = signal1.astype(float) * self.weight_str1
        s2 = signal2.astype(float) * self.weight_str2
        s3 = signal3.astype(float) * self.weight_str3
        s4 = signal4.astype(float) * self.weight_str4
        s5 = signal5.astype(float) * self.weight_str5
        
        # Sum all signals
        return s1 + s2 + s3 + s4 + s5

    def check_daily_profit_limit(self, current_time: datetime) -> bool:
        """
        检查是否达到每日利润限制，如果是新的一天则重置计数器
        返回True表示已达到限制，False表示未达到限制
        """
        # 获取当前日期
        today = current_time.date()
        
        # 如果是新的一天或首次运行，重置计数器
        if self.current_day != today:
            self.current_day = today
            self.daily_profit = 0
            self.daily_profit_reached = False
            logger.info(f"重置每日利润计数器，新的交易日: {today}")
        
        # 检查是否已达到每日利润限制
        if self.daily_profit >= self.daily_profit_limit:
            if not self.daily_profit_reached:
                self.daily_profit_reached = True
                logger.info(f"已达到每日利润限制 {self.daily_profit_limit}U，当前利润: {self.daily_profit:.2f}U")
            return True
        
        return False

    def update_daily_profit(self, profit_amount: float) -> None:
        """
        更新每日利润累计值
        """
        self.daily_profit += profit_amount
        logger.info(f"更新每日利润: +{profit_amount:.2f}U, 累计: {self.daily_profit:.2f}U")

    def should_exit_all_positions(self, current_time: datetime) -> bool:
        """
        检查是否应该退出所有头寸（例如：达到每日利润限制）
        """
        return self.check_daily_profit_limit(current_time)

    def can_enter_new_trade(self, pair: str, current_time: datetime) -> bool:
        """
        检查是否允许开新仓
        """
        # 检查是否达到每日利润限制
        if self.check_daily_profit_limit(current_time):
            logger.info(f"{pair} - 拒绝入场: 已达到每日利润限制 {self.daily_profit_limit}U")
            return False
        
        # 获取当前开放的交易数量
        open_trades = Trade.get_trades_proxy(is_open=True)
        current_open_trades = len(open_trades)
        
        # 检查是否达到最大开仓数量
        if current_open_trades >= self.max_open_trades:
            logger.info(f"{pair} - 拒绝入场: 已达到最大开仓数量 {current_open_trades}/{self.max_open_trades}")
            return False
        
        # 如果未达到即时开仓数量限制，允许立即开仓
        if current_open_trades < self.immediate_entry_slots:
            logger.info(f"{pair} - 允许入场: 即时开仓阶段 {current_open_trades+1}/{self.immediate_entry_slots}")
            return True
        
        # 对于需要逐步开仓的交易，检查时间间隔
        time_since_last_entry = (current_time - self.last_trade_time).total_seconds() / 3600
        if time_since_last_entry < self.min_time_between_trades:
            logger.info(f"{pair} - 拒绝入场: 逐步开仓阶段，距离上次开仓时间不足 {self.min_time_between_trades}小时"
                      f"（当前间隔: {time_since_last_entry:.2f}小时）")
            return False
        
        # 允许开仓
        logger.info(f"{pair} - 允许入场: 逐步开仓阶段 {current_open_trades+1}/{self.max_open_trades}，"
                  f"距离上次开仓: {time_since_last_entry:.2f}小时")
        return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''
        
        # Generate entry signals
        if self.str_0:  # Use weighted strategy approach
            # Adjust weights if strategies are disabled
            if not self.str_1:
                self.weight_str1 = 0
            if not self.str_2:
                self.weight_str2 = 0
            if not self.str_3:
                self.weight_str3 = 0
            if not self.str_4:
                self.weight_str4 = 0
            if not self.str_5:
                self.weight_str5 = 0
                
            # Long entries
            long_mask = ((dataframe['w_total_long'] >= self.weight_trigger) & 
                       (self.allow_longs))
            dataframe.loc[long_mask, 'enter_long'] = 1
            dataframe.loc[long_mask, 'enter_tag'] = 'weighted_long'
            
            # Short entries
            short_mask = ((dataframe['w_total_short'] >= self.weight_trigger) & 
                        (self.allow_shorts))
            dataframe.loc[short_mask, 'enter_short'] = 1
            dataframe.loc[short_mask, 'enter_tag'] = 'weighted_short'
        else:  # Use individual strategies
            if self.allow_longs:
                # Strategy 1: MACD
                if self.str_1:
                    mask = dataframe['long_signal1']
                    dataframe.loc[mask, 'enter_long'] = 1
                    dataframe.loc[mask, 'enter_tag'] = 'long_macd'
                
                # Strategy 2: Stoch RSI
                if self.str_2:
                    mask = dataframe['long_signal2']
                    dataframe.loc[mask & (dataframe['enter_long'] != 1), 'enter_long'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'long_stoch_rsi'
                
                # Strategy 3: RSI
                if self.str_3:
                    mask = dataframe['long_signal3']
                    dataframe.loc[mask & (dataframe['enter_long'] != 1), 'enter_long'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'long_rsi'
                
                # Strategy 4: Supertrend
                if self.str_4:
                    mask = dataframe['long_signal4']
                    dataframe.loc[mask & (dataframe['enter_long'] != 1), 'enter_long'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'long_supertrend'
                
                # Strategy 5: MA Cross
                if self.str_5:
                    mask = dataframe['long_signal5']
                    dataframe.loc[mask & (dataframe['enter_long'] != 1), 'enter_long'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'long_ma_cross'
            
            if self.allow_shorts:
                # Strategy 1: MACD
                if self.str_1:
                    mask = dataframe['short_signal1']
                    dataframe.loc[mask, 'enter_short'] = 1
                    dataframe.loc[mask, 'enter_tag'] = 'short_macd'
                
                # Strategy 2: Stoch RSI
                if self.str_2:
                    mask = dataframe['short_signal2']
                    dataframe.loc[mask & (dataframe['enter_short'] != 1), 'enter_short'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'short_stoch_rsi'
                
                # Strategy 3: RSI
                if self.str_3:
                    mask = dataframe['short_signal3']
                    dataframe.loc[mask & (dataframe['enter_short'] != 1), 'enter_short'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'short_rsi'
                
                # Strategy 4: Supertrend
                if self.str_4:
                    mask = dataframe['short_signal4']
                    dataframe.loc[mask & (dataframe['enter_short'] != 1), 'enter_short'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'short_supertrend'
                
                # Strategy 5: MA Cross
                if self.str_5:
                    mask = dataframe['short_signal5']
                    dataframe.loc[mask & (dataframe['enter_short'] != 1), 'enter_short'] = 1
                    dataframe.loc[mask & (dataframe['enter_tag'] == ''), 'enter_tag'] = 'short_ma_cross'
        
        # 更新波动率黑名单
        pairlist = self.dp.current_whitelist()
        self.update_volatility_blacklist(pairlist)
        
        # 如果当前币对在黑名单中，清除所有入场信号
        pair = metadata['pair']
        if self.volatility_filter_enabled and pair in self.volatility_blacklist:
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
            dataframe['enter_tag'] = ''
            logger.debug(f"{pair} - 信号已过滤: 波动率排名靠前 {self.volatility_exclude_top_n}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        dataframe.loc[:, 'exit_tag'] = ''
        
        # Long exits
        if self.allow_longs and not self.allow_shorts:
            if self.str_0:  # Use weighted strategy approach
                # Exit long when short signal meets weight threshold
                mask = (dataframe['w_total_short'] >= self.weight_trigger)
                dataframe.loc[mask, 'exit_long'] = 1
                dataframe.loc[mask, 'exit_tag'] = 'weighted_short'
            else:  # Use individual strategies
                # Strategy 1: MACD
                if self.str_1:
                    mask = dataframe['close_long1']
                    dataframe.loc[mask, 'exit_long'] = 1
                    dataframe.loc[mask, 'exit_tag'] = 'exit_macd'
                
                # Strategy 2: Stoch RSI
                if self.str_2:
                    mask = dataframe['close_long2']
                    dataframe.loc[mask & (dataframe['exit_long'] != 1), 'exit_long'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_stoch_rsi'
                
                # Strategy 3: RSI
                if self.str_3:
                    mask = dataframe['close_long3']
                    dataframe.loc[mask & (dataframe['exit_long'] != 1), 'exit_long'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_rsi'
                
                # Strategy 4: Supertrend
                if self.str_4:
                    mask = dataframe['close_long4']
                    dataframe.loc[mask & (dataframe['exit_long'] != 1), 'exit_long'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_supertrend'
                
                # Strategy 5: MA Cross
                if self.str_5:
                    mask = dataframe['close_long5']
                    dataframe.loc[mask & (dataframe['exit_long'] != 1), 'exit_long'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_ma_cross'
        
        # Short exits
        if self.allow_shorts and not self.allow_longs:
            if self.str_0:  # Use weighted strategy approach
                # Exit short when long signal meets weight threshold
                mask = (dataframe['w_total_long'] >= self.weight_trigger)
                dataframe.loc[mask, 'exit_short'] = 1
                dataframe.loc[mask, 'exit_tag'] = 'weighted_long'
            else:  # Use individual strategies
                # Strategy 1: MACD
                if self.str_1:
                    mask = dataframe['close_short1']
                    dataframe.loc[mask, 'exit_short'] = 1
                    dataframe.loc[mask, 'exit_tag'] = 'exit_macd'
                
                # Strategy 2: Stoch RSI
                if self.str_2:
                    mask = dataframe['close_short2']
                    dataframe.loc[mask & (dataframe['exit_short'] != 1), 'exit_short'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_stoch_rsi'
                
                # Strategy 3: RSI
                if self.str_3:
                    mask = dataframe['close_short3']
                    dataframe.loc[mask & (dataframe['exit_short'] != 1), 'exit_short'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_rsi'
                
                # Strategy 4: Supertrend
                if self.str_4:
                    mask = dataframe['close_short4']
                    dataframe.loc[mask & (dataframe['exit_short'] != 1), 'exit_short'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_supertrend'
                
                # Strategy 5: MA Cross
                if self.str_5:
                    mask = dataframe['close_short5']
                    dataframe.loc[mask & (dataframe['exit_short'] != 1), 'exit_short'] = 1
                    dataframe.loc[mask & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'exit_ma_cross'
                    
        # Additional exits for potential TOP/BOTTOM (Strategy 6)
        if self.str_6:
            # TOP detection for long exit
            if self.allow_longs:
                mask = dataframe['condition_OB_several']
                if self.top_qty == 100:  # Full exit
                    dataframe.loc[mask, 'exit_long'] = 1
                    dataframe.loc[mask, 'exit_tag'] = 'top_detection'
            
            # BOTTOM detection for short exit
            if self.allow_shorts:
                mask = dataframe['condition_OS_several']
                if self.bottom_qty == 100:  # Full exit
                    dataframe.loc[mask, 'exit_short'] = 1
                    dataframe.loc[mask, 'exit_tag'] = 'bottom_detection'
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new stoploss percentage.
        """
        # 如果不使用自定义止损，返回默认值
        if not self.use_custom_stoploss:
            return self.stoploss
        
        # 检查是否达到每日利润限制，如果是，返回当前利润作为止损（实现立即平仓）
        if self.should_exit_all_positions(current_time) and current_profit > 0:
            return current_profit
        
        # 确定交易方向 - 兼容不同版本的 Freqtrade
        is_long = True  # 默认假设是多头交易
        if hasattr(trade, 'is_short') and trade.is_short:
            is_long = False
        
        # 获取利润百分比
        profit_perc = self.long_profit_perc if is_long else self.short_profit_perc
        
        # 初始化止损值
        sl_percent = -self.stoploss_perc  # 对于多头是负数，对于空头稍后会反转
        
        # 获取 ATR 动态止损值（如果启用）
        atr_stoploss = None
        if self.use_atr_stoploss:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                last_candle = dataframe.iloc[-1]
                atr_value = last_candle['atr']
                
                if is_long:
                    atr_stoploss = -self.atr_stoploss_factor * atr_value / current_rate
                else:
                    atr_stoploss = self.atr_stoploss_factor * atr_value / current_rate
                    
                # 限制 ATR 止损的最大值
                if is_long and atr_stoploss < -self.max_stoploss:
                    atr_stoploss = -self.max_stoploss
                elif not is_long and atr_stoploss > self.max_stoploss:
                    atr_stoploss = self.max_stoploss
        
        # 获取交易以来最高/最低价格
        if is_long:  # 多头交易
            # 获取达到的最高价
            peak_price = max(current_rate, trade.max_rate)
            trade.max_rate = peak_price  # 更新历史最高价
            
            # 检查达到了哪个止盈等级
            for i in range(1, self.MAX_TP + 1):
                tp_price = trade.open_rate * (1 + i * profit_perc)
                
                # 如果我们达到了这个止盈等级
                if peak_price > tp_price:
                    # 根据移动止损设置应用不同的止损逻辑
                    if self.movestoploss == 'Percentage':
                        sl_percent = (1 + i * profit_perc - self.stoploss_perc * self.move_stoploss_factor) - 1
                    elif self.movestoploss == 'TP-1' and i > 1:
                        # 移动到 TP-1 等级
                        sl_percent = (1 + (i-1) * profit_perc) - 1
                    elif self.movestoploss == 'TP-2' and i > 2:
                        # 移动到 TP-2 等级
                        sl_percent = (1 + (i-2) * profit_perc) - 1
                    elif self.movestoploss == 'TP-3' and i > 3:
                        # 移动到 TP-3 等级
                        sl_percent = (1 + (i-3) * profit_perc) - 1
                    
                    # 如果使用 movestoploss_entry 并且等级匹配，移动到入场点
                    if self.movestoploss_entry:
                        if ((self.movestoploss == 'TP-1' and i == 1) or 
                            (self.movestoploss == 'TP-2' and i <= 2) or 
                            (self.movestoploss == 'TP-3' and i <= 3)):
                            sl_percent = 0  # 入场点
            
            # 检查并应用 ATR 止损
            if atr_stoploss is not None:
                # 选择更严格的止损
                if current_profit > 0:
                    # 盈利时，选择更宽松的止损（较高的值）
                    sl_percent = max(sl_percent, atr_stoploss)
                else:
                    # 亏损时，选择更严格的止损（较高的值）
                    sl_percent = max(sl_percent, atr_stoploss)
                    
            # 确保止损不会超过允许的最大距离
            if current_profit > 0 and sl_percent < 0:
                # 如果有利润，确保止损距离当前价格不会太远
                distance = current_profit - sl_percent
                if distance > self.max_stoploss_distance:
                    sl_percent = current_profit - self.max_stoploss_distance
                    
        else:  # 空头交易
            # 获取达到的最低价
            lowest_price = min(current_rate, trade.min_rate if hasattr(trade, 'min_rate') else current_rate)
            if hasattr(trade, 'min_rate'):
                trade.min_rate = lowest_price  # 更新历史最低价
            
            # 检查达到了哪个止盈等级
            for i in range(1, self.MAX_TP + 1):
                tp_price = trade.open_rate * (1 - i * profit_perc)
                
                # 如果我们达到了这个止盈等级
                if lowest_price < tp_price:
                    # 根据移动止损设置应用不同的止损逻辑
                    if self.movestoploss == 'Percentage':
                        sl_percent = (1 - i * profit_perc + self.stoploss_perc * self.move_stoploss_factor) - 1
                    elif self.movestoploss == 'TP-1' and i > 1:
                        # 移动到 TP-1 等级
                        sl_percent = (1 - (i-1) * profit_perc) - 1
                    elif self.movestoploss == 'TP-2' and i > 2:
                        # 移动到 TP-2 等级
                        sl_percent = (1 - (i-2) * profit_perc) - 1
                    elif self.movestoploss == 'TP-3' and i > 3:
                        # 移动到 TP-3 等级
                        sl_percent = (1 - (i-3) * profit_perc) - 1
                    
                    # 如果使用 movestoploss_entry 并且等级匹配，移动到入场点
                    if self.movestoploss_entry:
                        if ((self.movestoploss == 'TP-1' and i == 1) or 
                            (self.movestoploss == 'TP-2' and i <= 2) or 
                            (self.movestoploss == 'TP-3' and i <= 3)):
                            sl_percent = 0  # 入场点
            
            # 检查并应用 ATR 止损
            if atr_stoploss is not None:
                # 选择更严格的止损
                if current_profit > 0:
                    # 盈利时，选择更宽松的止损（较低的值）
                    sl_percent = min(sl_percent, atr_stoploss)
                else:
                    # 亏损时，选择更严格的止损（较低的值）
                    sl_percent = min(sl_percent, atr_stoploss)
                    
            # 确保止损不会超过允许的最大距离
            if current_profit > 0 and sl_percent < 0:
                # 如果有利润，确保止损距离当前价格不会太远
                distance = current_profit - sl_percent
                if distance > self.max_stoploss_distance:
                    sl_percent = current_profit - self.max_stoploss_distance
            
            # 对于空头，我们需要取反止损百分比
            sl_percent = -sl_percent
                
        return sl_percent

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[Tuple[str, float]]:
        """
        Custom exit signal logic with support for partial exits.
        Returns tuple (exit_tag, exit_percentage) or just exit_tag for full exit.
        """
        # 检查是否达到每日利润限制
        if self.should_exit_all_positions(current_time):
            return "daily_profit_limit_reached"
        
        # 获取此币对的数据帧
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # 如果数据帧为空则跳过
        if dataframe.empty:
            return None
        
        # 获取最后一根蜡烛
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 确定交易方向 - 兼容不同版本的Freqtrade
        is_long = True  # 默认假设是多头交易
        if hasattr(trade, 'is_short') and trade.is_short:
            is_long = False
        
        # 创建或获取交易自定义信息存储
        trade_data = self.custom_info.setdefault(trade.pair, {})
        
        # 分批止盈逻辑
        if self.take_profits:
            profit_perc = self.long_profit_perc if is_long else self.short_profit_perc
            exit_qty_perc = self.long_profit_qty if is_long else self.short_profit_qty
            
            # 检查每个止盈等级
            for i in range(1, self.MAX_TP + 1):
                tp_level = i * profit_perc
                tp_key = f"tp_level_{i}_executed"
                
                # 检查是否达到该止盈等级且之前未触发
                if current_profit >= tp_level and not trade_data.get(tp_key, False):
                    # 标记此止盈等级为已执行
                    trade_data[tp_key] = True
                    self.custom_info[trade.pair] = trade_data
                    
                    # 决定退出数量
                    exit_proportion = exit_qty_perc / 100
                    
                    # 记录止盈
                    logger.info(f"{pair} - 触发止盈等级 {i}: 当前利润 {current_profit:.2%}, "
                              f"止盈阈值 {tp_level:.2%}, 退出比例 {exit_qty_perc}%")
                    
                    # 更新每日利润统计
                    profit_amount = trade.stake_amount * current_profit * exit_proportion
                    self.update_daily_profit(profit_amount)
                    
                    return f"tp_level_{i}", exit_proportion
        
        # Strategy 6: 潜在的顶部/底部部分退出
        if self.str_6:
            if is_long and last_candle['condition_OB_several'] and self.top_qty < 100:
                # 部分退出多头
                exit_proportion = self.top_qty / 100
                profit_amount = trade.stake_amount * current_profit * exit_proportion
                self.update_daily_profit(profit_amount)
                return "top_detection_partial", exit_proportion
                
            if not is_long and last_candle['condition_OS_several'] and self.bottom_qty < 100:
                # 部分退出空头
                exit_proportion = self.bottom_qty / 100
                profit_amount = trade.stake_amount * current_profit * exit_proportion
                self.update_daily_profit(profit_amount)
                return "bottom_detection_partial", exit_proportion
        
        # 时间基础退出
        if self.use_time_exit:
            # 计算交易持续时间（小时）
            trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
            
            # 如果交易持续时间超过阈值且利润低于阈值，则退出
            if (trade_duration > self.max_trade_duration and 
                current_profit < self.time_exit_profit_threshold):
                
                # 更新每日利润统计
                profit_amount = trade.stake_amount * current_profit
                self.update_daily_profit(profit_amount)
                
                return f"timeout_exit_{self.max_trade_duration}h"
        
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        """
        Called before placing a buy order.
        :return: True if the trade entry should be confirmed, False otherwise
        """
        # 检查是否在波动率黑名单中
        if self.volatility_filter_enabled and pair in self.volatility_blacklist:
            logger.info(f"{pair} - 拒绝入场: 此币对波动率过高 (在黑名单中)")
            return False
            
        # 检查是否允许开新仓
        if not self.can_enter_new_trade(pair, current_time):
            return False
        
        # 获取当前开放的交易数量
        open_trades = Trade.get_trades_proxy(is_open=True)
        current_open_trades = len(open_trades)
        
        # 始终更新最后入场时间
        self.last_trade_time = current_time
        logger.info(f"{pair} - 更新最后开仓时间: {current_time}")
               
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Called before placing a regular sell order.
        :return: True if the trade exit should be confirmed, False otherwise
        """
        # 如果是因为每日利润限制触发的退出，始终确认
        if exit_reason == "daily_profit_limit_reached":
            logger.info(f"{pair} - 确认退出: 已达到每日利润限制")
            return True
            
        # 计算预计的利润金额
        profit_ratio = trade.calc_profit_ratio(rate)
        profit_amount = trade.stake_amount * profit_ratio
        
        # 如果是亏损交易，锁定此币对
        if profit_ratio < self.loss_threshold:
            self.lock_pair_after_loss(pair, profit_ratio, current_time)
        
        # 如果是部分退出，调整利润金额
        if "," in str(amount):  # 部分退出的标志
            # 解析退出的比例
            _, proportion = exit_reason.split(",")
            proportion = float(proportion)
            profit_amount *= proportion
        
        # 更新每日利润统计
        self.update_daily_profit(profit_amount)
        
        return True
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade.
        :return: A leverage amount
        """
        # 尝试使用3倍杠杆，但不超过交易所允许的最大值
        actual_leverage = min(3.0, max_leverage)
        
        # 记录实际使用的杠杆
        logger.info(f"交易对 {pair} - 请求杠杆: 3.0, 交易所允许最大: {max_leverage}, 实际使用: {actual_leverage}")
        
        return actual_leverage
