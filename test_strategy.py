#!/usr/bin/env python3
"""
Test script for AlexNexusForgeML strategy
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

# Mock FreqTrade components for testing
class MockTrade:
    def __init__(self, open_rate=100.0, open_date=None):
        self.open_rate = open_rate
        self.open_date_utc = open_date or datetime.now()
    
    def calc_profit_ratio(self, rate):
        return (rate - self.open_rate) / self.open_rate

class MockStrategy:
    class DecimalParameter:
        def __init__(self, low, high, default, optimize=True, space='buy'):
            self.value = default
    
    class IntParameter:
        def __init__(self, low, high, default, optimize=True, space='buy'):
            self.value = default
    
    class BooleanParameter:
        def __init__(self, default, optimize=True, space='buy'):
            self.value = default

# Mock freqtrade modules
class MockFreqTrade:
    def __init__(self):
        pass

sys.modules['freqtrade'] = MockFreqTrade()
sys.modules['freqtrade.strategy'] = type('MockModule', (), {
    'IStrategy': type,
    'DecimalParameter': MockStrategy.DecimalParameter,
    'IntParameter': MockStrategy.IntParameter,
    'BooleanParameter': MockStrategy.BooleanParameter
})()
sys.modules['freqtrade.persistence'] = type('MockModule', (), {'Trade': MockTrade})()
sys.modules['freqtrade.vendor'] = type('MockModule', (), {})()
sys.modules['freqtrade.vendor.qtpylib'] = type('MockModule', (), {})()
sys.modules['freqtrade.vendor.qtpylib.indicators'] = type('MockModule', (), {})()

def create_test_dataframe(periods=500):
    """Create test dataframe with realistic OHLCV data"""
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 50000
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='5min')
    
    # Random walk with trend
    returns = np.random.normal(0.0001, 0.01, periods)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Create OHLCV
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, periods)),
        'low': prices * (1 - np.random.uniform(0, 0.02, periods)),
        'close': prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
        'volume': np.random.uniform(1000, 10000, periods)
    })
    
    # Ensure high >= close >= low
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    return df

def test_strategy():
    """Test the AlexNexusForgeML strategy"""
    try:
        from AlexNexusForgeML import AlexNexusForgeML
        print("✅ Strategy import successful")
        
        # Initialize strategy
        config = {}
        strategy = AlexNexusForgeML(config)
        print("✅ Strategy initialization successful")
        
        # Create test data
        df = create_test_dataframe(300)
        metadata = {'pair': 'BTC/USDT:USDT'}
        print(f"✅ Test dataframe created: {len(df)} rows")
        
        # Test populate_indicators
        print("🔧 Testing populate_indicators...")
        df_with_indicators = strategy.populate_indicators(df.copy(), metadata)
        print(f"✅ Indicators populated: {len(df_with_indicators.columns)} columns")
        
        # Check for ML features
        ml_columns = [col for col in df_with_indicators.columns if 'ml_' in col]
        print(f"🤖 ML columns found: {len(ml_columns)}")
        for col in ml_columns[:5]:  # Show first 5
            print(f"   - {col}")
        
        # Test populate_entry_trend
        print("🎯 Testing populate_entry_trend...")
        df_with_entries = strategy.populate_entry_trend(df_with_indicators.copy(), metadata)
        entry_signals = df_with_entries['enter_long'].sum()
        print(f"✅ Entry signals generated: {entry_signals}")
        
        # Test populate_exit_trend
        print("🛑 Testing populate_exit_trend...")
        df_with_exits = strategy.populate_exit_trend(df_with_entries.copy(), metadata)
        exit_signals = df_with_exits['exit_long'].sum()
        print(f"✅ Exit signals generated: {exit_signals}")
        
        # Test trade confirmation methods
        print("🔍 Testing trade confirmation methods...")
        
        # Test entry confirmation
        entry_confirmed = strategy.confirm_trade_entry(
            'BTC/USDT:USDT', 'market', 0.1, 50000, 'gtc', 
            datetime.now(), 'ml_high_conf'
        )
        print(f"✅ Entry confirmation: {entry_confirmed}")
        
        # Test exit confirmation
        mock_trade = MockTrade(50000, datetime.now() - timedelta(hours=2))
        exit_confirmed = strategy.confirm_trade_exit(
            'BTC/USDT:USDT', mock_trade, 'market', 0.1, 51000, 
            'gtc', 'ml_exit', datetime.now()
        )
        print(f"✅ Exit confirmation: {exit_confirmed}")
        
        # Test custom stoploss
        stoploss = strategy.custom_stoploss(
            'BTC/USDT:USDT', mock_trade, datetime.now(), 51000, 0.02
        )
        print(f"✅ Custom stoploss: {stoploss}")
        
        # Show sample of final dataframe
        print("\n📊 Sample of final dataframe:")
        key_columns = ['close', 'ml_entry_probability', 'ml_confidence', 'enter_long', 'exit_long', 'enter_tag']
        available_columns = [col for col in key_columns if col in df_with_exits.columns]
        if available_columns:
            print(df_with_exits[available_columns].tail(10).to_string())
        
        # Strategy statistics
        print(f"\n📈 Strategy Statistics:")
        print(f"   Total candles: {len(df_with_exits)}")
        print(f"   Entry signals: {df_with_exits['enter_long'].sum()}")
        print(f"   Exit signals: {df_with_exits['exit_long'].sum()}")
        print(f"   Entry rate: {df_with_exits['enter_long'].mean():.2%}")
        
        if 'ml_entry_probability' in df_with_exits.columns:
            avg_ml_prob = df_with_exits['ml_entry_probability'].mean()
            print(f"   Avg ML probability: {avg_ml_prob:.3f}")
        
        if 'composite_score' in df_with_exits.columns:
            avg_score = df_with_exits['composite_score'].mean()
            print(f"   Avg composite score: {avg_score:.1f}")
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_strategy()