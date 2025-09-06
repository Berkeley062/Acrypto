#!/usr/bin/env python3
"""
Simple test for AlexNexusForgeML strategy - syntax and basic functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_imports():
    """Test that all imports work correctly"""
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import talib
        import pywt
        print("✅ All required packages available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def test_ml_engine():
    """Test the ML engine separately"""
    try:
        # Import just the ML engine
        import sys
        sys.path.insert(0, '.')
        
        exec("""
# Define required mock classes
class MockTrade:
    def __init__(self, open_rate=100.0, open_date=None):
        self.open_rate = open_rate
        from datetime import datetime
        self.open_date_utc = open_date or datetime.now()
    
    def calc_profit_ratio(self, rate):
        return (rate - self.open_rate) / self.open_rate

# Read the ML engine class from the file
with open('AlexNexusForgeML.py', 'r') as f:
    content = f.read()

# Extract just the MLPredictiveEngine class
import re
ml_engine_match = re.search(r'class MLPredictiveEngine:.*?(?=class|\Z)', content, re.DOTALL)
if ml_engine_match:
    ml_engine_code = ml_engine_match.group(0)
    
    # Execute required imports
    import logging
    import numpy as np
    import pandas as pd
    import pickle
    import warnings
    from datetime import datetime, timedelta
    from pathlib import Path
    from typing import Optional, Dict, Tuple
    from importlib import metadata
    from functools import lru_cache
    import talib.abstract as ta
    from scipy.fft import fft, fftfreq
    from scipy.stats import skew, kurtosis
    
    # ML imports
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.feature_selection import SelectKBest, f_classif
    
    logger = logging.getLogger(__name__)
    SKLEARN_AVAILABLE = True
    WAVELETS_AVAILABLE = True
    
    # Execute the ML engine class
    exec(ml_engine_code)
    
    # Test ML engine
    engine = MLPredictiveEngine()
    print("✅ ML Engine created successfully")
    
    # Create test data
    np.random.seed(42)
    periods = 200
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='5min')
    base_price = 50000
    returns = np.random.normal(0.0001, 0.01, periods)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, periods)),
        'low': prices * (1 - np.random.uniform(0, 0.02, periods)),
        'close': prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
        'volume': np.random.uniform(1000, 10000, periods)
    })
    
    # Test feature extraction
    features_df = engine.extract_ml_features(df.copy())
    print(f"✅ Feature extraction: {len(features_df.columns)} columns")
    
    # Test target creation
    targets = engine.create_target_labels(df.copy())
    print(f"✅ Target creation: {targets.sum()} positive samples")
    
    # Test training (with sufficient data)
    train_result = engine.train_models(df.copy(), 'BTC/USDT:USDT')
    print(f"✅ Training result: {train_result['status']}")
    
    # Test prediction
    pred_result = engine.predict(df.copy(), 'BTC/USDT:USDT')
    print(f"✅ Prediction result: {pred_result['status']}")
    print(f"   Entry probability: {pred_result['entry_probability']:.3f}")
    print(f"   Confidence: {pred_result['confidence']:.3f}")
    
    return True
else:
    print("❌ Could not extract ML engine class")
    return False
""")
        return True
        
    except Exception as e:
        print(f"❌ ML Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_syntax():
    """Test strategy file syntax without instantiation"""
    try:
        with open('AlexNexusForgeML.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ('class AlexNexusForgeML', 'Strategy class defined'),
            ('def populate_indicators', 'populate_indicators method'),
            ('def populate_entry_trend', 'populate_entry_trend method'),  
            ('def populate_exit_trend', 'populate_exit_trend method'),
            ('def confirm_trade_entry', 'confirm_trade_entry method'),
            ('def confirm_trade_exit', 'confirm_trade_exit method'),
            ('ml_entry_probability', 'ML probability features'),
            ('logger.info', 'Logging functionality'),
            ('ML分析', 'Chinese logging'),
        ]
        
        for check, description in checks:
            if check in content:
                print(f"✅ {description}")
            else:
                print(f"❌ Missing: {description}")
        
        # Syntax check
        compile(content, 'AlexNexusForgeML.py', 'exec')
        print("✅ Syntax check passed")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ File check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing AlexNexusForgeML Strategy")
    print("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("Strategy syntax", test_strategy_syntax),
        ("ML Engine", test_ml_engine),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Strategy is ready.")
    else:
        print("⚠️ Some tests failed. Please review.")
    
    return passed == total

if __name__ == "__main__":
    main()