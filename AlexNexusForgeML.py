import logging
import numpy as np
import pandas as pd
import pickle
import warnings
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from importlib import metadata
from functools import lru_cache
import talib.abstract as ta
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

# Suppress deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True

    # Check sklearn version using modern approach
    try:
        sklearn_version = metadata.version("scikit-learn")
        logger.info(f"AlexNexusForgeML: Using scikit-learn version: {sklearn_version}")
    except Exception as e:
        logger.debug(f"Could not get sklearn version: {e}")

except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# Modern PyWavelets import
try:
    import pywt
    WAVELETS_AVAILABLE = True
    try:
        pywt_version = metadata.version("PyWavelets")
        logger.info(f"AlexNexusForgeML: Using PyWavelets version: {pywt_version}")
    except Exception as e:
        logger.debug(f"Could not get PyWavelets version: {e}")
except ImportError as e:
    logger.warning(f"PyWavelets not available: {e}")
    WAVELETS_AVAILABLE = False

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]

# =============================================================================
# UTILITY FUNCTIONS - DIRECTLY INTEGRATED
# =============================================================================

def calculate_minima_maxima(df, window):
    """Calculate minima and maxima using Heikin Ashi close."""
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):
        window_data = df['ha_close'].iloc[i - window:i + 1]
        if df['ha_close'].iloc[i] == window_data.min() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            minima[i] = -window
        if df['ha_close'].iloc[i] == window_data.max() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            maxima[i] = window

    return minima, maxima


def calc_slope_advanced(series, period):
    """
    Enhanced linear regression slope calculation with Wavelet Transform and FFT analysis
    for superior trend detection and noise filtering - ML optimized version
    """
    if len(series) < period:
        return 0

    # Use only the last 'period' values for consistency
    y = series.values[-period:]

    # Enhanced data validation
    if np.isnan(y).any() or np.isinf(y).any():
        return 0

    try:
        # === 1. WAVELET DENOISING (if available) ===
        y_denoised = y.copy()
        if WAVELETS_AVAILABLE and len(y) >= 8:
            try:
                # Adaptive wavelet selection based on signal length
                wavelet_name = 'db4' if len(y) >= 16 else 'haar'
                w = pywt.Wavelet(wavelet_name)
                max_level = min(pywt.dwt_max_level(len(y), w.dec_len), 3)
                
                coeffs = pywt.wavedec(y, wavelet_name, level=max_level, mode='symmetric')
                
                # Adaptive threshold based on signal characteristics
                sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(y))) if sigma > 0 else 0
                
                # Apply soft thresholding to detail coefficients
                coeffs_thresh = coeffs.copy()
                for i in range(1, len(coeffs_thresh)):
                    coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode='soft')
                
                y_denoised = pywt.waverec(coeffs_thresh, wavelet_name, mode='symmetric')
                
                # Ensure same length
                if len(y_denoised) != len(y):
                    y_denoised = y_denoised[:len(y)]
                    
            except Exception:
                y_denoised = y.copy()
        
        # === 2. FFT TREND ANALYSIS ===
        try:
            fft_vals = fft(y_denoised)
            freqs = fftfreq(len(y_denoised))
            
            # Filter low-frequency components (trend)
            trend_threshold = 0.1
            trend_mask = np.abs(freqs) <= trend_threshold
            fft_trend = fft_vals.copy()
            fft_trend[~trend_mask] = 0
            
            # Reconstruct trend component
            y_trend = np.real(np.fft.ifft(fft_trend))
            
            # Calculate trend frequency weight
            trend_power = np.sum(np.abs(fft_trend)**2)
            total_power = np.sum(np.abs(fft_vals)**2)
            trend_frequency_weight = trend_power / (total_power + 1e-9)
            
        except Exception:
            y_trend = y_denoised.copy()
            trend_frequency_weight = 1.0
        
        # === 3. MULTIPLE SLOPE CALCULATIONS ===
        x = np.linspace(0, period-1, period)
        
        # Original slope
        slope_original = np.polyfit(x, y, 1)[0]
        
        # Denoised slope
        slope_denoised = np.polyfit(x, y_denoised, 1)[0]
        
        # Trend slope
        slope_trend = np.polyfit(x, y_trend, 1)[0]
        
        # === 4. ROBUST SLOPE (Huber regression alternative) ===
        try:
            # Simple robust estimation using median-based approach
            x_mid = len(x) // 2
            slope_robust = (np.median(y_denoised[x_mid:]) - np.median(y_denoised[:x_mid])) / x_mid
        except Exception:
            slope_robust = slope_denoised
        
        # === 5. WEIGHTED COMBINATION ===
        # Adaptive weights based on signal characteristics
        noise_level = np.std(y - y_denoised) / (np.std(y) + 1e-9)
        
        weights = {
            'original': max(0.1, 1.0 - noise_level),
            'denoised': min(0.5, 0.3 + noise_level),
            'trend': min(0.4, 0.2 + trend_frequency_weight),
            'robust': 0.2
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Combine slopes
        slope_combined = (
            slope_original * weights['original'] +
            slope_denoised * weights['denoised'] +
            slope_trend * weights['trend'] +
            slope_robust * weights['robust']
        )
        
        # Apply frequency weighting
        final_slope = slope_combined * trend_frequency_weight
        
        # === 6. ENHANCED VALIDATION ===
        if np.isnan(final_slope) or np.isinf(final_slope):
            return (slope_original if not
                   (np.isnan(slope_original) or np.isinf(slope_original))
                   else 0)

        # Normalize extreme slopes
        max_reasonable_slope = np.std(y) / period
        if abs(final_slope) > max_reasonable_slope * 15:
            return np.sign(final_slope) * max_reasonable_slope * 15

        return final_slope

    except Exception:
        # Fallback to enhanced simple method if advanced processing fails
        try:
            # Apply simple moving average smoothing as fallback
            if len(y) >= 3:
                y_smooth = (
                    pd.Series(y).rolling(window=3, center=True)
                    .mean().bfill().ffill().values
                )
                x = np.linspace(0, period-1, period)
                slope = np.polyfit(x, y_smooth, 1)[0]

                if not (np.isnan(slope) or np.isinf(slope)):
                    return slope

            # Ultimate fallback: simple difference
            simple_slope = (y[-1] - y[0]) / (period - 1)
            return (simple_slope if not
                   (np.isnan(simple_slope) or np.isinf(simple_slope))
                   else 0)

        except Exception:
            return 0


def calculate_advanced_trend_strength_with_wavelets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced trend strength calculation using Wavelet Transform and FFT analysis
    Optimized for ML feature extraction
    """
    try:
        # === WAVELET-ENHANCED SLOPE CALCULATION ===
        dataframe['slope_5_advanced'] = dataframe['close'].rolling(5).apply(
            lambda x: calc_slope_advanced(x, 5), raw=False
        )
        dataframe['slope_10_advanced'] = dataframe['close'].rolling(10).apply(
            lambda x: calc_slope_advanced(x, 10), raw=False
        )
        dataframe['slope_20_advanced'] = dataframe['close'].rolling(20).apply(
            lambda x: calc_slope_advanced(x, 20), raw=False
        )
        
        # === WAVELET TREND DECOMPOSITION ===
        def wavelet_trend_analysis(series, window=20):
            """Analyze trend using adaptive wavelet (haar/db4), safe levels, symmetric mode, robust threshold."""
            if not WAVELETS_AVAILABLE or len(series) < window:
                return pd.Series([0.0] * len(series), index=series.index)
            
            results: list[float] = []
            for i in range(len(series)):
                if i < window:
                    results.append(0.0)
                    continue
                
                window_data = series.iloc[i-window+1:i+1].values
                n = len(window_data)
                if n < 12:
                    results.append(0.0)
                    continue
                
                wavelet_name = 'haar' if n < 24 else 'db4'
                try:
                    w = pywt.Wavelet(wavelet_name)
                    max_level = pywt.dwt_max_level(n, w.dec_len)
                except Exception:
                    max_level = 1
                
                if n < 48:
                    max_level = min(max_level, 2)
                use_level = max(1, min(3, max_level))
                
                try:
                    coeffs = pywt.wavedec(window_data, wavelet_name, level=use_level, mode='symmetric')
                    
                    # Estimate sigma from finest detail
                    if len(coeffs) > 1 and len(coeffs[-1]):
                        detail = coeffs[-1]
                        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
                        thr = sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
                    else:
                        thr = 0.0
                    
                    for j in range(1, len(coeffs)):
                        coeffs[j] = pywt.threshold(coeffs[j], thr, mode='soft')
                    
                    approx = coeffs[0]
                    trend_strength = np.std(approx) / (np.std(window_data) + 1e-9)
                    
                    direction = 0
                    if len(approx) >= 2:
                        direction = 1 if approx[-1] > approx[0] else -1
                    
                    results.append(trend_strength * direction)
                    
                except Exception:
                    results.append(0.0)
            
            return pd.Series(results, index=series.index)

        # === APPLY WAVELET ANALYSIS ===
        dataframe['wavelet_trend'] = wavelet_trend_analysis(dataframe['close'], window=20)
        dataframe['wavelet_trend_short'] = wavelet_trend_analysis(dataframe['close'], window=10)
        dataframe['wavelet_trend_long'] = wavelet_trend_analysis(dataframe['close'], window=40)
        
        # === FFT FREQUENCY ANALYSIS ===
        def fft_frequency_analysis(series, window=30):
            """Analyze dominant frequencies using FFT for trend identification."""
            results = []
            for i in range(len(series)):
                if i < window:
                    results.append(0.0)
                    continue
                
                window_data = series.iloc[i-window+1:i+1].values
                
                try:
                    # Apply FFT
                    fft_vals = fft(window_data)
                    freqs = fftfreq(window, d=1.0)
                    
                    # Analyze low-frequency components (trend)
                    low_freq_mask = np.abs(freqs) <= 0.1
                    low_freq_power = np.sum(np.abs(fft_vals[low_freq_mask])**2)
                    total_power = np.sum(np.abs(fft_vals)**2)
                    
                    trend_ratio = low_freq_power / (total_power + 1e-9)
                    
                    # Determine trend direction from phase
                    if low_freq_power > 0:
                        low_freq_components = fft_vals[low_freq_mask]
                        avg_phase = np.angle(np.mean(low_freq_components))
                        direction = 1 if avg_phase > 0 else -1
                    else:
                        direction = 0
                    
                    results.append(trend_ratio * direction)
                    
                except Exception:
                    results.append(0.0)
            
            return pd.Series(results, index=series.index)

        dataframe['fft_trend'] = fft_frequency_analysis(dataframe['close'])
        
        # === COMPOSITE TREND STRENGTH ===
        # Combine all trend measures
        dataframe['trend_strength_composite'] = (
            dataframe['slope_20_advanced'] * 0.3 +
            dataframe['wavelet_trend'] * 0.3 +
            dataframe['fft_trend'] * 0.2 +
            dataframe['slope_10_advanced'] * 0.2
        )
        
        # === TREND QUALITY METRICS ===
        # Consistency across timeframes
        dataframe['trend_consistency'] = (
            (np.sign(dataframe['slope_5_advanced']) == np.sign(dataframe['slope_10_advanced'])).astype(int) +
            (np.sign(dataframe['slope_10_advanced']) == np.sign(dataframe['slope_20_advanced'])).astype(int) +
            (np.sign(dataframe['wavelet_trend']) == np.sign(dataframe['slope_20_advanced'])).astype(int)
        )
        
        # Trend acceleration
        dataframe['trend_acceleration'] = dataframe['slope_5_advanced'] - dataframe['slope_20_advanced']
        
        # Fill NaN values
        columns_to_fill = [
            'slope_5_advanced', 'slope_10_advanced', 'slope_20_advanced',
            'wavelet_trend', 'wavelet_trend_short', 'wavelet_trend_long',
            'fft_trend', 'trend_strength_composite', 'trend_consistency',
            'trend_acceleration'
        ]
        
        for col in columns_to_fill:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].fillna(0)
        
        return dataframe
        
    except Exception as e:
        logger.warning(f"Error in calculate_advanced_trend_strength_with_wavelets: {e}")
        # Return basic trend calculation as fallback
        dataframe['slope_5_advanced'] = 0
        dataframe['slope_10_advanced'] = 0  
        dataframe['slope_20_advanced'] = 0
        dataframe['wavelet_trend'] = 0
        dataframe['wavelet_trend_short'] = 0
        dataframe['wavelet_trend_long'] = 0
        dataframe['fft_trend'] = 0
        dataframe['trend_strength_composite'] = 0
        dataframe['trend_consistency'] = 0
        dataframe['trend_acceleration'] = 0
        return dataframe


# =============================================================================
# ML ENGINE - ENHANCED FOR FULL CONTROL
# =============================================================================

class MLTradingEngine:
    """
    Advanced machine learning engine for full trading control
    Takes complete responsibility for entry and exit decisions
    """
    
    def __init__(self, strategy_name="AlexNexusForgeML"):
        self.strategy_name = strategy_name
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_performance = {}
        self.last_predictions = {}
        self.prediction_confidence = {}
        self.feature_importance = {}
        
        # ML Configuration for full control
        self.min_training_samples = 500
        self.retrain_frequency = 100  # Retrain every N candles
        self.confidence_threshold = 0.6  # Minimum confidence for signals
        self.ensemble_voting = 'soft'  # Use probability voting
        
        # Model ensemble for robust predictions
        self.model_configs = {
            'rf_entry': {
                'model': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=-1
                ),
                'purpose': 'entry_long'
            },
            'rf_exit': {
                'model': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=-1
                ),
                'purpose': 'exit_long'
            },
            'gb_entry': {
                'model': GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1,
                    subsample=0.8, random_state=42
                ),
                'purpose': 'entry_long'
            },
            'gb_exit': {
                'model': GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1,
                    subsample=0.8, random_state=42
                ),
                'purpose': 'exit_long'
            }
        }
        
        logger.info(f"ML Trading Engine initialized for {strategy_name}")
    
    def extract_ml_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for ML prediction with advanced indicators
        Optimized for full trading control
        """
        try:
            if dataframe is None or dataframe.empty:
                return dataframe
            
            # === PRICE ACTION FEATURES ===
            dataframe['price_change'] = dataframe['close'].pct_change()
            dataframe['price_change_2'] = dataframe['close'].pct_change(2)
            dataframe['price_change_5'] = dataframe['close'].pct_change(5)
            dataframe['price_volatility'] = dataframe['price_change'].rolling(20).std()
            
            # === VOLUME FEATURES ===
            dataframe['volume_change'] = dataframe['volume'].pct_change()
            dataframe['volume_ma'] = dataframe['volume'].rolling(20).mean()
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
            dataframe['volume_spike'] = (dataframe['volume_ratio'] > 2).astype(int)
            
            # === MOMENTUM FEATURES ===
            try:
                dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
            except Exception:
                # Fallback RSI calculation
                delta = dataframe['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                dataframe['rsi'] = 100 - (100 / (1 + rs))
            
            dataframe['rsi_change'] = dataframe['rsi'].diff()
            dataframe['rsi_oversold'] = (dataframe['rsi'] < 30).astype(int)
            dataframe['rsi_overbought'] = (dataframe['rsi'] > 70).astype(int)
            
            # MACD
            try:
                macd_result = ta.MACD(dataframe['close'])
                dataframe['macd'] = macd_result['macd']
                dataframe['macd_signal'] = macd_result['macdsignal']
                dataframe['macd_histogram'] = macd_result['macdhist']
                dataframe['macd_bullish'] = (dataframe['macd'] > dataframe['macd_signal']).astype(int)
            except Exception as e:
                # Fallback MACD calculation
                exp1 = dataframe['close'].ewm(span=12).mean()
                exp2 = dataframe['close'].ewm(span=26).mean()
                dataframe['macd'] = exp1 - exp2
                dataframe['macd_signal'] = dataframe['macd'].ewm(span=9).mean()
                dataframe['macd_histogram'] = dataframe['macd'] - dataframe['macd_signal']
                dataframe['macd_bullish'] = (dataframe['macd'] > dataframe['macd_signal']).astype(int)
            
            # === TREND FEATURES ===
            # Enhanced trend analysis using our utility functions
            dataframe = calculate_advanced_trend_strength_with_wavelets(dataframe)
            
            # Moving averages
            try:
                dataframe['ema_9'] = ta.EMA(dataframe['close'], timeperiod=9)
                dataframe['ema_21'] = ta.EMA(dataframe['close'], timeperiod=21)
                dataframe['ema_50'] = ta.EMA(dataframe['close'], timeperiod=50)
                dataframe['sma_200'] = ta.SMA(dataframe['close'], timeperiod=200)
            except Exception:
                # Fallback moving average calculations
                dataframe['ema_9'] = dataframe['close'].ewm(span=9).mean()
                dataframe['ema_21'] = dataframe['close'].ewm(span=21).mean()
                dataframe['ema_50'] = dataframe['close'].ewm(span=50).mean()
                dataframe['sma_200'] = dataframe['close'].rolling(200).mean()
            
            # MA relationships
            dataframe['price_above_ema9'] = (dataframe['close'] > dataframe['ema_9']).astype(int)
            dataframe['price_above_ema21'] = (dataframe['close'] > dataframe['ema_21']).astype(int)
            dataframe['price_above_ema50'] = (dataframe['close'] > dataframe['ema_50']).astype(int)
            dataframe['price_above_sma200'] = (dataframe['close'] > dataframe['sma_200']).astype(int)
            
            # === VOLATILITY FEATURES ===
            try:
                dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
            except Exception:
                # Fallback ATR calculation
                high_low = dataframe['high'] - dataframe['low']
                high_close = np.abs(dataframe['high'] - dataframe['close'].shift())
                low_close = np.abs(dataframe['low'] - dataframe['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                dataframe['atr'] = true_range.rolling(14).mean()
            
            dataframe['atr_ratio'] = dataframe['atr'] / dataframe['close']
            
            # Bollinger Bands
            try:
                bb_result = ta.BBANDS(dataframe['close'])
                dataframe['bb_upper'] = bb_result['upperband']
                dataframe['bb_middle'] = bb_result['middleband']
                dataframe['bb_lower'] = bb_result['lowerband']
            except Exception:
                # Fallback Bollinger Bands calculation
                dataframe['bb_middle'] = dataframe['close'].rolling(20).mean()
                bb_std = dataframe['close'].rolling(20).std()
                dataframe['bb_upper'] = dataframe['bb_middle'] + (bb_std * 2)
                dataframe['bb_lower'] = dataframe['bb_middle'] - (bb_std * 2)
            
            dataframe['bb_position'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
            
            # === SUPPORT/RESISTANCE FEATURES ===
            # Local highs and lows
            dataframe['local_high'] = dataframe['high'].rolling(10, center=True).max() == dataframe['high']
            dataframe['local_low'] = dataframe['low'].rolling(10, center=True).min() == dataframe['low']
            
            # Distance to recent high/low
            dataframe['high_20'] = dataframe['high'].rolling(20).max()
            dataframe['low_20'] = dataframe['low'].rolling(20).min()
            dataframe['distance_to_high'] = (dataframe['close'] - dataframe['high_20']) / dataframe['high_20']
            dataframe['distance_to_low'] = (dataframe['close'] - dataframe['low_20']) / dataframe['low_20']
            
            # === PATTERN FEATURES ===
            # Candlestick patterns (basic)
            dataframe['doji'] = (abs(dataframe['open'] - dataframe['close']) / (dataframe['high'] - dataframe['low'] + 1e-9) < 0.1).astype(int)
            dataframe['hammer'] = ((dataframe['low'] < dataframe[['open', 'close']].min(axis=1)) & 
                                  (dataframe['high'] - dataframe[['open', 'close']].max(axis=1) < 
                                   (dataframe[['open', 'close']].max(axis=1) - dataframe['low']) * 0.3)).astype(int)
            
            # === TIME FEATURES ===
            dataframe['hour'] = dataframe.index.hour if hasattr(dataframe.index, 'hour') else 0
            dataframe['day_of_week'] = dataframe.index.dayofweek if hasattr(dataframe.index, 'dayofweek') else 0
            
            # === COMPOSITE FEATURES ===
            # Momentum composite
            dataframe['momentum_composite'] = (
                dataframe['rsi'] / 100 * 0.3 +
                dataframe['macd_bullish'] * 0.3 +
                dataframe['price_above_ema21'] * 0.4
            )
            
            # Trend composite
            dataframe['trend_composite'] = (
                dataframe['trend_strength_composite'] * 0.4 +
                dataframe['price_above_ema50'] * 0.3 +
                dataframe['trend_consistency'] / 3 * 0.3
            )
            
            # Volatility composite
            dataframe['volatility_composite'] = (
                dataframe['atr_ratio'] * 0.5 +
                dataframe['price_volatility'] * 0.5
            )
            
            # === FUTURE PREDICTION FEATURES ===
            # These help ML learn to predict future movements
            dataframe['future_return_1'] = dataframe['close'].shift(-1) / dataframe['close'] - 1
            dataframe['future_return_3'] = dataframe['close'].shift(-3) / dataframe['close'] - 1
            dataframe['future_return_5'] = dataframe['close'].shift(-5) / dataframe['close'] - 1
            
            # High/low touch in future (for exit prediction)
            dataframe['future_high_3'] = dataframe['high'].shift(-1).rolling(3).max()
            dataframe['future_low_3'] = dataframe['low'].shift(-1).rolling(3).min()
            
            # === CLEAN DATA ===
            # Fill NaN values
            dataframe = dataframe.ffill().fillna(0)
            
            # Remove infinite values
            dataframe = dataframe.replace([np.inf, -np.inf], 0)
            
            logger.info(f"ML feature extraction completed. Shape: {dataframe.shape}")
            return dataframe
            
        except Exception as e:
            logger.error(f"Error in ML feature extraction: {e}")
            return dataframe
    
    def create_training_labels(self, dataframe: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create training labels for ML models
        Entry and exit signals based on future profitability
        """
        try:
            labels = {}
            
            # === ENTRY LABELS ===
            # Define profitable entry as achieving >1% gain within 5 candles
            future_max = dataframe['high'].shift(-1).rolling(5).max()
            entry_profit = (future_max / dataframe['close'] - 1)
            labels['entry_long'] = (entry_profit > 0.01).astype(int)
            
            # More conservative entry (>2% gain within 10 candles)
            future_max_long = dataframe['high'].shift(-1).rolling(10).max()
            entry_profit_long = (future_max_long / dataframe['close'] - 1)
            labels['entry_long_conservative'] = (entry_profit_long > 0.02).astype(int)
            
            # === EXIT LABELS ===
            # Define exit signal as risk of >-1% loss within 3 candles
            future_min = dataframe['low'].shift(-1).rolling(3).min()
            exit_risk = (future_min / dataframe['close'] - 1)
            labels['exit_long'] = (exit_risk < -0.01).astype(int)
            
            # Alternative exit: Take profit opportunity (>3% gain achieved)
            labels['exit_profit'] = (entry_profit > 0.03).astype(int)
            
            # === SHORT LABELS (if strategy supports shorting) ===
            labels['entry_short'] = (entry_profit < -0.01).astype(int)
            labels['exit_short'] = (exit_risk > 0.01).astype(int)
            
            logger.info(f"Training labels created: {list(labels.keys())}")
            return labels
            
        except Exception as e:
            logger.error(f"Error creating training labels: {e}")
            return {}
    
    def get_feature_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns for ML training
        Excludes target variables and non-predictive columns
        """
        exclude_columns = [
            'open', 'high', 'low', 'close', 'volume',  # Basic OHLCV
            'date', 'timestamp',  # Time columns
            'future_return_1', 'future_return_3', 'future_return_5',  # Future data
            'future_high_3', 'future_low_3',  # Future data
            'enter_long', 'enter_short', 'exit_long', 'exit_short',  # Target columns
        ]
        
        feature_columns = [col for col in dataframe.columns 
                          if col not in exclude_columns and 
                          not col.startswith('entry_') and 
                          not col.startswith('exit_')]
        
        logger.info(f"Selected {len(feature_columns)} feature columns for ML")
        return feature_columns
    
    def train_models(self, dataframe: pd.DataFrame, pair: str) -> Dict[str, float]:
        """
        Train ML models for the given pair
        Returns performance metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, cannot train ML models")
            return {}
        
        try:
            # Extract features and create labels
            dataframe = self.extract_ml_features(dataframe)
            labels = self.create_training_labels(dataframe)
            
            if not labels:
                logger.warning(f"No labels created for {pair}")
                return {}
            
            feature_columns = self.get_feature_columns(dataframe)
            X = dataframe[feature_columns].values
            
            # Check data quality
            if len(X) < self.min_training_samples:
                logger.warning(f"Insufficient data for training {pair}: {len(X)} samples")
                return {}
            
            performance = {}
            
            # Train models for each purpose
            for model_name, config in self.model_configs.items():
                purpose = config['purpose']
                
                if purpose not in labels:
                    continue
                
                y = labels[purpose].values
                
                # Skip if no positive examples
                if y.sum() < 10:
                    logger.warning(f"Insufficient positive examples for {model_name} on {pair}: {y.sum()}")
                    continue
                
                try:
                    # Feature selection
                    selector = SelectKBest(f_classif, k=min(50, len(feature_columns)))
                    X_selected = selector.fit_transform(X, y)
                    
                    # Scale features
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X_selected)
                    
                    # Train model
                    model = config['model']
                    model.fit(X_scaled, y)
                    
                    # Store components
                    model_key = f"{pair}_{model_name}"
                    self.models[model_key] = model
                    self.scalers[model_key] = scaler
                    self.feature_selectors[model_key] = selector
                    
                    # Calculate performance
                    y_pred = model.predict(X_scaled)
                    y_pred_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    accuracy = accuracy_score(y, y_pred)
                    precision = precision_score(y, y_pred, zero_division=0)
                    recall = recall_score(y, y_pred, zero_division=0)
                    f1 = f1_score(y, y_pred, zero_division=0)
                    
                    performance[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'positive_rate': y.mean()
                    }
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
                        importance_dict = dict(zip(selected_features, model.feature_importances_))
                        self.feature_importance[model_key] = importance_dict
                    
                    logger.info(f"Trained {model_name} for {pair}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {pair}: {e}")
                    continue
            
            self.model_performance[pair] = performance
            return performance
            
        except Exception as e:
            logger.error(f"Error in model training for {pair}: {e}")
            return {}
    
    def predict_signals(self, dataframe: pd.DataFrame, pair: str) -> Dict[str, float]:
        """
        Generate ML predictions for entry and exit signals
        Returns probabilities and confidence scores
        """
        if not SKLEARN_AVAILABLE or dataframe.empty:
            return {
                'entry_long_prob': 0.0,
                'exit_long_prob': 0.0,
                'entry_confidence': 0.0,
                'exit_confidence': 0.0,
                'ml_signal': 'no_signal'
            }
        
        try:
            # Extract features for prediction
            dataframe = self.extract_ml_features(dataframe)
            feature_columns = self.get_feature_columns(dataframe)
            
            # Get the latest data point
            latest_features = dataframe[feature_columns].iloc[-1:].values
            
            predictions = {
                'entry_long_prob': 0.0,
                'exit_long_prob': 0.0,
                'entry_confidence': 0.0,
                'exit_confidence': 0.0,
                'ml_signal': 'no_signal'
            }
            
            entry_predictions = []
            exit_predictions = []
            
            # Get predictions from all models
            for model_name, config in self.model_configs.items():
                model_key = f"{pair}_{model_name}"
                
                if model_key not in self.models:
                    continue
                
                try:
                    # Transform features
                    selector = self.feature_selectors[model_key]
                    scaler = self.scalers[model_key]
                    model = self.models[model_key]
                    
                    features_selected = selector.transform(latest_features)
                    features_scaled = scaler.transform(features_selected)
                    
                    # Get prediction probability
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(features_scaled)[0, 1]
                    else:
                        prob = model.predict(features_scaled)[0]
                    
                    # Store prediction
                    purpose = config['purpose']
                    if 'entry' in purpose:
                        entry_predictions.append(prob)
                    elif 'exit' in purpose:
                        exit_predictions.append(prob)
                    
                except Exception as e:
                    logger.debug(f"Error predicting with {model_name} for {pair}: {e}")
                    continue
            
            # Ensemble predictions
            if entry_predictions:
                predictions['entry_long_prob'] = np.mean(entry_predictions)
                predictions['entry_confidence'] = 1.0 - np.std(entry_predictions)  # Lower std = higher confidence
            
            if exit_predictions:
                predictions['exit_long_prob'] = np.mean(exit_predictions)
                predictions['exit_confidence'] = 1.0 - np.std(exit_predictions)
            
            # Generate signal based on probabilities and confidence
            if (predictions['entry_long_prob'] > self.confidence_threshold and 
                predictions['entry_confidence'] > 0.5):
                predictions['ml_signal'] = 'entry_long'
            elif (predictions['exit_long_prob'] > self.confidence_threshold and 
                  predictions['exit_confidence'] > 0.5):
                predictions['ml_signal'] = 'exit_long'
            
            # Store for logging
            self.last_predictions[pair] = predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in ML prediction for {pair}: {e}")
            return predictions
    
    def should_retrain(self, pair: str, current_candle: int) -> bool:
        """
        Determine if models should be retrained
        """
        # Check if we have models for this pair
        pair_models = [k for k in self.models.keys() if k.startswith(f"{pair}_")]
        
        if not pair_models:
            return True  # No models yet, need to train
        
        # Retrain every N candles
        if current_candle % self.retrain_frequency == 0:
            return True
        
        return False
    
    def log_ml_decision(self, pair: str, signal_type: str, dataframe: pd.DataFrame):
        """
        Enhanced logging for ML decisions with probabilities and market conditions
        """
        try:
            if pair not in self.last_predictions:
                return
            
            pred = self.last_predictions[pair]
            latest_data = dataframe.iloc[-1]
            
            # Log ML prediction details
            logger.info(f"🤖 ML DECISION for {pair} - {signal_type}")
            logger.info(f"  Entry Probability: {pred['entry_long_prob']:.3f} (Confidence: {pred['entry_confidence']:.3f})")
            logger.info(f"  Exit Probability: {pred['exit_long_prob']:.3f} (Confidence: {pred['exit_confidence']:.3f})")
            logger.info(f"  ML Signal: {pred['ml_signal']}")
            
            # Market conditions
            logger.info(f"  Volume: {latest_data.get('volume', 0):.0f} (Ratio: {latest_data.get('volume_ratio', 0):.2f})")
            logger.info(f"  RSI: {latest_data.get('rsi', 0):.1f}")
            logger.info(f"  Trend Strength: {latest_data.get('trend_strength_composite', 0):.3f}")
            logger.info(f"  Trend Consistency: {latest_data.get('trend_consistency', 0):.0f}/3")
            
            # Future predictions (if available)
            if 'future_return_1' in latest_data:
                logger.info(f"  Predicted 1-candle return: {latest_data.get('future_return_1', 0):.3f}")
                logger.info(f"  Predicted 3-candle return: {latest_data.get('future_return_3', 0):.3f}")
            
        except Exception as e:
            logger.debug(f"Error in ML logging for {pair}: {e}")


# =============================================================================
# END OF PART 1
# =============================================================================

# This concludes Part 1 of the AlexNexusForgeML implementation
# Part 1 includes:
# - All utility functions directly integrated
# - Enhanced ML engine for full trading control  
# - Comprehensive feature extraction
# - Advanced model training and prediction
# - ML-focused logging system

# Part 2 will complete the implementation with:
# - Full IStrategy class implementation
# - ML-driven populate_entry_trend and populate_exit_trend methods
# - Integration with freqtrade framework
# - Final testing and validation

logger.info("AlexNexusForgeML Part 1 loaded successfully - ML Engine and Utilities ready")