# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import logging
import numpy as np
import pandas as pd
import pickle
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple
from importlib import metadata
from functools import lru_cache
import talib.abstract as ta
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
# Mock qtpylib for testing - will be replaced in production
try:
    import freqtrade.vendor.qtpylib.indicators as qtpylib
except ImportError:
    # Create mock qtpylib for testing
    class MockQtpylib:
        @staticmethod
        def typical_price(df):
            return (df['high'] + df['low'] + df['close']) / 3
    qtpylib = MockQtpylib()
# Mock FreqTrade for testing - will be replaced in production
try:
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
    from freqtrade.persistence import Trade
except ImportError:
    # Create mock FreqTrade classes for testing
    class IStrategy:
        INTERFACE_VERSION = 3
        timeframe = '5m'
        can_short = False
        stoploss = -0.10
        startup_candle_count = 200
        minimal_roi = {"0": 0.15}
        trailing_stop = True
        
        def __init__(self, config):
            self.config = config
        
        def informative_pairs(self):
            return []
    
    class DecimalParameter:
        def __init__(self, low, high, default, optimize=True, space='buy'):
            self.value = default
    
    class IntParameter:
        def __init__(self, low, high, default, optimize=True, space='buy'):
            self.value = default
    
    class BooleanParameter:
        def __init__(self, default, optimize=True, space='buy'):
            self.value = default
    
    class Trade:
        def __init__(self, open_rate=100.0, open_date=None):
            self.open_rate = open_rate
            self.open_date_utc = open_date or datetime.now()
        
        def calc_profit_ratio(self, rate):
            return (rate - self.open_rate) / self.open_rate

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

# ML Dependencies
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
    try:
        sklearn_version = metadata.version("scikit-learn")
        logger.info(f"Using scikit-learn version: {sklearn_version}")
    except Exception as e:
        logger.debug(f"Could not get sklearn version: {e}")
except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# PyWavelets for advanced analysis
try:
    import pywt
    WAVELETS_AVAILABLE = True
    try:
        pywt_version = metadata.version("PyWavelets")
        logger.info(f"Using PyWavelets version: {pywt_version}")
    except Exception as e:
        logger.debug(f"Could not get PyWavelets version: {e}")
except ImportError as e:
    logger.warning(f"PyWavelets not available: {e}")
    WAVELETS_AVAILABLE = False


class MLPredictiveEngine:
    """
    Advanced Machine Learning Engine for Trade Prediction
    Primary decision maker for entry/exit signals
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = {}
        self.training_history = {}
        self.is_trained = {}
        
        # Model persistence
        self.models_dir = Path("user_data/strategies/ml_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training control
        self.min_training_samples = 200
        self.retrain_interval_hours = 12
        self.last_train_time = {}
        
        # Prediction thresholds (optimized for precision)
        self.entry_threshold_high = 0.85    # High confidence entries
        self.entry_threshold_medium = 0.75  # Medium confidence entries
        self.entry_threshold_low = 0.65     # Low confidence entries
        self.exit_threshold = 0.70          # Exit threshold
        
        logger.info("🤖 ML Predictive Engine initialized")
    
    def extract_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive ML features for prediction
        """
        try:
            # === PRICE ACTION FEATURES ===
            # Multi-timeframe returns
            for period in [1, 2, 3, 5, 8, 13, 21]:
                df[f'return_{period}'] = df['close'].pct_change(period)
                df[f'return_volatility_{period}'] = df[f'return_{period}'].rolling(10).std()
                df[f'return_momentum_{period}'] = df[f'return_{period}'].rolling(5).mean()
            
            # Price position features
            for window in [10, 20, 50]:
                high_window = df['high'].rolling(window).max()
                low_window = df['low'].rolling(window).min()
                range_size = high_window - low_window
                df[f'price_position_{window}'] = (df['close'] - low_window) / (range_size + 1e-8)
                df[f'price_range_pct_{window}'] = range_size / df['close']
            
            # === TECHNICAL INDICATORS ===
            # RSI variants
            df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
            df['rsi_21'] = ta.RSI(df['close'], timeperiod=21)
            df['rsi_slope'] = df['rsi_14'].diff(3)
            df['rsi_divergence'] = (df['close'].diff(5) * df['rsi_14'].diff(5)) < 0
            
            # MACD
            macd, macdsignal, macdhist = ta.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macdsignal
            df['macd_histogram'] = macdhist
            df['macd_slope'] = macd.diff(3)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
            
            # === VOLUME FEATURES ===
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            
            # OBV and derivatives
            df['obv'] = ta.OBV(df['close'], df['volume'])
            df['obv_sma'] = df['obv'].rolling(20).mean()
            df['obv_slope'] = df['obv'].diff(5)
            
            # === VOLATILITY FEATURES ===
            # ATR
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'])
            df['atr_pct'] = df['atr'] / df['close']
            df['atr_ratio'] = df['atr'] / df['atr'].rolling(20).mean()
            
            # Realized volatility
            returns = df['close'].pct_change()
            for window in [10, 20, 50]:
                df[f'volatility_{window}'] = returns.rolling(window).std()
                df[f'volatility_rank_{window}'] = df[f'volatility_{window}'].rolling(100).rank(pct=True)
            
            # === MOMENTUM FEATURES ===
            # Multiple momentum indicators
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = ta.MOM(df['close'], timeperiod=period)
                df[f'roc_{period}'] = ta.ROC(df['close'], timeperiod=period)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
            df['stoch_slope'] = df['stoch_k'].diff(3)
            
            # === TREND FEATURES ===
            # Multiple EMAs
            for period in [9, 21, 50, 200]:
                df[f'ema_{period}'] = ta.EMA(df['close'], timeperiod=period)
                df[f'ema_distance_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
            
            # EMA slopes
            df['ema_slope_fast'] = df['ema_9'].diff(3)
            df['ema_slope_slow'] = df['ema_21'].diff(5)
            
            # Trend strength
            df['trend_strength'] = (df['ema_9'] - df['ema_50']) / df['ema_50']
            
            # === STATISTICAL FEATURES ===
            # Rolling statistics
            for window in [10, 20, 50]:
                df[f'skewness_{window}'] = returns.rolling(window).skew()
                df[f'kurtosis_{window}'] = returns.rolling(window).kurt()
                df[f'entropy_{window}'] = self._calculate_entropy(df['close'], window)
            
            # === ADVANCED PATTERN FEATURES ===
            # Candle patterns
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            df['candle_range'] = (df['high'] - df['low']) / df['close']
            
            # Consecutive patterns
            df['green_candle'] = (df['close'] > df['open']).astype(int)
            df['red_candle'] = (df['close'] < df['open']).astype(int)
            df['consecutive_green'] = df['green_candle'].rolling(5).sum()
            df['consecutive_red'] = df['red_candle'].rolling(5).sum()
            
            # === TIME-BASED FEATURES ===
            if 'date' in df.columns:
                df['hour'] = pd.to_datetime(df['date']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            logger.info(f"🔧 Extracted {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'date']])} ML features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return df
    
    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling entropy"""
        def entropy(data):
            if len(data) < 5:
                return 0
            returns = np.diff(data) / (data[:-1] + 1e-10)
            hist, _ = np.histogram(returns, bins=10)
            probs = hist / (hist.sum() + 1e-10)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        return series.rolling(window).apply(entropy, raw=False)
    
    def create_target_labels(self, df: pd.DataFrame, forward_periods: int = 6, 
                           profit_threshold: float = 0.02) -> pd.Series:
        """
        Create target labels for ML training
        Uses forward-looking profit potential
        """
        # Calculate forward returns
        forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
        
        # Calculate maximum profit potential in forward window
        forward_highs = df['high'].rolling(forward_periods).max().shift(-forward_periods)
        max_profit = (forward_highs - df['close']) / df['close']
        
        # Calculate maximum loss potential
        forward_lows = df['low'].rolling(forward_periods).min().shift(-forward_periods)
        max_loss = (df['close'] - forward_lows) / df['close']
        
        # Risk-adjusted target
        risk_reward = max_profit / (max_loss + 1e-8)
        
        # Create labels based on multiple criteria
        target = (
            (forward_returns > profit_threshold) &  # Basic profit requirement
            (max_profit > profit_threshold * 1.2) &  # Higher max profit potential
            (risk_reward > 1.5) &  # Good risk-reward ratio
            (max_loss < profit_threshold * 0.8)  # Limited downside risk
        ).astype(int)
        
        positive_ratio = target.mean()
        logger.info(f"📊 Target labels created: {target.sum()}/{len(target)} positive ({positive_ratio:.2%})")
        
        return target
    
    def train_models(self, df: pd.DataFrame, pair: str) -> Dict:
        """
        Train ensemble of ML models for the pair
        """
        if not SKLEARN_AVAILABLE:
            logger.error("❌ scikit-learn not available for ML training")
            return {'status': 'sklearn_unavailable'}
        
        try:
            # Extract features and create targets
            feature_df = self.extract_ml_features(df.copy())
            target = self.create_target_labels(df.copy())
            
            # Select feature columns
            feature_columns = []
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'enter_long', 'enter_short', 'exit_long', 'exit_short']
            
            for col in feature_df.columns:
                if (col not in exclude_cols and 
                    feature_df[col].dtype in ['float64', 'int64'] and
                    not col.startswith('enter_') and
                    not col.startswith('exit_')):
                    feature_columns.append(col)
            
            if len(feature_columns) < 10:
                logger.warning(f"⚠️ Insufficient features ({len(feature_columns)}) for {pair}")
                return {'status': 'insufficient_features'}
            
            # Prepare data
            X = feature_df[feature_columns].fillna(0)
            y = target.fillna(0)
            
            # Remove invalid samples
            valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < self.min_training_samples:
                logger.warning(f"⚠️ Insufficient training data ({len(X)}) for {pair}")
                return {'status': 'insufficient_data'}
            
            # Remove constant features
            feature_variance = X.var()
            variable_features = feature_variance[feature_variance > 1e-10].index.tolist()
            X = X[variable_features]
            feature_columns = variable_features
            
            # Check class balance
            positive_count = y.sum()
            if positive_count < 20:
                logger.warning(f"⚠️ Too few positive samples ({positive_count}) for {pair}")
                return {'status': 'insufficient_positive_samples'}
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble models
            models = {}
            results = {}
            
            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            models['random_forest'] = rf
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
            gb.fit(X_train_scaled, y_train)
            models['gradient_boosting'] = gb
            
            # Extra Trees
            et = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=25,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            et.fit(X_train_scaled, y_train)
            models['extra_trees'] = et
            
            # Evaluate models
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                logger.info(f"🎯 {name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Store models and results
            self.models[pair] = models
            self.scalers[pair] = scaler
            self.feature_importance[pair] = {
                'features': feature_columns,
                'rf_importance': rf.feature_importances_ if 'random_forest' in models else None
            }
            self.training_history[pair] = {
                'timestamp': datetime.now(),
                'results': results,
                'samples': len(X_train),
                'features': len(feature_columns)
            }
            self.is_trained[pair] = True
            self.last_train_time[pair] = datetime.now()
            
            logger.info(f"✅ ML models trained successfully for {pair}")
            logger.info(f"📈 Training samples: {len(X_train)}, Features: {len(feature_columns)}")
            
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"❌ ML training failed for {pair}: {e}")
            return {'status': 'training_failed', 'error': str(e)}
    
    def predict(self, df: pd.DataFrame, pair: str) -> Dict:
        """
        Generate ML predictions for entry/exit signals
        """
        if pair not in self.models or not self.is_trained.get(pair, False):
            return {
                'entry_probability': 0.5,
                'exit_probability': 0.5,
                'confidence': 0.0,
                'model_agreement': 0.0,
                'prediction_6h': 0.5,
                'status': 'not_trained'
            }
        
        try:
            # Extract features
            feature_df = self.extract_ml_features(df.copy())
            feature_columns = self.feature_importance[pair]['features']
            
            # Prepare current sample
            X_current = feature_df[feature_columns].fillna(0).iloc[-1:].values
            X_scaled = self.scalers[pair].transform(X_current)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models[pair].items():
                try:
                    pred_proba = model.predict_proba(X_scaled)[0, 1]
                    pred = 1 if pred_proba > 0.5 else 0
                    
                    predictions[name] = pred
                    probabilities[name] = pred_proba
                except Exception as e:
                    logger.warning(f"⚠️ Prediction failed for {name}: {e}")
                    predictions[name] = 0
                    probabilities[name] = 0.5
            
            # Calculate ensemble metrics
            avg_probability = np.mean(list(probabilities.values()))
            model_agreement = len([p for p in predictions.values() if p == 1]) / len(predictions)
            confidence = np.std(list(probabilities.values()))  # Lower std = higher confidence
            confidence = 1 - min(confidence * 2, 1)  # Invert and normalize
            
            # 6-hour prediction (simplified - uses current probability as proxy)
            prediction_6h = avg_probability
            
            # Store prediction history
            if pair not in self.prediction_history:
                self.prediction_history[pair] = []
            
            self.prediction_history[pair].append({
                'timestamp': datetime.now(),
                'probability': avg_probability,
                'confidence': confidence,
                'agreement': model_agreement
            })
            
            # Keep only recent history
            if len(self.prediction_history[pair]) > 100:
                self.prediction_history[pair] = self.prediction_history[pair][-100:]
            
            return {
                'entry_probability': avg_probability,
                'exit_probability': 1 - avg_probability,  # Inverse for exit
                'confidence': confidence,
                'model_agreement': model_agreement,
                'prediction_6h': prediction_6h,
                'individual_predictions': probabilities,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {pair}: {e}")
            return {
                'entry_probability': 0.5,
                'exit_probability': 0.5,
                'confidence': 0.0,
                'model_agreement': 0.0,
                'prediction_6h': 0.5,
                'status': 'prediction_failed'
            }


class AlexNexusForgeML(IStrategy):
    """
    ALEX NEXUS FORGE ML - FULLY MACHINE LEARNING-DRIVEN STRATEGY
    
    🤖 Primary Decision Maker: Machine Learning Models
    📊 Secondary Support: Traditional Technical Indicators
    🎯 Focus: High-precision entries with comprehensive ML analysis
    """

    INTERFACE_VERSION = 3
    
    # === STRATEGY PARAMETERS ===
    
    # Core ML thresholds
    ml_entry_threshold_high = DecimalParameter(0.80, 0.95, default=0.85, optimize=True, space='buy')
    ml_entry_threshold_medium = DecimalParameter(0.70, 0.85, default=0.75, optimize=True, space='buy')
    ml_confidence_threshold = DecimalParameter(0.60, 0.90, default=0.70, optimize=True, space='buy')
    ml_agreement_threshold = DecimalParameter(0.60, 0.90, default=0.75, optimize=True, space='buy')
    
    # Traditional indicator support
    rsi_support_enabled = BooleanParameter(default=True, optimize=True, space='buy')
    volume_support_enabled = BooleanParameter(default=True, optimize=True, space='buy')
    trend_support_enabled = BooleanParameter(default=True, optimize=True, space='buy')
    
    # Risk management
    max_open_trades = IntParameter(1, 8, default=3, optimize=True, space='buy')
    risk_per_trade = DecimalParameter(0.01, 0.05, default=0.02, optimize=True, space='buy')
    
    # Retraining control
    retrain_interval_hours = IntParameter(6, 48, default=12, optimize=False, space='buy')
    min_training_samples = IntParameter(100, 500, default=200, optimize=False, space='buy')
    
    # === FREQTRADE SETTINGS ===
    timeframe = '5m'
    can_short = False
    stoploss = -0.10
    startup_candle_count = 200
    
    # ROI table
    minimal_roi = {
        "0": 0.15,
        "30": 0.08,
        "60": 0.04,
        "120": 0.02,
        "240": 0.01,
        "480": 0.0
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Initialize ML engine
        self.ml_engine = MLPredictiveEngine()
        
        # Performance tracking
        self.prediction_accuracy = {}
        self.trade_history = {}
        
        logger.info("🚀 AlexNexusForgeML Strategy Initialized")
        logger.info("🤖 Primary Decision Maker: Machine Learning")
        logger.info("📊 Secondary Support: Traditional Indicators")
    
    def informative_pairs(self):
        """Return additional pairs for informative data"""
        return []
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate all indicators - ML features are primary, traditional indicators are secondary
        """
        pair = metadata.get('pair', 'UNKNOWN')
        
        # === TRADITIONAL INDICATORS (SUPPORTING ROLE) ===
        
        # Basic trend indicators
        dataframe['ema_9'] = ta.EMA(dataframe['close'], timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe['close'], timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe['close'], timeperiod=50)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['rsi_oversold'] = dataframe['rsi'] < 30
        dataframe['rsi_overbought'] = dataframe['rsi'] > 70
        
        # Volume indicators
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        dataframe['volume_above_avg'] = dataframe['volume_ratio'] > 1.2
        
        # Volatility
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']
        
        # Basic trend direction
        dataframe['trend_up'] = (
            (dataframe['close'] > dataframe['ema_9']) &
            (dataframe['ema_9'] > dataframe['ema_21']) &
            (dataframe['ema_21'] > dataframe['ema_50'])
        )
        
        # === ML FEATURE EXTRACTION ===
        try:
            dataframe = self.ml_engine.extract_ml_features(dataframe)
            logger.debug(f"✅ ML features extracted for {pair}")
        except Exception as e:
            logger.error(f"❌ ML feature extraction failed for {pair}: {e}")
        
        # === ML MODEL TRAINING ===
        try:
            # Check if retraining is needed
            should_retrain = self._should_retrain(pair, len(dataframe))
            
            if should_retrain:
                logger.info(f"🔄 Retraining ML models for {pair}")
                result = self.ml_engine.train_models(dataframe.copy(), pair)
                if result['status'] == 'success':
                    logger.info(f"✅ ML models retrained successfully for {pair}")
                else:
                    logger.warning(f"⚠️ ML retraining failed for {pair}: {result.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ ML training failed for {pair}: {e}")
        
        # === ML PREDICTIONS ===
        try:
            ml_predictions = self.ml_engine.predict(dataframe, pair)
            
            # Add ML predictions to dataframe
            dataframe['ml_entry_probability'] = ml_predictions['entry_probability']
            dataframe['ml_exit_probability'] = ml_predictions['exit_probability']
            dataframe['ml_confidence'] = ml_predictions['confidence']
            dataframe['ml_model_agreement'] = ml_predictions['model_agreement']
            dataframe['ml_prediction_6h'] = ml_predictions['prediction_6h']
            dataframe['ml_status'] = ml_predictions['status']
            
            # ML-based entry signals
            dataframe['ml_high_confidence'] = (
                (dataframe['ml_entry_probability'] > self.ml_entry_threshold_high.value) &
                (dataframe['ml_confidence'] > self.ml_confidence_threshold.value) &
                (dataframe['ml_model_agreement'] > self.ml_agreement_threshold.value)
            )
            
            dataframe['ml_medium_confidence'] = (
                (dataframe['ml_entry_probability'] > self.ml_entry_threshold_medium.value) &
                (dataframe['ml_confidence'] > (self.ml_confidence_threshold.value - 0.1)) &
                (dataframe['ml_model_agreement'] > (self.ml_agreement_threshold.value - 0.1))
            ) & ~dataframe['ml_high_confidence']
            
            dataframe['ml_low_confidence'] = (
                (dataframe['ml_entry_probability'] > 0.65) &
                (dataframe['ml_confidence'] > 0.5)
            ) & ~(dataframe['ml_high_confidence'] | dataframe['ml_medium_confidence'])
            
            # ML-based exit signals
            dataframe['ml_exit_signal'] = (
                (dataframe['ml_exit_probability'] > 0.70) |
                (dataframe['ml_entry_probability'] < 0.40)
            )
            
        except Exception as e:
            logger.error(f"❌ ML prediction failed for {pair}: {e}")
            # Fallback values
            dataframe['ml_entry_probability'] = 0.5
            dataframe['ml_exit_probability'] = 0.5
            dataframe['ml_confidence'] = 0.0
            dataframe['ml_model_agreement'] = 0.0
            dataframe['ml_prediction_6h'] = 0.5
            dataframe['ml_status'] = 'failed'
            dataframe['ml_high_confidence'] = False
            dataframe['ml_medium_confidence'] = False
            dataframe['ml_low_confidence'] = False
            dataframe['ml_exit_signal'] = False
        
        # === TRADITIONAL INDICATOR SUPPORT CONDITIONS ===
        
        # RSI support (when enabled)
        if self.rsi_support_enabled.value:
            dataframe['rsi_support'] = (
                (dataframe['rsi'] > 25) & (dataframe['rsi'] < 75)  # Not extreme
            )
        else:
            dataframe['rsi_support'] = True
        
        # Volume support (when enabled)
        if self.volume_support_enabled.value:
            dataframe['volume_support'] = dataframe['volume_above_avg']
        else:
            dataframe['volume_support'] = True
        
        # Trend support (when enabled)
        if self.trend_support_enabled.value:
            dataframe['trend_support'] = (
                dataframe['trend_up'] |
                (dataframe['close'] > dataframe['ema_21'])  # At least above medium-term EMA
            )
        else:
            dataframe['trend_support'] = True
        
        # Combined traditional support
        dataframe['traditional_support'] = (
            dataframe['rsi_support'] &
            dataframe['volume_support'] &
            dataframe['trend_support']
        )
        
        # === FINAL COMPOSITE SCORES ===
        
        # ML-driven entry score (0-100)
        dataframe['ml_entry_score'] = (
            dataframe['ml_entry_probability'] * 40 +  # 40% weight
            dataframe['ml_confidence'] * 30 +         # 30% weight
            dataframe['ml_model_agreement'] * 20 +    # 20% weight
            dataframe['ml_prediction_6h'] * 10        # 10% weight
        )
        
        # Traditional support bonus (0-20 points)
        dataframe['traditional_bonus'] = (
            dataframe['rsi_support'].astype(int) * 5 +
            dataframe['volume_support'].astype(int) * 8 +
            dataframe['trend_support'].astype(int) * 7
        )
        
        # Final composite score
        dataframe['composite_score'] = dataframe['ml_entry_score'] + dataframe['traditional_bonus']
        
        # === COMPREHENSIVE LOGGING ===
        self._log_ml_analysis(dataframe, pair)
        
        return dataframe
    
    def _should_retrain(self, pair: str, current_samples: int) -> bool:
        """Determine if ML models should be retrained"""
        if pair not in self.ml_engine.is_trained or not self.ml_engine.is_trained[pair]:
            return current_samples >= self.min_training_samples.value
        
        last_train = self.ml_engine.last_train_time.get(pair)
        if last_train is None:
            return True
        
        hours_since_train = (datetime.now() - last_train).total_seconds() / 3600
        return hours_since_train >= self.retrain_interval_hours.value
    
    def _log_ml_analysis(self, dataframe: pd.DataFrame, pair: str):
        """
        Comprehensive bilingual logging of ML analysis
        """
        try:
            latest = dataframe.iloc[-1]
            
            # ML metrics
            ml_prob = latest['ml_entry_probability']
            ml_conf = latest['ml_confidence']
            ml_agreement = latest['ml_model_agreement']
            ml_6h = latest['ml_prediction_6h']
            composite_score = latest['composite_score']
            
            # Traditional metrics
            rsi = latest['rsi']
            volume_ratio = latest['volume_ratio']
            atr_pct = latest['atr_pct']
            
            # Count satisfied conditions
            conditions_satisfied = sum([
                latest['ml_high_confidence'],
                latest['ml_medium_confidence'],
                latest['traditional_support'],
                latest['volume_above_avg'],
                latest['trend_up']
            ])
            
            # Log in English and Chinese
            logger.info(f"📊 {pair} ML Analysis | ML分析")
            logger.info(f"🤖 Entry Probability: {ml_prob:.3f} | 入场概率: {ml_prob:.3f}")
            logger.info(f"🎯 ML Confidence: {ml_conf:.3f} | ML置信度: {ml_conf:.3f}")
            logger.info(f"🤝 Model Agreement: {ml_agreement:.3f} | 模型一致性: {ml_agreement:.3f}")
            logger.info(f"🔮 6H Prediction: {ml_6h:.3f} | 6小时预测: {ml_6h:.3f}")
            logger.info(f"📈 Composite Score: {composite_score:.1f}/120 | 综合评分: {composite_score:.1f}/120")
            logger.info(f"📊 RSI: {rsi:.1f} | Volume: {volume_ratio:.2f}x | ATR: {atr_pct:.3f}")
            logger.info(f"✅ Conditions Met: {conditions_satisfied}/5 | 满足条件: {conditions_satisfied}/5")
            
            # Log high-confidence situations
            if latest['ml_high_confidence']:
                logger.info(f"🚀 {pair} HIGH CONFIDENCE ML SIGNAL | 高置信度ML信号")
            elif latest['ml_medium_confidence']:
                logger.info(f"⚡ {pair} MEDIUM CONFIDENCE ML SIGNAL | 中等置信度ML信号")
            
        except Exception as e:
            logger.error(f"❌ Logging failed for {pair}: {e}")
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ML-DRIVEN ENTRY LOGIC
        Primary: ML predictions | Secondary: Traditional indicator confirmation
        """
        pair = metadata.get('pair', 'UNKNOWN')
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        
        # === HIGH CONFIDENCE ML ENTRIES ===
        high_confidence_entry = (
            dataframe['ml_high_confidence'] &
            dataframe['traditional_support'] &  # Basic traditional confirmation
            (dataframe['composite_score'] > 90)
        )
        
        # === MEDIUM CONFIDENCE ML ENTRIES ===
        medium_confidence_entry = (
            dataframe['ml_medium_confidence'] &
            dataframe['traditional_support'] &
            (dataframe['composite_score'] > 75) &
            ~high_confidence_entry  # Not already covered
        )
        
        # === LOW CONFIDENCE ML ENTRIES (AGGRESSIVE) ===
        low_confidence_entry = (
            dataframe['ml_low_confidence'] &
            dataframe['trend_support'] &  # At least trend confirmation
            (dataframe['volume_ratio'] > 1.5) &  # Strong volume
            (dataframe['composite_score'] > 65) &
            ~(high_confidence_entry | medium_confidence_entry)
        )
        
        # === SPECIAL ML BREAKOUT ENTRIES ===
        ml_breakout_entry = (
            (dataframe['ml_entry_probability'] > 0.80) &
            (dataframe['ml_prediction_6h'] > 0.75) &
            (dataframe['volume_ratio'] > 2.0) &  # Very strong volume
            dataframe['trend_up'] &
            (dataframe['rsi'] < 70) &  # Not overbought
            ~(high_confidence_entry | medium_confidence_entry | low_confidence_entry)
        )
        
        # === APPLY ENTRY SIGNALS ===
        
        # High confidence entries
        dataframe.loc[high_confidence_entry, 'enter_long'] = 1
        dataframe.loc[high_confidence_entry, 'enter_tag'] = 'ml_high_conf_95'
        
        # Medium confidence entries
        dataframe.loc[medium_confidence_entry, 'enter_long'] = 1
        dataframe.loc[medium_confidence_entry, 'enter_tag'] = 'ml_medium_conf_80'
        
        # Low confidence entries
        dataframe.loc[low_confidence_entry, 'enter_long'] = 1
        dataframe.loc[low_confidence_entry, 'enter_tag'] = 'ml_low_conf_aggressive'
        
        # Breakout entries
        dataframe.loc[ml_breakout_entry, 'enter_long'] = 1
        dataframe.loc[ml_breakout_entry, 'enter_tag'] = 'ml_breakout_volume'
        
        # === ENTRY LOGGING ===
        recent_entries = dataframe['enter_long'].tail(5).sum()
        if recent_entries > 0:
            latest_entry_tag = dataframe['enter_tag'].iloc[-1]
            latest_ml_prob = dataframe['ml_entry_probability'].iloc[-1]
            latest_score = dataframe['composite_score'].iloc[-1]
            
            logger.info(f"🎯 {pair} ENTRY SIGNAL | 入场信号")
            logger.info(f"🏷️ Type: {latest_entry_tag} | 类型: {latest_entry_tag}")
            logger.info(f"🤖 ML Probability: {latest_ml_prob:.3f} | ML概率: {latest_ml_prob:.3f}")
            logger.info(f"📊 Composite Score: {latest_score:.1f} | 综合评分: {latest_score:.1f}")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ML-DRIVEN EXIT LOGIC
        Primary: ML exit signals | Secondary: Risk management
        """
        pair = metadata.get('pair', 'UNKNOWN')
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''
        
        # === ML-BASED EXITS ===
        
        # Primary ML exit signal
        ml_primary_exit = dataframe['ml_exit_signal']
        
        # ML confidence degradation
        ml_confidence_exit = (
            (dataframe['ml_entry_probability'] < 0.35) &
            (dataframe['ml_confidence'] < 0.40)
        )
        
        # ML model disagreement
        ml_disagreement_exit = (
            (dataframe['ml_model_agreement'] < 0.30) &
            (dataframe['ml_entry_probability'] < 0.50)
        )
        
        # === TRADITIONAL RISK MANAGEMENT EXITS ===
        
        # RSI overbought exit
        rsi_exit = (
            (dataframe['rsi'] > 80) &
            (dataframe['ml_entry_probability'] < 0.60)
        )
        
        # Volume exhaustion exit
        volume_exit = (
            (dataframe['volume_ratio'] < 0.5) &  # Very low volume
            (dataframe['rsi'] > 70) &
            (dataframe['ml_entry_probability'] < 0.55)
        )
        
        # Trend breakdown exit
        trend_exit = (
            (dataframe['close'] < dataframe['ema_21']) &
            (dataframe['ema_9'] < dataframe['ema_21']) &
            (dataframe['ml_entry_probability'] < 0.45)
        )
        
        # === EMERGENCY EXITS ===
        
        # Emergency ML exit
        emergency_ml_exit = (
            (dataframe['ml_exit_probability'] > 0.85) |
            (dataframe['ml_entry_probability'] < 0.20)
        )
        
        # Emergency volatility exit
        emergency_vol_exit = (
            (dataframe['atr_pct'] > 0.08) &  # Very high volatility
            (dataframe['ml_confidence'] < 0.30)
        )
        
        # === APPLY EXIT SIGNALS ===
        
        # Primary ML exits
        dataframe.loc[ml_primary_exit, 'exit_long'] = 1
        dataframe.loc[ml_primary_exit, 'exit_tag'] = 'ml_primary_exit'
        
        # ML confidence exits
        dataframe.loc[ml_confidence_exit & ~ml_primary_exit, 'exit_long'] = 1
        dataframe.loc[ml_confidence_exit & ~ml_primary_exit, 'exit_tag'] = 'ml_confidence_low'
        
        # ML disagreement exits
        dataframe.loc[ml_disagreement_exit & (dataframe['exit_tag'] == ''), 'exit_long'] = 1
        dataframe.loc[ml_disagreement_exit & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'ml_disagreement'
        
        # Traditional exits
        dataframe.loc[rsi_exit & (dataframe['exit_tag'] == ''), 'exit_long'] = 1
        dataframe.loc[rsi_exit & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'rsi_overbought'
        
        dataframe.loc[volume_exit & (dataframe['exit_tag'] == ''), 'exit_long'] = 1
        dataframe.loc[volume_exit & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'volume_exhaustion'
        
        dataframe.loc[trend_exit & (dataframe['exit_tag'] == ''), 'exit_long'] = 1
        dataframe.loc[trend_exit & (dataframe['exit_tag'] == ''), 'exit_tag'] = 'trend_breakdown'
        
        # Emergency exits (override others)
        dataframe.loc[emergency_ml_exit, 'exit_long'] = 1
        dataframe.loc[emergency_ml_exit, 'exit_tag'] = 'emergency_ml_exit'
        
        dataframe.loc[emergency_vol_exit, 'exit_long'] = 1
        dataframe.loc[emergency_vol_exit, 'exit_tag'] = 'emergency_volatility'
        
        # === EXIT LOGGING ===
        recent_exits = dataframe['exit_long'].tail(5).sum()
        if recent_exits > 0:
            latest_exit_tag = dataframe['exit_tag'].iloc[-1]
            latest_ml_exit_prob = dataframe['ml_exit_probability'].iloc[-1]
            
            logger.info(f"🛑 {pair} EXIT SIGNAL | 退出信号")
            logger.info(f"🏷️ Type: {latest_exit_tag} | 类型: {latest_exit_tag}")
            logger.info(f"🤖 ML Exit Probability: {latest_ml_exit_prob:.3f} | ML退出概率: {latest_ml_exit_prob:.3f}")
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str, **kwargs) -> bool:
        """
        Final ML-based trade entry confirmation
        """
        try:
            # Get current ML predictions
            if pair in self.ml_engine.prediction_history and self.ml_engine.prediction_history[pair]:
                latest_prediction = self.ml_engine.prediction_history[pair][-1]
                ml_probability = latest_prediction['probability']
                ml_confidence = latest_prediction['confidence']
                
                # High confidence threshold for final confirmation
                if ml_probability > 0.80 and ml_confidence > 0.70:
                    logger.info(f"✅ {pair} ML ENTRY CONFIRMED | ML入场确认")
                    logger.info(f"🤖 ML Probability: {ml_probability:.3f} | ML概率: {ml_probability:.3f}")
                    logger.info(f"🎯 Confidence: {ml_confidence:.3f} | 置信度: {ml_confidence:.3f}")
                    return True
                elif ml_probability > 0.70:
                    logger.info(f"⚡ {pair} ML ENTRY APPROVED (Medium) | ML入场批准(中等)")
                    return True
                else:
                    logger.warning(f"❌ {pair} ML ENTRY REJECTED - Low probability | ML入场拒绝-低概率")
                    return False
            
            # Fallback: allow entry if no ML data
            logger.info(f"⚠️ {pair} No ML data - allowing entry | 无ML数据-允许入场")
            return True
            
        except Exception as e:
            logger.error(f"❌ ML entry confirmation failed for {pair}: {e}")
            return True  # Allow entry on error
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str, 
                          current_time: datetime, **kwargs) -> bool:
        """
        ML-based trade exit confirmation
        """
        try:
            current_profit = trade.calc_profit_ratio(rate)
            trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
            
            # Always allow stoploss and ROI exits
            if exit_reason in ['stoploss', 'roi', 'trailing_stop_loss']:
                logger.info(f"✅ {pair} {exit_reason.upper()} EXIT | {exit_reason.upper()}退出")
                return True
            
            # Get ML exit confirmation
            if pair in self.ml_engine.prediction_history and self.ml_engine.prediction_history[pair]:
                latest_prediction = self.ml_engine.prediction_history[pair][-1]
                ml_probability = latest_prediction['probability']
                
                # Strong ML exit signal
                if ml_probability < 0.30:
                    logger.info(f"✅ {pair} ML EXIT CONFIRMED - Strong signal | ML退出确认-强信号")
                    return True
                
                # Profit protection
                if current_profit > 0.02 and ml_probability < 0.50:
                    logger.info(f"✅ {pair} ML EXIT CONFIRMED - Profit protection | ML退出确认-利润保护")
                    return True
                
                # Time-based exit with ML confirmation
                if trade_duration > 240 and ml_probability < 0.60:  # 4 hours
                    logger.info(f"✅ {pair} ML EXIT CONFIRMED - Time limit | ML退出确认-时间限制")
                    return True
                
                # Reject weak exit signals if ML is still bullish
                if ml_probability > 0.70:
                    logger.info(f"❌ {pair} ML EXIT REJECTED - Still bullish | ML退出拒绝-仍看涨")
                    return False
            
            # Default: allow exit
            return True
            
        except Exception as e:
            logger.error(f"❌ ML exit confirmation failed for {pair}: {e}")
            return True
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ML-enhanced dynamic stoploss
        """
        try:
            # Base stoploss
            stoploss_value = self.stoploss
            
            # Get ML confidence for dynamic adjustment
            if pair in self.ml_engine.prediction_history and self.ml_engine.prediction_history[pair]:
                latest_prediction = self.ml_engine.prediction_history[pair][-1]
                ml_probability = latest_prediction['probability']
                ml_confidence = latest_prediction['confidence']
                
                # Tighter stoploss if ML confidence is low
                if ml_confidence < 0.40:
                    stoploss_value = max(stoploss_value, -0.05)  # Tighter stoploss
                elif ml_confidence > 0.80 and ml_probability > 0.70:
                    stoploss_value = min(stoploss_value * 0.8, -0.02)  # Looser stoploss for high confidence
                
                # Progressive tightening based on ML probability
                if current_profit > 0.02:  # In profit
                    if ml_probability < 0.40:
                        stoploss_value = max(-0.02, current_profit - 0.03)  # Tight trailing
                    elif ml_probability < 0.50:
                        stoploss_value = max(-0.05, current_profit - 0.04)
            
            return stoploss_value
            
        except Exception as e:
            logger.error(f"❌ Custom stoploss calculation failed for {pair}: {e}")
            return self.stoploss
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        ML-based leverage adjustment
        """
        try:
            # Base leverage
            base_leverage = min(proposed_leverage, max_leverage, 3.0)  # Max 3x
            
            # Adjust based on ML confidence
            if pair in self.ml_engine.prediction_history and self.ml_engine.prediction_history[pair]:
                latest_prediction = self.ml_engine.prediction_history[pair][-1]
                ml_confidence = latest_prediction['confidence']
                
                if ml_confidence > 0.85:
                    leverage = min(base_leverage * 1.2, max_leverage)  # Increase for high confidence
                elif ml_confidence < 0.50:
                    leverage = base_leverage * 0.7  # Reduce for low confidence
                else:
                    leverage = base_leverage
                
                logger.info(f"🎯 {pair} Leverage adjusted: {leverage:.1f}x (ML confidence: {ml_confidence:.3f})")
                return leverage
            
            return base_leverage
            
        except Exception as e:
            logger.error(f"❌ Leverage calculation failed for {pair}: {e}")
            return min(proposed_leverage, max_leverage, 2.0)