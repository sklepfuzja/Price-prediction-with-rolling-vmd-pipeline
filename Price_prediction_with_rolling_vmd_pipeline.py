"""
ALGOTRADING RESEARCH PIPELINE - EURUSD PREDICTION
Author: [sklepfuzja]
Date: 2024

Description: Multi-timeframe feature engineering with rolling VMD decomposition
and ensemble modeling (XGBoost + LSTM) for EURUSD prediction.
"""

# ==================== IMPORTS ====================
import numpy as np
import pandas as pd
from datetime import datetime
import time
import warnings
import matplotlib.pyplot as plt

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Data & Features
import MetaTrader5 as mt5
from Data_download import DataFetcherMT5, PrepareData
from sktime.transformations.series.vmd import VmdTransformer
from Utility_2 import MultiTargetTransformer

# Config
from sklearn import set_config
set_config(transform_output="pandas")
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==================== CONFIGURATION ====================
SYMBOL = 'BTCUSD'
TIMEFRAME = 'M1'
DATE_FROM = datetime(2025, 7, 24)
DATE_TO = datetime(2025, 7, 26)
SEQUENCE_LENGTH = 240
TEST_SIZE = 0.66
EPOCHS = 10
BATCH_SIZE = 128

# MT5 Timeframes
M1, M2, M3, M4, M5, M6 = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M2, mt5.TIMEFRAME_M3, 
                          mt5.TIMEFRAME_M4, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M6]

# ==================== DATA FETCHING ====================
print("📊 Fetching data...")

fetcher = DataFetcherMT5(login=None, password=None, server=None)
df_ticks = fetcher.fetch_data_ticks_range(symbol=SYMBOL, date_from=DATE_FROM, date_to=DATE_TO)

# Create multiple timeframes
df1m = fetcher.aggregate_data_bid(frequency='1T', offset=None, df=df_ticks)
df2m0 = fetcher.aggregate_data_bid(frequency='2T', offset=None, df=df_ticks)
df2m1 = fetcher.aggregate_data_bid(frequency='2T', offset='1T', df=df_ticks)
df3m0 = fetcher.aggregate_data_bid(frequency='3T', offset=None, df=df_ticks)
df3m1 = fetcher.aggregate_data_bid(frequency='3T', offset='1T', df=df_ticks)
df3m2 = fetcher.aggregate_data_bid(frequency='3T', offset='2T', df=df_ticks)
df4m0 = fetcher.aggregate_data_bid(frequency='4T', offset=None, df=df_ticks)
df4m1 = fetcher.aggregate_data_bid(frequency='4T', offset='1T', df=df_ticks)
df4m2 = fetcher.aggregate_data_bid(frequency='4T', offset='2T', df=df_ticks)
df4m3 = fetcher.aggregate_data_bid(frequency='4T', offset='3T', df=df_ticks)
df5m0 = fetcher.aggregate_data_bid(frequency='5T', offset=None, df=df_ticks)
df5m1 = fetcher.aggregate_data_bid(frequency='5T', offset='1T', df=df_ticks)
df5m2 = fetcher.aggregate_data_bid(frequency='5T', offset='2T', df=df_ticks)
df5m3 = fetcher.aggregate_data_bid(frequency='5T', offset='3T', df=df_ticks)
df5m4 = fetcher.aggregate_data_bid(frequency='5T', offset='4T', df=df_ticks)
df6m0 = fetcher.aggregate_data_bid(frequency='6T', offset=None, df=df_ticks)
df6m1 = fetcher.aggregate_data_bid(frequency='6T', offset='1T', df=df_ticks)
df6m2 = fetcher.aggregate_data_bid(frequency='6T', offset='2T', df=df_ticks)
df6m3 = fetcher.aggregate_data_bid(frequency='6T', offset='3T', df=df_ticks)
df6m4 = fetcher.aggregate_data_bid(frequency='6T', offset='4T', df=df_ticks)
df6m5 = fetcher.aggregate_data_bid(frequency='6T', offset='5T', df=df_ticks)

print(f"✅ Data fetched - ticks: {df_ticks.shape}")

# ==================== FEATURE ENGINEERING ====================
print("⚙️ Computing features...")

feature_processor = PrepareData()
start_time = time.time()

# List of pairs (variable_name, dataframe)
frames = [
    ('df1m', df1m), ('df2m0', df2m0), ('df2m1', df2m1), ('df3m0', df3m0), ('df3m1', df3m1),
    ('df3m2', df3m2), ('df4m0', df4m0), ('df4m1', df4m1), ('df4m2', df4m2), ('df4m3', df4m3),
    ('df5m0', df5m0), ('df5m1', df5m1), ('df5m2', df5m2), ('df5m3', df5m3), ('df5m4', df5m4),
    ('df6m0', df6m0), ('df6m1', df6m1), ('df6m2', df6m2), ('df6m3', df6m3), ('df6m4', df6m4), ('df6m5', df6m5)
]

# Processing in loop
for name, df in frames:
    processed_df = feature_processor.feature_dataset_1(df).fillna(0)
    globals()[name] = processed_df  # Update original variable

print(f"✅ Features computed in {time.time() - start_time:.2f}s")

# ==================== VMD DECOMPOSITION ====================
print("🌀 Applying VMD decomposition...")

def rolling_vmd_decomposition(df, columns=['open', 'high', 'low', 'close'], window_size=120, k=10):
    """Rolling VMD implementation"""
    vmd = VmdTransformer(K=k, kMax=10)
    df = df.copy()
    
    for col in columns:
        for i in range(k):
            df[f'{col}_VMD{i+1}'] = np.nan
        
        for i in range(len(df) - window_size + 1):
            window = df[col].iloc[i:i+window_size].reset_index(drop=True)
            decomposition = vmd.fit_transform(pd.DataFrame(window))
            
            for j in range(k):
                df.loc[df.index[i+window_size-1], f'{col}_VMD{j+1}'] = decomposition.iloc[-1, j]
    
    return df

# List of dataframes for VMD
vmd_dataframes = [df1m, df2m0, df2m1, df3m0, df3m1, df3m2, df4m0, df4m1, df4m2, df4m3,
                 df5m0, df5m1, df5m2, df5m3, df5m4, df6m0, df6m1, df6m2, df6m3, df6m4, df6m5]

# Apply VMD to all
for i, df in enumerate(vmd_dataframes):
    vmd_dataframes[i] = rolling_vmd_decomposition(df, window_size=120).drop(columns=['open', 'high', 'low', 'close'])

print("✅ VMD decomposition completed")

# ==================== DATA COMBINATION ====================
print("🔗 Combining datasets...")

# Combine timeframes
df2m = pd.concat([df2m0.add_suffix('_M2'), df2m1.add_suffix('_M2')], axis=0).sort_index().fillna(0)
df3m = pd.concat([df3m0.add_suffix('_M3'), df3m1.add_suffix('_M3'), df3m2.add_suffix('_M3')], axis=0).sort_index().fillna(0)
df4m = pd.concat([df4m0.add_suffix('_M4'), df4m1.add_suffix('_M4'), df4m2.add_suffix('_M4'), df4m3.add_suffix('_M4')], axis=0).sort_index().fillna(0)
df5m = pd.concat([df5m0.add_suffix('_M5'), df5m1.add_suffix('_M5'), df5m2.add_suffix('_M5'), df5m3.add_suffix('_M5'), df5m4.add_suffix('_M5')], axis=0).sort_index().fillna(0)
df6m = pd.concat([df6m0.add_suffix('_M6'), df6m1.add_suffix('_M6'), df6m2.add_suffix('_M6'), df6m3.add_suffix('_M6'), df6m4.add_suffix('_M6'), df6m5.add_suffix('_M6')], axis=0).sort_index().fillna(0)

df1m = pd.concat([df1m, df2m, df3m, df4m, df5m, df6m], axis=1)

print(f"✅ Combined dataset shape: {df1m.shape}")

# ==================== MODEL TRAINING ====================
print("🤖 Training models...")

# Create targets
X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = feature_processor.create_targets_with_specific_shift_small(df1m, type='binary', validation_dataset=True, shift=1, start_time='02:00', end_time='22:00')
X_train1, X_val1, X_test1, y_train1_reg_diff, y_val1_reg_diff, y_test1_reg_diff = feature_processor.create_targets_with_specific_shift_small(df1m, type='reg_diff', validation_dataset=True, shift=1, start_time='02:00', end_time='22:00')
X_train1, X_val1, X_test1, y_train1_reg_raw, y_val1_reg_raw, y_test1_reg_raw = feature_processor.create_targets_with_specific_shift_small(df1m, type='reg_raw', validation_dataset=True, shift=1, start_time='02:00', end_time='22:00')

# Train multi-target models
multi_target_transformer = MultiTargetTransformer(base_pipeline=Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('scaler_2', QuantileTransformer()),
    ('selectkbest', SelectKBest(f_classif, k=60)),
    ('pca', PCA()),
    ('classifier2', XGBClassifier())
]), n_targets=y_train1.shape[1])

multi_target_transformer.fit(X_train1, y_train1)
train_predictions = multi_target_transformer.predict(X_train1)
val_predictions = multi_target_transformer.predict(X_val1)
test_predictions = multi_target_transformer.predict(X_test1)

print("✅ Multi-target models trained")

# ==================== RESULTS ====================
print("📊 Final Results:")

for i in range(multi_target_transformer.n_targets):
    accuracy = accuracy_score(y_val1.iloc[:, i], val_predictions.iloc[:, i])
    print(f"Target {i} Accuracy: {accuracy:.2f}")

print("🎉 Pipeline completed successfully!")

# ==================== LSTM SEQUENCE MODEL ====================
print("🧠 Training LSTM sequence model...")

def create_sequences(X, y, n_steps):
    """Create sequences for LSTM training."""
    Xs, ys = [], []
    for i in range(len(X) - n_steps + 1):
        Xs.append(X.iloc[i:i + n_steps].values)
        ys.append(y.iloc[i + n_steps - 1])
    return np.array(Xs), np.array(ys)

# Use binary predictions as features
X_train_enhanced = pd.concat([X_train1, train_predictions.add_suffix('_pred')], axis=1)
X_val_enhanced = pd.concat([X_val1, val_predictions.add_suffix('_pred')], axis=1)
X_test_enhanced = pd.concat([X_test1, test_predictions.add_suffix('_pred')], axis=1)

# Prepare sequence data (using enhanced features)
X_seq_data = pd.concat([X_val_enhanced, X_test_enhanced], axis=0).fillna(0)
y_seq_target = pd.concat([y_val1.iloc[:, 0], y_test1.iloc[:, 0]], axis=0).fillna(0)

# Create sequences
X_sequences, y_sequences = create_sequences(X_seq_data, y_seq_target, SEQUENCE_LENGTH)
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
    X_sequences, y_sequences, test_size=TEST_SIZE, shuffle=False
)

print(f"✅ Sequences created - Train: {X_seq_train.shape}, Test: {X_seq_test.shape}")

# Build and train LSTM
lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True,
         input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = lstm_model.fit(
    X_seq_train, y_seq_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_seq_test, y_seq_test),
    verbose=1
)

# Evaluate model
y_seq_pred = lstm_model.predict(X_seq_test)
final_accuracy = accuracy_score(y_seq_test > 0.5, y_seq_pred > 0.5)

# ==================== RESULTS ====================
print("📊 Final Results:")

# Multi-target performance
for i in range(multi_target_transformer.n_targets):
    accuracy = accuracy_score(y_val1.iloc[:, i], val_predictions.iloc[:, i])
    print(f"Target {i} Accuracy: {accuracy:.2f}")

print(f"🎯 LSTM Sequence Model Accuracy: {final_accuracy:.2%}")
print("🎉 Pipeline completed successfully!")