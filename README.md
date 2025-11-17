# Multi-timeframe VMD Price Prediction Pipeline

Advanced algorithmic trading system combining multi-timeframe analysis, Variational Mode Decomposition (VMD), and ensemble machine learning for BTCUSD price prediction.

## Features

- **Multi-timeframe Data Aggregation**: M1-M6 timeframes with offset sampling
- **VMD Signal Decomposition**: Rolling Variational Mode Decomposition for feature enhancement
- **Ensemble Modeling**: XGBoost + LSTM hybrid approach
- **Multi-target Prediction**: Simultaneous prediction of multiple price targets
- **Sequence Learning**: LSTM networks for temporal pattern recognition
- **Advanced Feature Engineering**: Comprehensive technical indicator calculation

## Technical Architecture
Tick Data → Multi-timeframe Aggregation → Feature Engineering → VMD Decomposition →
Multi-target Training → Prediction Stacking → LSTM Sequence Modeling → Ensemble Evaluation

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Disclaimer
This trading system is for educational and research purposes. Always test strategies thoroughly with historical data and paper trading before deploying with real capital. Past performance does not guarantee future results.
