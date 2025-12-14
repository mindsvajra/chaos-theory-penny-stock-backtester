# Chaos Theory-Based Penny Stock Backtester (Educational Project)

**⚠️ IMPORTANT DISCLAIMER ⚠️**

This is a **proof-of-concept educational script** exploring the application of nonlinear dynamics and chaos theory to financial time series.

- Chaos detection in markets is highly debated — most academic research concludes financial markets are noisy/stochastic, not truly chaotic.
- This strategy is **not profitable in live trading** (backtests often overfit, especially on volatile penny stocks).
- Penny stocks are extremely risky: low liquidity, high manipulation risk, pump-and-dump schemes common.
- **Do NOT use this for real money.** Past performance (even in backtests) is no guarantee of future results.
- Trading involves substantial risk of loss.

This repo is for **portfolio/educational purposes only** — to demonstrate:
- Phase space reconstruction
- Mutual information for time delay (τ)
- False nearest neighbors for embedding dimension (m)
- Correlation dimension & Lyapunov exponent estimation
- Wavelet transforms for reversal detection
- Monte Carlo robustness testing
- Realistic backtesting with slippage/fees

## Features
- Fetches 5 years of daily data via yfinance
- Chaos metrics: τ, m, correlation dimension D, largest Lyapunov λ
- Simplified nearest-neighbor prediction
- Wavelet (Daubechies) reversal signals
- Backtesting with profit target/stop loss/horizon
- Monte Carlo simulations (50 runs with price noise)
- Per-stock equity curve PNG reports (with trade markers)
- CSV export of results

Note: Some tickers in the default list may be delisted/invalid by late 2025 — edit the list as needed.

## Requirements
```txt
yfinance==0.2.40
numpy==2.1.2
pandas==2.2.3
scipy==1.14.1
pywavelets==1.7.0
scikit-learn==1.5.2
matplotlib==3.9.2

Install with: pip install -r requirements.txt

How to Run

Save the main script as chaos_penny_backtester.py (or similar)
Run: python chaos_penny_backtester.py
Outputs:penny_stocks_strategy_results.csv
Individual {ticker}_backtest_report.png plots

