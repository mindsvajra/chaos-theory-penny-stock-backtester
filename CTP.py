# Note: This script implements an enhanced chaos theory-based trading strategy for 20 penny stocks.
# Enhancements include:
# - Slippage: Simulated by adjusting entry and exit prices adversely (default 0.1%).
# - Trading fees: Commission rate applied to buy and sell amounts (default 0.1%).
# - Monte Carlo simulations: Run multiple backtests with Gaussian noise added to prices to reduce overfitting biases and estimate robustness.
# - Visual report: For each stock, generates a PNG plot showing the equity curve, starting/ending capital, and vertical lines for trade entry/exit dates.
# It requires installation of additional libraries: pip install yfinance pywavelets scikit-learn matplotlib
# Run this on your local machine with internet access to fetch data from yfinance.
# The script fetches 5 years of daily closing prices, performs chaos detection,
# prediction modeling (simplified), wavelet-based reversal detection, and enhanced backtesting.
# Results (e.g., chaos metrics, backtest returns with MC stats) are exported to 'penny_stocks_strategy_results.csv'.

import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean
import pywt  # For wavelet transforms
from sklearn.neighbors import NearestNeighbors  # For nearest neighbors in phase space
from scipy.stats import linregress  # For Lyapunov slope
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# List of 20 penny stocks (selected based on most active under $5 as of October 2025)
tickers = [
    'BYND', 'RANI', 'GTVH', 'CAN', 'NBRI', 'QEDN', 'DVLT', 'GPUS', 'BTBT', 'IOBT',
    'LTNC', 'VIVK', 'BIEL', 'ASST', 'RDHL', 'MSAI', 'QSI', 'IOVA', 'IXHL', 'CGC'
]

# Function to compute time delay tau using mutual information
def compute_tau(data, max_tau=20):
    def mutual_info(x, y):
        # Simple histogram-based mutual information
        bins = 10
        x = np.asarray(x).ravel()  # Ensure 1D array
        y = np.asarray(y).ravel()  # Ensure 1D array
        c_xy = np.histogram2d(x, y, bins)[0]
        c_x = np.histogram(x, bins)[0]
        c_y = np.histogram(y, bins)[0]
        
        # Normalize to probabilities
        p_x = c_x / np.sum(c_x) if np.sum(c_x) > 0 else np.ones_like(c_x) / len(c_x)
        p_y = c_y / np.sum(c_y) if np.sum(c_y) > 0 else np.ones_like(c_y) / len(c_y)
        p_xy = c_xy / np.sum(c_xy) if np.sum(c_xy) > 0 else np.ones_like(c_xy.ravel()) / len(c_xy.ravel())
        
        h_x = stats.entropy(p_x)
        h_y = stats.entropy(p_y)
        h_xy = stats.entropy(p_xy.ravel())
        return h_x + h_y - h_xy
    
    mi = []
    for tau in range(1, max_tau + 1):
        x = data[:-tau]
        y = data[tau:]
        mi.append(mutual_info(x, y))
    return np.argmin(mi) + 1  # First minimum

# Function to compute embedding dimension m using false nearest neighbors
def compute_m(data, tau, max_m=10, rtol=15, atol=2):
    std_data = np.std(data)
    def false_nearest_neighbors(phase_space, m):
        nbrs = NearestNeighbors(n_neighbors=2).fit(phase_space)
        distances, indices = nbrs.kneighbors(phase_space)
        fnn = 0
        count = 0  # Count valid points to avoid division by zero
        for i in range(len(phase_space) - tau):
            j = indices[i, 1]
            if j + m * tau >= len(data):
                continue
            dist_m = distances[i, 1]
            if dist_m == 0:
                continue
            next_diff = abs(data[i + m * tau] - data[j + m * tau])
            next_dist_rel = next_diff / dist_m
            if next_dist_rel > rtol or next_diff > atol * std_data:
                fnn += 1
            count += 1
        return fnn / count if count > 0 else 1.0  # Assume high FNN if no valid points
    
    for m in range(1, max_m + 1):
        phase_space = reconstruct_phase_space(data, tau, m)
        fnn_ratio = false_nearest_neighbors(phase_space, m)
        if fnn_ratio < 0.05:  # Threshold for convergence
            return m
    return max_m  # Default to max if not converged

# Function to compute correlation dimension D (simplified Grassberger-Procaccia)
def compute_correlation_dimension(phase_space, r_values=np.logspace(-3, 0, 10)):
    c_r = []
    for r in r_values:
        nbrs = NearestNeighbors(radius=r).fit(phase_space)
        n_neighbors = nbrs.radius_neighbors(return_distance=False)
        c_r.append(np.mean([len(neigh) - 1 for neigh in n_neighbors]) / len(phase_space)**2)
    c_r = np.array(c_r)
    valid = (c_r > 0) & (c_r < 1)
    if np.sum(valid) < 2:
        return np.nan
    p = np.polyfit(np.log(r_values[valid]), np.log(c_r[valid]), 1)
    return p[0]  # Slope is the first coefficient

# Function to compute largest Lyapunov exponent (Rosenstein method, simplified)
def compute_lyapunov(phase_space, tau, k_max=10):
    nbrs = NearestNeighbors(n_neighbors=2).fit(phase_space)
    distances, indices = nbrs.kneighbors(phase_space)
    divergences = []
    for i in range(len(phase_space) - k_max * tau):
        j = indices[i, 1]
        if j + k_max * tau >= len(phase_space):
            continue
        div = []
        for k in range(1, k_max + 1):
            d_k = euclidean(phase_space[i + k], phase_space[j + k])
            if d_k > 0:
                div.append(np.log(d_k))
        if len(div) >= 2:  # Need at least 2 points for regression
            ks = list(range(1, len(div) + 1))
            slope = linregress(ks, div).slope
            divergences.append(slope)
    if not divergences:
        return np.nan
    return np.mean(divergences)

# Function to reconstruct phase space
def reconstruct_phase_space(data, tau, m):
    n = len(data) - (m - 1) * tau
    phase_space = np.zeros((n, m))
    for i in range(m):
        phase_space[:, i] = data[i * tau : i * tau + n]
    return phase_space

# Function to normalize phase space (improvement for scale invariance)
def normalize_phase_space(phase_space):
    mean = np.mean(phase_space, axis=0)
    std = np.std(phase_space, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (phase_space - mean) / std

# Function for simple prediction using nearest neighbors
def predict_next(phase_space, steps=1):
    if len(phase_space) <= steps + 1:  # Need at least steps + 2 for fit + query
        return np.nan
    nbrs = NearestNeighbors(n_neighbors=1).fit(phase_space[:-steps])
    _, idx = nbrs.kneighbors(phase_space[-steps:])
    predictions = []
    for i in idx.flatten():
        if i + steps >= len(phase_space):
            continue  # Skip invalid
        predictions.append(phase_space[i + steps, -1])  # Evolve from neighbor
    return np.mean(predictions) if predictions else np.nan

# Function for wavelet-based reversal detection
def detect_reversals(data, wavelet='db4', level=4, threshold=2):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Reconstruct details only (zero out approximation)
    details_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    reconstructed_details = pywt.waverec(details_coeffs, wavelet)
    # Trim to match data length if padded
    reconstructed_details = reconstructed_details[:len(data)]
    spikes = np.abs(reconstructed_details) > threshold * np.std(reconstructed_details)
    diff_sign = np.sign(np.diff(data))
    reversals = np.diff(diff_sign)
    # Pad reversals to match data length: add 0 at start and end
    padded_reversals = np.concatenate(([0], reversals, [0]))
    signals = np.where(spikes & (padded_reversals != 0))[0]
    return signals  # Indices of potential reversals

# Enhanced function for basic backtesting with fees, slippage, equity curve, and trade dates
def backtest_strategy(df, predicted_next, reversals, horizon=5, profit_target=0.05, stop_loss=0.02,
                      start_capital=10000, fee_rate=0.001, slippage=0.001):
    prices = df.values
    dates = df.index
    trade_returns = []
    trades = []  # List of (entry_date, exit_date)
    equity = np.zeros(len(prices))
    equity[0] = start_capital
    current_cash = start_capital
    in_position = False
    shares = 0
    invested = 0
    entry_i = 0

    for i in range(1, len(prices)):
        if in_position:
            # Mark-to-market (without slippage, as it's unrealized)
            equity[i] = current_cash + shares * prices[i]
            current_return = (prices[i] - (invested / shares)) / (invested / shares)  # Approximate for check
            if current_return >= profit_target or current_return <= -stop_loss or (i - entry_i >= horizon):
                # Exit
                exit_price_with_slip = prices[i] * (1 - slippage)
                gross_proceeds = shares * exit_price_with_slip
                sell_fee = gross_proceeds * fee_rate
                net_proceeds = gross_proceeds - sell_fee
                current_cash += net_proceeds
                trade_return = (net_proceeds - invested) / invested
                trade_returns.append(trade_return)
                trades.append((dates[entry_i], dates[i]))
                in_position = False
                shares = 0
                invested = 0
        else:
            equity[i] = current_cash

        if not in_position and i in reversals and not np.isnan(predicted_next[i-1]) and predicted_next[i-1] > prices[i]:
            # Note: predicted_next[i-1] predicts prices[i], so check if predicted > current
            # Enter
            entry_price_with_slip = prices[i] * (1 + slippage)
            buy_fee = current_cash * fee_rate
            invested = current_cash - buy_fee
            shares = invested / entry_price_with_slip
            current_cash = 0
            in_position = True
            entry_i = i

    # If still in position at end, force exit at last price
    if in_position:
        i = len(prices) - 1
        exit_price_with_slip = prices[i] * (1 - slippage)
        gross_proceeds = shares * exit_price_with_slip
        sell_fee = gross_proceeds * fee_rate
        net_proceeds = gross_proceeds - sell_fee
        current_cash += net_proceeds
        trade_return = (net_proceeds - invested) / invested
        trade_returns.append(trade_return)
        trades.append((dates[entry_i], dates[i]))
        equity[i] = current_cash

    total_return = (current_cash / start_capital) - 1
    sharpe = np.mean(trade_returns) / np.std(trade_returns) if len(trade_returns) > 1 and np.std(trade_returns) != 0 else 0
    num_trades = len(trade_returns)
    ending_capital = current_cash

    return total_return, sharpe, num_trades, ending_capital, equity, trades, dates

# Main processing
results = []
mc_sims = 50  # Number of Monte Carlo simulations
noise_std = 0.001  # Standard deviation of Gaussian noise for price perturbation

for ticker in tickers:
    try:
        # Fetch data: last 5 years daily closes
        df = yf.download(ticker, period='5y', progress=False)['Close'].dropna()
        if len(df) < 500:
            print(f"Skipping {ticker}: Insufficient data")
            continue
        data = df.values.ravel()  # Numpy array of closes, flattened to 1D

        # Phase 1: Chaos Detection
        tau = compute_tau(data)
        m = compute_m(data, tau)
        phase_space = reconstruct_phase_space(data, tau, m)
        norm_phase_space = normalize_phase_space(phase_space)  # Improvement: normalize for metrics
        D = compute_correlation_dimension(norm_phase_space)
        lambda_exp = compute_lyapunov(norm_phase_space, tau)
        is_chaotic = not np.isnan(D) and lambda_exp > 0
        
        if not is_chaotic:
            print(f"Skipping {ticker}: Not chaotic")
            continue
        
        # Phase 2: Prediction (rolling next-step predictions)
        steps = 1
        min_length = (m - 1) * tau + steps + 1
        predicted_next = np.full(len(data), np.nan)  # Aligned: predicted_next[i] = pred for data[i+1]
        for i in range(len(data) - 1):  # Up to len-2 to avoid predicting beyond data
            subset_length = i + 1
            if subset_length < min_length:
                continue
            ps = reconstruct_phase_space(data[:subset_length], tau, m)
            pred = predict_next(ps, steps=steps)
            predicted_next[i] = pred  # Prediction for data[i+1]
        
        # Compute in-sample prediction error (for existing data)
        valid_mask = ~np.isnan(predicted_next[:-1])
        if np.any(valid_mask):
            predicted_vals = predicted_next[:-1][valid_mask]
            actual_vals = data[1:][valid_mask]
            error = np.mean(np.abs((predicted_vals - actual_vals) / actual_vals)) * 100
        else:
            error = np.nan
        
        # Phase 3: Reversal Detection
        reversals = detect_reversals(data)
        
        # Backtesting with Monte Carlo for stats
        horizon = int(1 / lambda_exp) if lambda_exp > 0 else 5
        mc_total_returns = []
        mc_sharpes = []
        mc_num_trades = []
        mc_ending_capitals = []
        
        for sim in range(mc_sims):
            perturbed_prices = data * (1 + np.random.normal(0, noise_std, len(data)))
            perturbed_df = pd.Series(perturbed_prices, index=df.index)
            total_return, sharpe, num_trades, ending_capital, _, _, _ = backtest_strategy(
                perturbed_df, predicted_next, reversals, horizon
            )
            mc_total_returns.append(total_return)
            mc_sharpes.append(sharpe)
            mc_num_trades.append(num_trades)
            mc_ending_capitals.append(ending_capital)
        
        mean_total_return = np.mean(mc_total_returns)
        std_total_return = np.std(mc_total_returns)
        mean_sharpe = np.mean(mc_sharpes)
        mean_num_trades = np.mean(mc_num_trades)
        mean_ending_capital = np.mean(mc_ending_capitals)
        
        # Run one backtest on original data for visual report
        _, _, _, ending_capital_original, equity_original, trades_original, dates_original = backtest_strategy(
            df, predicted_next, reversals, horizon
        )
        
        # Generate visual report (PNG)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_original, equity_original, label='Equity Curve')
        for entry_date, exit_date in trades_original:
            ax.axvline(entry_date, color='green', linestyle='--', alpha=0.5, label='Entry' if 'Entry' not in ax.get_legend_handles_labels()[1] else '')
            ax.axvline(exit_date, color='red', linestyle='--', alpha=0.5, label='Exit' if 'Exit' not in ax.get_legend_handles_labels()[1] else '')
        ax.set_title(f'{ticker} Backtest: Start Capital ${10000:.2f}, End Capital ${float(ending_capital_original):.2f}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        plt.savefig(f'{ticker}_backtest_report.png')
        plt.close()
        
        # Collect results
        results.append({
            'Ticker': ticker,
            'Tau': tau,
            'M': m,
            'D': D,
            'Lambda': lambda_exp,
            'Prediction_Error_%': error,
            'Mean_Total_Return': mean_total_return,
            'Std_Total_Return': std_total_return,
            'Mean_Sharpe_Ratio': mean_sharpe,
            'Mean_Num_Trades': mean_num_trades,
            'Mean_Ending_Capital': mean_ending_capital
        })
        print(f"Processed {ticker}")
        
    except Exception as e:
        import traceback
        print(f"Error processing {ticker}: {e}")
        print(traceback.format_exc())

# Export to CSV
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('penny_stocks_strategy_results.csv', index=False)
    print("Results exported to penny_stocks_strategy_results.csv")
    print("Visual reports saved as PNG files for each ticker.")
else:
    print("No results to export")