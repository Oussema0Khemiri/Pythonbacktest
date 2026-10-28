import os
import io
import base64
import numpy as np
import pandas as pd
import quantstats as qs
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
# This command extends Pandas so that QuantStats functions can be called directly on DataFrames
qs.extend_pandas()

def get_mt5_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, n_candles=1000):
    """Fetches data from MT5. (Requires MT5 to be initialized globally first)"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    if rates is None:
        print(f"Failed to get data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def generate_tearsheet(strategy_returns, title="Strategy Performance", output_folder="reports"):
    """
    Generates the QuantStats HTML with Monte Carlo and saves it inside 
    a dedicated subfolder (default: 'reports/').
    """
    clean_rets = strategy_returns.dropna()
    daily_returns = (1 + clean_rets).resample('D').prod() - 1
    daily_returns = daily_returns.dropna()
    # Trim leading zeros only (before the strategy starts trading)
    if (daily_returns != 0).any():
        first_nonzero = daily_returns.ne(0).idxmax()
        daily_returns = daily_returns.loc[first_nonzero:]

    if len(daily_returns) == 0:
        return

    # 1. Create the reports folder if it doesn't already exist
    target_dir = os.path.join(os.getcwd(), output_folder)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # 2. Set the destination path inside the folder
        safe_title = title.replace(' ', '_').replace('/', '_')
        html_filename = f"{safe_title}_tearsheet.html"
        full_path = os.path.join(target_dir, html_filename)
        
        # 3. Generate base QuantStats HTML
        qs.reports.html(daily_returns, title=title, output=full_path)
        
        # 4. Generate Monte Carlo Plot
        n_days = len(daily_returns)
        mc_simulations = 1000
        rng = np.random.default_rng(seed=42)
        random_indices = rng.integers(0, n_days, size=(mc_simulations, n_days))
        mc_paths = daily_returns.values[random_indices]
        cum_paths = np.cumprod(1 + mc_paths, axis=1)
        actual_cum = np.cumprod(1 + daily_returns.values)
        
        # Plot percentile bands instead of 1,000 individual lines (renders in <1s)
        p5 = np.percentile(cum_paths, 5, axis=0)
        p50 = np.percentile(cum_paths, 50, axis=0)
        p95 = np.percentile(cum_paths, 95, axis=0)
        
        plt.figure(figsize=(10, 5))
        plt.fill_between(range(n_days), p5, p95, alpha=0.2, color='grey', label='5th-95th Percentile')
        plt.plot(p50, color='grey', linewidth=1, linestyle='--', label='Median Path')
        plt.plot(actual_cum, color='#d62728', linewidth=2, label='Actual Strategy')
        plt.title(f'Monte Carlo Stress Test: {title}')
        plt.ylabel('Growth Multiple')
        plt.xlabel('Trading Days')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 5. Convert to Base64 and Inject into HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        img_tag = f'<div style="text-align:center; margin-top: 50px;"><h2>Monte Carlo Stress Test</h2><img src="data:image/png;base64,{img_base64}" style="max-width:100%; border: 1px solid #ddd; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>'
        
        with open(full_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        html_content = html_content.replace('</body>', f'{img_tag}</body>')
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    except Exception as e:
        print(f"❌ Failed to generate report for {title}: {e}")

def calculate_ai_metrics(strategy_returns, mc_simulations=1000):
    """
    Generates a massive dictionary of advanced statistical and Monte Carlo metrics.
    This is designed specifically to be fed into an AI hyperparameter optimizer.
    """
    # Clean and convert to daily returns (industry standard for risk metrics)
    clean_rets = strategy_returns.dropna()
    if len(clean_rets) < 2:
        return None
        
    daily_rets = (1 + clean_rets).resample('D').prod() - 1
    daily_rets = daily_rets.dropna()
    # Trim leading zeros only (before the strategy starts trading)
    if (daily_rets != 0).any():
        first_nonzero = daily_rets.ne(0).idxmax()
        daily_rets = daily_rets.loc[first_nonzero:]
    
    if len(daily_rets) < 2:
        return None

    # 1. ADVANCED RISK & REWARD METRICS
    metrics = {
        'cagr': qs.stats.cagr(daily_rets),                   # Compounded Annual Growth Rate
        'max_drawdown': qs.stats.max_drawdown(daily_rets),   # Worst historical drop
        'sharpe': qs.stats.sharpe(daily_rets),               # Return vs. Volatility
        'sortino': qs.stats.sortino(daily_rets),             # Return vs. Downside Volatility
        'calmar': qs.stats.calmar(daily_rets),               # Return vs. Drawdown
        'win_rate': qs.stats.win_rate(daily_rets),
        'profit_factor': qs.stats.profit_factor(daily_rets),
        'tail_ratio': qs.stats.tail_ratio(daily_rets),       # Size of right tail (wins) vs left tail (losses)
        'value_at_risk': qs.stats.value_at_risk(daily_rets), # 95% confidence max daily loss
        'expected_shortfall': qs.stats.expected_shortfall(daily_rets), # Average loss on the worst 5% of days
        'kelly_criterion': qs.stats.kelly_criterion(daily_rets),       # Ideal % of bankroll to risk
        'kurtosis': qs.stats.kurtosis(daily_rets),           # Fat tails (extreme events) probability
        'skewness': qs.stats.skew(daily_rets),               # Asymmetry of returns
        'n_trades': int((daily_rets != 0).sum())              # Number of active trading days
    }

    # 2. MONTE CARLO SIMULATION (Bootstrapping)
    # We scramble the daily returns randomly 1,000 times to simulate alternate timelines.
    n_days = len(daily_rets)
    
    # Fast numpy matrix generation: create 1,000 different randomized histories
    rng = np.random.default_rng(seed=42)
    random_indices = rng.integers(0, n_days, size=(mc_simulations, n_days))
    mc_paths = daily_rets.values[random_indices]
    
    # Calculate cumulative equity curves for all 1,000 realities
    cum_paths = np.cumprod(1 + mc_paths, axis=1)
    
    # Extract final returns for all 1,000 realities
    final_returns = cum_paths[:, -1] - 1
    
    # Extract max drawdowns for all 1,000 realities
    running_max = np.maximum.accumulate(cum_paths, axis=1)
    drawdowns = (cum_paths - running_max) / running_max
    max_drawdowns = np.min(drawdowns, axis=1)
    
    # 3. MONTE CARLO METRICS (The "Anti-Curve-Fitting" safeguards)
    metrics['mc_median_return'] = np.median(final_returns)
    
    # The 5th Percentile: If we simulate 1000 futures, what does the WORST 5% look like?
    # This is what you will tell the AI to optimize for.
    metrics['mc_worst_5%_return'] = np.percentile(final_returns, 5)
    metrics['mc_worst_5%_drawdown'] = np.percentile(max_drawdowns, 5)

    # Clean up NaN or Inf values for the AI
    for key, value in metrics.items():
        if pd.isna(value) or np.isinf(value):
            metrics[key] = 0.0
            
    return metrics

def split_data(df, train_pct=0.7):
    """
    Splits the data chronologically. 
    e.g., First 70% of years for AI Training, Last 30% for Blind Testing.
    """
    split_index = int(len(df) * train_pct)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    print(f"Data Split: {len(train_df)} Training Candles | {len(test_df)} Testing Candles")
    return train_df, test_df

def apply_trading_costs(df, positions, point_size, spread_points):
    """
    Universally calculates returns and deducts broker spreads.
    Works flawlessly on Forex, Gold, Crypto, and Indices.
    """
    # 1. Base market returns
    market_rets = df['close'].pct_change()
    
    # 2. Strategy gross returns
    gross_rets = positions.shift(1) * market_rets
    
    # 3. Calculate how many trades happened
    # Binary change detection: any position change = 1 spread charge
    # Consistent with apply_sl_tp_and_costs which charges once per flip
    position_changes = (positions.diff().abs() > 0).astype(float).fillna(0)
    
    # 4. Calculate exact cost using MT5's native point and spread
    # Example EURUSD: spread_points (15) * point_size (0.00001) = 0.00015 cost
    cost_in_price = spread_points * point_size
    cost_in_pct = cost_in_price / df['close']
    
    # 5. Total cost for this specific hour
    total_costs = position_changes * cost_in_pct
    
    return gross_rets - total_costs

def apply_sl_tp_and_costs(df, positions, point_size, spread_points, sl_points=0, tp_points=0, trail_points=0,
                          slippage_points=0, swap_pct_per_night=0.0):
    """
    NumPy engine with Fixed SL, Take Profit and Trailing Stop.

    Per-bar order of events (signal = positions[i-1], decided at close[i-1], acted on at open[i]):
      1. If the target position differs from the held one (flip OR flat), exit AT THE OPEN.
      2. Otherwise check the stop/TP intrabar. Pessimistic rule: SL is checked before TP.
         Gaps: if the bar opens beyond the stop, the fill is the OPEN, not the stop price.
      3. If flat and not stopped this bar, enter at open +/- spread, with a same-candle SL/TP check.
         The new leg's open->close return is booked on the same bar (flips included).

    Stop lock: after a stop/TP exit the engine stays flat until the target signal CHANGES
    (no instant re-entry into the same signal, no spread churn).

    Spread convention: one spread is charged per ROUND TRIP (baked into entry_price). With bid-quoted
    data that is the correct cost: longs pay the ask in and exit at the bid, shorts the reverse.

    Optional stress parameters (both default to OFF, so results are identical to before):
      slippage_points    : adverse price move, in points, on every ENTRY and every market/stop EXIT.
                           Take-profit limit fills get no slippage.
      swap_pct_per_night : cost, as a fraction of price, per calendar night a position is held.
                           Charged per calendar day crossed, so a weekend counts 3 nights
                           (a close match to the broker's weekly swap total).
    """
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    signals = positions.values

    n = len(df)
    returns = np.zeros(n)

    in_trade = 0          # 1 long, -1 short, 0 flat
    blocked = 0           # direction locked out after a stop/TP exit
    entry_price = 0.0
    current_sl = 0.0
    current_tp = 0.0
    highest_seen = 0.0
    lowest_seen = 0.0

    sl_dist = sl_points * point_size
    tp_dist = tp_points * point_size
    trail_dist = trail_points * point_size
    spread_cost = spread_points * point_size
    slip = slippage_points * point_size

    if swap_pct_per_night > 0:
        days = df.index.values.astype('datetime64[D]')
        day_gap = np.r_[0, np.diff(days).astype('int64')]
    else:
        day_gap = None

    for i in range(1, n):
        sig = signals[i-1]
        prev_close = closes[i-1]
        growth = 1.0
        stopped_now = False

        # Release the lock as soon as the target signal differs from the stopped direction
        if sig != blocked:
            blocked = 0

        # Overnight swap on a position carried across one or more calendar nights
        if in_trade != 0 and day_gap is not None and day_gap[i] > 0:
            growth *= 1.0 - swap_pct_per_night * day_gap[i]

        # =====================================================
        # A. HOLDING A TRADE
        # =====================================================
        if in_trade != 0:
            d = in_trade

            if sig != d:
                # Strategy flipped or went flat: exit at the open (also covers gaps through the stop)
                fill = opens[i] - d * slip
                growth *= 1.0 + d * (fill - prev_close) / prev_close
                in_trade = 0
            else:
                if d == 1:
                    hit_sl = current_sl != -np.inf and lows[i] <= current_sl
                    hit_tp = current_tp != np.inf and highs[i] >= current_tp
                else:
                    hit_sl = current_sl != np.inf and highs[i] >= current_sl
                    hit_tp = current_tp != -np.inf and lows[i] <= current_tp

                if hit_sl:
                    # Gap-aware fill: long fills at min(open, stop), short at max(open, stop)
                    fill = (min(opens[i], current_sl) if d == 1 else max(opens[i], current_sl)) - d * slip
                    growth *= 1.0 + d * (fill - prev_close) / prev_close
                    in_trade = 0
                    blocked = d
                    stopped_now = True
                elif hit_tp:
                    fill = max(opens[i], current_tp) if d == 1 else min(opens[i], current_tp)
                    growth *= 1.0 + d * (fill - prev_close) / prev_close
                    in_trade = 0
                    blocked = d
                    stopped_now = True
                else:
                    growth *= 1.0 + d * (closes[i] - prev_close) / prev_close
                    # Update trailing stop only after surviving the candle
                    if trail_points > 0:
                        if d == 1:
                            highest_seen = max(highest_seen, highs[i])
                            current_sl = max(current_sl, highest_seen - trail_dist)
                        else:
                            lowest_seen = min(lowest_seen, lows[i])
                            current_sl = min(current_sl, lowest_seen + trail_dist)

        # =====================================================
        # B. FLAT -> ENTER NEW TRADE (also the 2nd leg of a flip)
        # =====================================================
        if in_trade == 0 and not stopped_now and sig != 0 and sig != blocked:
            d = int(sig)
            in_trade = d
            if d == 1:
                entry_price = opens[i] + spread_cost + slip
                highest_seen = entry_price
                lowest_seen = 0.0
                current_sl = entry_price - sl_dist if sl_points > 0 else (entry_price - trail_dist if trail_points > 0 else -np.inf)
                current_tp = entry_price + tp_dist if tp_points > 0 else np.inf
                hit_sl = current_sl != -np.inf and lows[i] <= current_sl
                hit_tp = current_tp != np.inf and highs[i] >= current_tp
            else:
                entry_price = opens[i] - spread_cost - slip
                lowest_seen = entry_price
                highest_seen = 0.0
                current_sl = entry_price + sl_dist if sl_points > 0 else (entry_price + trail_dist if trail_points > 0 else np.inf)
                current_tp = entry_price - tp_dist if tp_points > 0 else -np.inf
                hit_sl = current_sl != np.inf and highs[i] >= current_sl
                hit_tp = current_tp != -np.inf and lows[i] <= current_tp

            # Same-candle check (pessimistic: SL first). Entry is at the open, so no gap issue.
            if hit_sl:
                growth *= 1.0 + d * ((current_sl - d * slip) - entry_price) / entry_price
                in_trade = 0
                blocked = d
            elif hit_tp:
                growth *= 1.0 + d * (current_tp - entry_price) / entry_price
                in_trade = 0
                blocked = d
            else:
                growth *= 1.0 + d * (closes[i] - entry_price) / entry_price
                if trail_points > 0:
                    if d == 1:
                        highest_seen = max(highest_seen, highs[i])
                        current_sl = max(current_sl, highest_seen - trail_dist)
                    else:
                        lowest_seen = min(lowest_seen, lows[i])
                        current_sl = min(current_sl, lowest_seen + trail_dist)

        returns[i] = growth - 1.0

    return pd.Series(returns, index=df.index)

DEFAULT_COST_SCENARIOS = {
    "base":                  {},
    "spread x1.5":           {"spread_mult": 1.5},
    "spread x2":             {"spread_mult": 2.0},
    "spread x3":             {"spread_mult": 3.0},
    "slippage 0.5x spread":  {"slip_spread_frac": 0.5},
    "swap":                  {"swap": True},
    "harsh (x2 + slip + swap)": {"spread_mult": 2.0, "slip_spread_frac": 0.5, "swap": True},
}

def cost_stress(df, positions, point_size, spread_points, sl_points=0, tp_points=0, trail_points=0,
                swap_pct_per_night=0.0002, scenarios=None, mc_simulations=100):
    """
    Re-runs the SAME positions and stops under harsher trading costs and tabulates the damage.
    Strategy-agnostic: it only needs positions + stop distances in points.

    Scenario keys: spread_mult (x the broker spread), slip_spread_frac (slippage per fill as a
    fraction of the spread), swap (True -> use swap_pct_per_night).
    swap_pct_per_night: fraction of price per night held, 0.0002 = 0.02%/night. Look up your
    broker's swap for the symbol and set it; swaps on CFDs and shorts can be far larger than FX.
    A strategy that does not survive the 'spread x2' and 'harsh' rows is not robust to costs.
    """
    scenarios = scenarios or DEFAULT_COST_SCENARIOS
    rows = []
    for name, sc in scenarios.items():
        sp = spread_points * sc.get("spread_mult", 1.0)
        slip = spread_points * sc.get("slip_spread_frac", 0.0)
        swap = swap_pct_per_night if sc.get("swap") else 0.0
        net = apply_sl_tp_and_costs(df, positions, point_size, sp, sl_points=sl_points,
                                    tp_points=tp_points, trail_points=trail_points,
                                    slippage_points=slip, swap_pct_per_night=swap)
        m = calculate_ai_metrics(net, mc_simulations=mc_simulations)
        if m is None:
            rows.append({"scenario": name, "cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan,
                         "mc_worst5_dd": np.nan, "active_days": 0, "survives": False})
            continue
        rows.append({"scenario": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "max_dd": m["max_drawdown"], "mc_worst5_dd": m["mc_worst_5%_drawdown"],
                     "active_days": m["n_trades"], "survives": bool(m["cagr"] > 0)})
    return pd.DataFrame(rows)