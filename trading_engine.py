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

ORDER_COLUMNS = ("position", "entry_price", "sl_price", "size", "exit_at_close")

class _Layer:
    """One tranche of the open position. Prices are bid-chart prices; entry has the spread baked in."""
    __slots__ = ("entry", "w", "units", "sl", "tp", "tp1", "tp1_done", "be_done", "hi", "lo",
                 "bar", "frac", "init_dist", "lev", "fresh_limit")

def _as_list(x, n, name, fill):
    """Optional per-bar input (Series/array/None) -> plain Python list of length n."""
    if x is None:
        return None
    a = np.asarray(x.values if hasattr(x, "values") else x, dtype=float)
    if a.shape != (n,):
        raise ValueError(f"{name} must have one value per bar ({n}), got shape {a.shape}")
    return np.where(np.isnan(a), fill, a).tolist()

def _unpack_orders(df, positions, entry_prices, sl_prices, size, exit_at_close):
    """
    positions may be a Series (classic) or a DataFrame with a 'position' column and any of the optional
    columns entry_price / sl_price / size / exit_at_close. Explicit keyword arrays win over columns,
    but giving the same input both ways is an error.
    """
    if isinstance(positions, pd.DataFrame):
        unknown = set(positions.columns) - set(ORDER_COLUMNS)
        if "position" not in positions.columns or unknown:
            raise ValueError(f"orders DataFrame needs a 'position' column and only {ORDER_COLUMNS}; "
                             f"got {list(positions.columns)}")
        cols = {"entry_price": entry_prices, "sl_price": sl_prices, "size": size, "exit_at_close": exit_at_close}
        for c in cols:
            if c in positions.columns:
                if cols[c] is not None:
                    raise ValueError(f"'{c}' given both as a column and as a keyword argument")
                cols[c] = positions[c]
        entry_prices, sl_prices, size, exit_at_close = (cols["entry_price"], cols["sl_price"],
                                                       cols["size"], cols["exit_at_close"])
        positions = positions["position"]
    sig = np.asarray(positions.values if hasattr(positions, "values") else positions, dtype=float)
    if sig.shape != (len(df),):
        raise ValueError(f"positions must have one value per bar ({len(df)}), got shape {sig.shape}")
    if np.isnan(sig).any():
        raise ValueError("positions contain NaN; use 0 for flat")
    if (sig != np.round(sig)).any():
        raise ValueError("positions must be whole numbers: sign = direction, size = number of layers. "
                         "Use the 'size' input for fractional sizing")
    return sig, entry_prices, sl_prices, size, exit_at_close

def apply_sl_tp_and_costs(df, positions, point_size, spread_points, sl_points=0, tp_points=0, trail_points=0,
                          slippage_points=0, swap_pct_per_night=0.0,
                          entry_prices=None, max_layers=1,
                          be_trigger_points=0, be_offset_points=0, sl_prices=None, max_bars=0,
                          tp1_points=0, tp1_fraction=0.5, tp1_move_to_be=False, exit_at_close=None,
                          sizing="unit", size=None, risk_pct=0.01, max_leverage=None, leverage_action="reduce",
                          swap_long_pct_per_night=None, swap_short_pct_per_night=None,
                          return_trades=False):
    """
    NumPy/pure-Python bar engine: market or pending entries, pyramiding, fixed / trailing / breakeven /
    indicator stops, partial and final take-profit, time stops, session-close exits, fixed-weight or
    risk-based position sizing with a leverage cap, and direction-dependent overnight swap.
    Every new input defaults to OFF, so the classic call gives exactly the old results.

    INPUTS PER BAR (decided at close[t-1], acted on during bar t)
      positions      : Series of whole numbers. Sign = direction, |value| = number of layers wanted
                       (capped at max_layers). Or a DataFrame with a 'position' column plus optional
                       entry_price / sl_price / size / exit_at_close columns (same meaning as the kwargs).
      entry_prices   : pending-order price for NEW layers (NaN = market order at the open). MT5 rules:
                       long level below close[t-1] = buy limit, above = buy stop (mirror for shorts).
                       Re-priced every bar while unfilled; cancelled when the signal no longer asks for it.
      sl_prices      : strategy stop level per bar (NaN = none). Combined with the engine's own stops:
                       the tighter one is used. It may move in either direction (it is the strategy's line).
      size           : per-layer size multiplier read when a layer opens (NaN or <= 0 = do not open).
      exit_at_close  : bool per bar; True closes everything at THIS bar's close (and opens nothing on it).

    PER-BAR ORDER OF EVENTS (pessimistic wherever the bar's internal path is unknown)
      1. Overnight swap on held layers (per calendar day crossed, so a weekend counts 3 nights).
      2. Exits at the open: opposite/flat signal closes all; time stop (layer held max_bars bars);
         a smaller layer count closes the newest layers first.
      3. Intrabar, per held layer: stop first (gap-aware: an open beyond the stop fills at the open),
         then TP1 (partial), then the final TP. Limit fills (TP1/TP) get no slippage.
      4. New layers (if the signal wants more than are held): market at the open, or pending fill.
         Same-bar stop check. Same-bar targets only for market/stop entries (a limit fill comes after a
         move AGAINST the target, so a same-bar target touch may have happened before the fill).
      5. Stop changes (trail, breakeven, TP1 -> breakeven) take effect from the NEXT bar.
      6. exit_at_close closes everything at the close.

    Stop lock: after a stop / TP / time-stop / session exit the engine opens nothing until the signal
    value CHANGES (no instant re-entry into the same signal). Partial TP1 does not lock.

    Spread: one spread per layer round trip, baked into its entry price (correct for bid-quoted MT5 data).
    Pending levels are bid-chart levels; a long fills at level + spread (in MT5: place it at level + spread).

    SIZING (returns are fractions of a compounding equity curve)
      sizing="unit" : each layer has weight `size` (default 1 = 100% of equity), re-marked every bar.
                      With one layer and size 1 this is exactly the classic engine.
      sizing="risk" : fixed lots per layer, chosen so the INITIAL stop distance loses risk_pct * size of
                      equity at entry (spread included; gaps and slippage can lose more). Needs a stop
                      (sl_points, trail_points or sl_prices); an entry with no valid stop is skipped.
      max_leverage  : cap on total notional / equity when a layer opens. leverage_action="reduce"
                      shrinks the new layer to fit, "skip" does not open it. None = no cap.
      If equity reaches 0 everything is closed and the run stops (ruin).

    Other options: slippage_points (adverse, on entries and market/stop exits); swap_pct_per_night
    (cost as a fraction of notional per night, both directions); swap_long_pct_per_night /
    swap_short_pct_per_night override it per direction (positive = you pay, negative = you earn);
    be_trigger_points / be_offset_points (stop to entry + offset once the bar extreme is that far in
    profit); max_bars (time stop, 0 = off); tp1_points / tp1_fraction / tp1_move_to_be (scale out).

    Returns the per-bar net return Series, or (returns, trades DataFrame) with return_trades=True.
    One trades row per exit (a TP1 partial is its own row); reasons: signal, sl, tp1, tp, time, close
    (exit_at_close), ruin, end (still open on the last bar, marked at its close).
    """
    if sizing not in ("unit", "risk"):
        raise ValueError("sizing must be 'unit' or 'risk'")
    if leverage_action not in ("reduce", "skip"):
        raise ValueError("leverage_action must be 'reduce' or 'skip'")
    if not 0 < tp1_fraction <= 1:
        raise ValueError("tp1_fraction must be in (0, 1]")
    if int(max_layers) < 1:
        raise ValueError("max_layers must be >= 1")
    max_layers = int(max_layers)
    max_bars = int(max_bars)
    n = len(df)
    sig_arr, entry_prices, sl_prices, size, exit_at_close = _unpack_orders(
        df, positions, entry_prices, sl_prices, size, exit_at_close)
    if sizing == "risk" and sl_points <= 0 and trail_points <= 0 and sl_prices is None:
        raise ValueError("sizing='risk' needs a stop: sl_points, trail_points or sl_prices")

    opens = df['open'].values.tolist()
    highs = df['high'].values.tolist()
    lows = df['low'].values.tolist()
    closes = df['close'].values.tolist()
    signals = sig_arr.tolist()
    nan = float("nan")
    ep_l = _as_list(entry_prices, n, "entry_prices", nan)
    slp_l = _as_list(sl_prices, n, "sl_prices", nan)
    size_l = _as_list(size, n, "size", nan)
    eac_l = None if exit_at_close is None else [bool(v) for v in _as_list(exit_at_close, n, "exit_at_close", 0.0)]

    returns = np.zeros(n)
    inf = np.inf

    sl_dist = sl_points * point_size
    tp_dist = tp_points * point_size
    tp1_dist = tp1_points * point_size
    trail_dist = trail_points * point_size
    be_dist = be_trigger_points * point_size
    be_off = be_offset_points * point_size
    spread_cost = spread_points * point_size
    slip = slippage_points * point_size
    unit = sizing == "unit"

    swap_long = swap_pct_per_night if swap_long_pct_per_night is None else swap_long_pct_per_night
    swap_short = swap_pct_per_night if swap_short_pct_per_night is None else swap_short_pct_per_night
    if swap_long != 0 or swap_short != 0:
        days = df.index.values.astype('datetime64[D]')
        day_gap = np.r_[0, np.diff(days).astype('int64')].tolist()
    else:
        day_gap = None

    times = df.index
    trades = [] if return_trades else None

    def log(L, d, i, fill, frac, reason):
        trades.append((L.bar, i, d, L.entry, fill, frac, reason, i - L.bar,
                       d * (fill - L.entry) / L.entry,
                       d * (fill - L.entry) / L.init_dist if L.init_dist > 0 else nan, L.lev))

    E = 1.0               # equity (starts at 1, compounds)
    layers = []           # open layers, oldest first; all share direction `d`
    d = 0
    blocked = 0           # signal value locked out after a stop/TP/time/session exit

    for i in range(1, n):
        sig = signals[i-1]
        pc = closes[i-1]
        o = opens[i]; h = highs[i]; lo = lows[i]; c = closes[i]
        E_start = E
        stopped_now = False

        # Release the lock as soon as the target signal differs from the locked value
        if sig != blocked:
            blocked = 0

        sd = 1 if sig > 0 else (-1 if sig < 0 else 0)
        target = (1 if max_layers == 1 else min(int(abs(sig)), max_layers)) if sd != 0 else 0
        slp = slp_l[i-1] if slp_l is not None else nan

        if layers:
            # ---------- 1. swap; unit sizing re-marks each layer to its weight of current equity ----------
            if unit:
                wsum = 0.0
                for L in layers:
                    wsum += L.w * L.frac
                notional = wsum * E
            else:
                notional = 0.0
                for L in layers:
                    notional += L.units * pc
            if day_gap is not None and day_gap[i] > 0:
                E -= (swap_long if d == 1 else swap_short) * day_gap[i] * notional
            if unit:
                for L in layers:
                    L.units = L.w * L.frac * E / pc

            # ---------- 2. exits at the open ----------
            if sd != d:
                # Strategy flipped or went flat: exit at the open (also covers gaps through the stop)
                fill = o - d * slip
                for L in layers:
                    E += d * L.units * (fill - pc)
                    if trades is not None: log(L, d, i, fill, L.frac, "signal")
                layers = []
            else:
                if max_bars > 0:
                    keep = []
                    for L in layers:
                        if i - L.bar >= max_bars:
                            fill = o - d * slip
                            E += d * L.units * (fill - pc)
                            if trades is not None: log(L, d, i, fill, L.frac, "time")
                            blocked = sig
                            stopped_now = True
                        else:
                            keep.append(L)
                    layers = keep
                while len(layers) > target:
                    L = layers.pop()
                    fill = o - d * slip
                    E += d * L.units * (fill - pc)
                    if trades is not None: log(L, d, i, fill, L.frac, "signal")

            # ---------- 3. intrabar stops / targets on held layers ----------
            if layers:
                keep = []
                for L in layers:
                    sl = L.sl
                    if slp == slp:
                        sl = (slp if slp > sl else sl) if d == 1 else (slp if slp < sl else sl)
                    if d == 1:
                        hit_sl = sl != -inf and lo <= sl
                    else:
                        hit_sl = sl != inf and h >= sl
                    if hit_sl:
                        # Gap-aware fill: long fills at min(open, stop), short at max(open, stop)
                        fill = (min(o, sl) if d == 1 else max(o, sl)) - d * slip
                        E += d * L.units * (fill - pc)
                        if trades is not None: log(L, d, i, fill, L.frac, "sl")
                        blocked = sig
                        stopped_now = True
                        continue
                    if not L.tp1_done and L.tp1 is not None and (h >= L.tp1 if d == 1 else lo <= L.tp1):
                        fill = max(o, L.tp1) if d == 1 else min(o, L.tp1)
                        part = L.units * tp1_fraction
                        E += d * part * (fill - pc)
                        if trades is not None: log(L, d, i, fill, L.frac * tp1_fraction, "tp1")
                        L.units -= part
                        L.frac *= 1.0 - tp1_fraction
                        L.tp1_done = True
                        if L.frac <= 1e-12:
                            blocked = sig
                            stopped_now = True
                            continue
                    if d == 1:
                        hit_tp = L.tp != inf and h >= L.tp
                    else:
                        hit_tp = L.tp != -inf and lo <= L.tp
                    if hit_tp:
                        fill = max(o, L.tp) if d == 1 else min(o, L.tp)
                        E += d * L.units * (fill - pc)
                        if trades is not None: log(L, d, i, fill, L.frac, "tp")
                        blocked = sig
                        stopped_now = True
                        continue
                    keep.append(L)
                layers = keep
            if not layers:
                d = 0

        # ---------- 4. open new layers (also the 2nd leg of a flip, and pyramid adds) ----------
        if (not stopped_now and sd != 0 and sig != blocked and len(layers) < target
                and not (eac_l is not None and eac_l[i])):
            ep = ep_l[i-1] if ep_l is not None else nan
            raw = nan
            limit = False
            if ep != ep:                                         # market order at the open
                raw = o
            else:
                limit = ep < pc if sd == 1 else ep > pc
                if sd == 1:
                    if limit:
                        if lo <= ep: raw = min(o, ep)            # gap below a buy limit fills at the (better) open
                    elif h >= ep: raw = max(o, ep)               # gap above a buy stop fills at the (worse) open
                else:
                    if limit:
                        if h >= ep: raw = max(o, ep)
                    elif lo <= ep: raw = min(o, ep)
            mult = size_l[i-1] if size_l is not None else 1.0
            if raw == raw and mult == mult and mult > 0:
                entry = raw + sd * (spread_cost + (0.0 if limit else slip))
                # Mark held layers to the fill price to size the new ones on current equity
                E_mtm = E
                exposure = 0.0
                for L in layers:
                    E_mtm += sd * L.units * (raw - pc)
                    exposure += L.units * raw
                for _ in range(target - len(layers)):
                    if sd == 1:
                        sl0 = entry - sl_dist if sl_points > 0 else (entry - trail_dist if trail_points > 0 else -inf)
                        tp0 = entry + tp_dist if tp_points > 0 else inf
                        sl_eff = sl0 if slp != slp else max(sl0, slp)
                    else:
                        sl0 = entry + sl_dist if sl_points > 0 else (entry + trail_dist if trail_points > 0 else inf)
                        tp0 = entry - tp_dist if tp_points > 0 else -inf
                        sl_eff = sl0 if slp != slp else min(sl0, slp)
                    if slp == slp and sd * (entry - slp) <= 0:
                        break                                    # strategy stop on the wrong side: MT5 rejects it
                    init_dist = sd * (entry - sl_eff) if sl_eff != -inf and sl_eff != inf else 0.0
                    if unit:
                        units = mult * E_mtm / entry
                    else:
                        if init_dist <= 0:
                            break                                # no valid stop -> cannot size the trade
                        units = risk_pct * mult * E_mtm / init_dist
                    if max_leverage is not None and E_mtm > 0:
                        room = max_leverage * E_mtm - exposure
                        if units * entry > room:
                            if leverage_action == "skip" or room <= 0:
                                break
                            units = room / entry
                    L = _Layer()
                    L.entry = entry; L.sl = sl0; L.tp = tp0; L.bar = i; L.frac = 1.0
                    L.tp1 = (entry + sd * tp1_dist) if tp1_points > 0 else None
                    L.tp1_done = False; L.be_done = False
                    L.hi = entry if sd == 1 else 0.0
                    L.lo = entry if sd == -1 else 0.0
                    L.init_dist = init_dist
                    L.units = units
                    L.w = units * entry / E_mtm if E_mtm > 0 else 0.0
                    L.lev = L.w
                    L.fresh_limit = limit
                    exposure += units * entry
                    if not layers:
                        d = sd
                    layers.append(L)

        # ---------- book bar P&L for held layers, same-bar checks for new ones, update stops ----------
        if layers:
            keep = []
            for L in layers:
                fresh = L.bar == i
                p0 = L.entry if fresh else pc
                if fresh:
                    # Same-candle check (pessimistic: SL first). Fresh fills sit inside the bar, so no gap issue.
                    sl = L.sl
                    if slp == slp:
                        sl = (slp if slp > sl else sl) if d == 1 else (slp if slp < sl else sl)
                    if (d == 1 and sl != -inf and lo <= sl) or (d == -1 and sl != inf and h >= sl):
                        fill = sl - d * slip
                        E += d * L.units * (fill - p0)
                        if trades is not None: log(L, d, i, fill, L.frac, "sl")
                        blocked = sig
                        continue
                    if not L.fresh_limit:
                        if L.tp1 is not None and (h >= L.tp1 if d == 1 else lo <= L.tp1):
                            part = L.units * tp1_fraction
                            E += d * part * (L.tp1 - p0)
                            if trades is not None: log(L, d, i, L.tp1, L.frac * tp1_fraction, "tp1")
                            L.units -= part
                            L.frac *= 1.0 - tp1_fraction
                            L.tp1_done = True
                            if L.frac <= 1e-12:
                                blocked = sig
                                continue
                        if (d == 1 and L.tp != inf and h >= L.tp) or (d == -1 and L.tp != -inf and lo <= L.tp):
                            E += d * L.units * (L.tp - p0)
                            if trades is not None: log(L, d, i, L.tp, L.frac, "tp")
                            blocked = sig
                            continue
                E += d * L.units * (c - p0)
                # Stop updates only after surviving the candle
                if trail_points > 0:
                    if d == 1:
                        L.hi = max(L.hi, h)
                        L.sl = max(L.sl, L.hi - trail_dist)
                    else:
                        L.lo = min(L.lo, lo)
                        L.sl = min(L.sl, L.lo + trail_dist)
                if be_trigger_points > 0 and not L.be_done and (h - L.entry >= be_dist if d == 1 else L.entry - lo >= be_dist):
                    L.sl = max(L.sl, L.entry + be_off) if d == 1 else min(L.sl, L.entry - be_off)
                    L.be_done = True
                if tp1_move_to_be and L.tp1_done:
                    L.sl = max(L.sl, L.entry) if d == 1 else min(L.sl, L.entry)
                keep.append(L)
            layers = keep

            # ---------- 6. session / forced exit at this bar's close ----------
            if layers and eac_l is not None and eac_l[i]:
                fill = c - d * slip
                for L in layers:
                    E += d * L.units * (fill - c)
                    if trades is not None: log(L, d, i, fill, L.frac, "close")
                layers = []
                blocked = sig
            if not layers:
                d = 0

        # ---------- ruin ----------
        if E <= 0:
            for L in layers:
                if trades is not None: log(L, d, i, c, L.frac, "ruin")
            layers = []
            returns[i] = -1.0
            break

        returns[i] = E / E_start - 1.0

    ret = pd.Series(returns, index=df.index)
    if not return_trades:
        return ret
    for L in layers:                                     # still open at the last bar: marked at its close
        log(L, d, n - 1, closes[n - 1], L.frac, "end")
    tr = pd.DataFrame(trades, columns=["entry_bar", "exit_bar", "direction", "entry_price", "exit_price",
                                       "fraction", "reason", "bars_held", "price_return", "r_multiple",
                                       "leverage_at_entry"])
    tr.insert(0, "entry_time", times[tr["entry_bar"].values] if len(tr) else pd.DatetimeIndex([]))
    tr.insert(1, "exit_time", times[tr["exit_bar"].values] if len(tr) else pd.DatetimeIndex([]))
    return ret, tr.drop(columns=["entry_bar", "exit_bar"])

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
                swap_pct_per_night=0.0002, scenarios=None, mc_simulations=100,
                swap_long_pct_per_night=None, swap_short_pct_per_night=None, **engine_kw):
    """
    Re-runs the SAME positions and stops under harsher trading costs and tabulates the damage.
    Strategy-agnostic: it only needs positions (Series or orders DataFrame) + engine settings.

    Scenario keys: spread_mult (x the broker spread), slip_spread_frac (slippage per fill as a
    fraction of the spread), swap (True -> charge overnight swap).
    swap_pct_per_night: fraction of notional per night held, 0.0002 = 0.02%/night, both directions.
    swap_long_pct_per_night / swap_short_pct_per_night: per-direction values (positive = you pay);
    when given they replace swap_pct_per_night in the swap scenarios.
    engine_kw: every other engine setting (sizing, breakeven, time stop, layers, ...), applied to every
    scenario. Costs (slippage, swap) are set by the scenarios, so do not pass them here.
    A strategy that does not survive the 'spread x2' and 'harsh' rows is not robust to costs.
    """
    clash = {"slippage_points", "swap_pct_per_night", "swap_long_pct_per_night",
             "swap_short_pct_per_night", "return_trades"} & set(engine_kw)
    if clash:
        raise ValueError(f"cost_stress sets these itself, do not pass them in engine_kw: {sorted(clash)}")
    directional = swap_long_pct_per_night is not None or swap_short_pct_per_night is not None
    scenarios = scenarios or DEFAULT_COST_SCENARIOS
    rows = []
    for name, sc in scenarios.items():
        sp = spread_points * sc.get("spread_mult", 1.0)
        slip = spread_points * sc.get("slip_spread_frac", 0.0)
        on = bool(sc.get("swap"))
        if directional:
            swap_kw = {"swap_long_pct_per_night": (swap_long_pct_per_night or 0.0) if on else 0.0,
                       "swap_short_pct_per_night": (swap_short_pct_per_night or 0.0) if on else 0.0}
        else:
            swap_kw = {"swap_pct_per_night": swap_pct_per_night if on else 0.0}
        net = apply_sl_tp_and_costs(df, positions, point_size, sp, sl_points=sl_points,
                                    tp_points=tp_points, trail_points=trail_points,
                                    slippage_points=slip, **swap_kw, **engine_kw)
        m = calculate_ai_metrics(net, mc_simulations=mc_simulations)
        if m is None:
            rows.append({"scenario": name, "cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan,
                         "mc_worst5_dd": np.nan, "active_days": 0, "survives": False})
            continue
        rows.append({"scenario": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "max_dd": m["max_drawdown"], "mc_worst5_dd": m["mc_worst_5%_drawdown"],
                     "active_days": m["n_trades"], "survives": bool(m["cagr"] > 0)})
    return pd.DataFrame(rows)
